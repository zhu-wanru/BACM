import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv
import functools
from util.loss_utils import lovasz_softmax_with_logit
from lib.pointgroup_ops.functions import pointgroup_ops
from lib.pointops2.functions import pointops2 as pointops
from .unet_block import ResidualBlock, VGGBlock, UBlock
from dataset.realistic_projection import Realistic_Projection

class SparseConvNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        in_channel = cfg.MODEL.BACKBONE.in_channel
        mid_channel = cfg.MODEL.BACKBONE.mid_channel
        try:
            n_classes = cfg.COMMON_CLASSES.n_classes
        except:
            n_classes = cfg.DATA_CONFIG.DATA_CLASS.n_classes
        block_reps = cfg.MODEL.BACKBONE.block_reps
        block_residual = cfg.MODEL.BACKBONE.block_residual

        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)
        if block_residual:
            block = ResidualBlock
        else:
            block = VGGBlock

        # backbone
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(in_channel, mid_channel, kernel_size=3, padding=1, bias=False, indice_key='subm1')
        )

        self.unet = UBlock([mid_channel, 2 * mid_channel, 3 * mid_channel, 4 * mid_channel, 5 * mid_channel,
                            6 * mid_channel, 7 * mid_channel], norm_fn, block_reps, block, indice_key_id=1)

        self.output_layer = spconv.SparseSequential(
            norm_fn(mid_channel),
            nn.ReLU()
        )
        self.linear = nn.Linear(mid_channel, n_classes)  
        self.linear_kl = nn.Linear(mid_channel, n_classes) 

        # init parameters
        self.apply(self.set_bn_init)

    @staticmethod
    def set_bn_init(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(0.0)

    def forward(self, input, input_map, mode):
        output = self.input_conv(input)
        output = self.unet(output)
        output = self.output_layer(output)
        output_feats = output.features[input_map.long()]  # voxel to points

        pc_output = self.linear(output_feats)  # (N, nClass), float
        pc_output_kl = self.linear_kl(output_feats)  # (N, nClass), float

        if mode == 'src':
            return pc_output, pc_output_kl
        else:
            return pc_output.detach(), pc_output_kl


def select_prompt_points(pc2imgs_confs, K, step, B, W, H, device):
    patches = pc2imgs_confs.unfold(1, step, step).unfold(2, step, step)
    patches_flat = patches.reshape(B, -1, step * step)
    _, max_indices = patches_flat.max(dim=2) # max_indices: (B, W/step * H/step)
    max_x = max_indices // step
    max_y = max_indices % step
    start_x = torch.arange(0, W, step).unsqueeze(1).unsqueeze(0).repeat(B, 1, H // step).to(device)
    start_y = torch.arange(0, H, step).unsqueeze(0).unsqueeze(0).repeat(B, W // step, 1).to(device)
    max_pos_x = start_x + max_x.view(B, W // step, H // step)
    max_pos_y = start_y + max_y.view(B, W // step, H // step)
    points = torch.cat([max_pos_y.view(B, -1, 1), max_pos_x.view(B, -1, 1)], dim=2) #(B, N, 2)
    if K == None:
        return points
    batch_indices = torch.arange(B, device=device).view(B, 1).expand(B, points.shape[1])
    points_confs = pc2imgs_confs[batch_indices, points[:, :, 1], points[:, :, 0]].squeeze(-1)
    _, indices = torch.topk(points_confs, K, dim=1)
    points = torch.gather(points, dim=1, index=indices.unsqueeze(-1).expand(-1, -1, 2))
    return points

def test_model_feat(cfg, batch_size, batch, model, sam, epoch, step, mode, beta):
    voxel_coords = batch['voxel_locs'].cuda(non_blocking=True)  # (M, 1 + 3), long, cuda
    p2v_map = batch['p2v_map'].cuda(non_blocking=True)          # (N), int, cuda
    v2p_map = batch['v2p_map'].cuda(non_blocking=True)          # (M, 1 + maxActive), int, cuda
    offsets = batch['offsets'].cuda(non_blocking=True)
    labels = batch['labels'].squeeze(-1).long().cuda(non_blocking=True)

    locs = batch['locs'].cuda(non_blocking=True)

    coords_float = batch['locs_float'].cuda(non_blocking=True)  # (N, 3), float32, cuda
    feats_pc = batch['feats'].cuda(non_blocking=True)              # (N, C), float32, cuda
    spatial_shape = batch['spatial_shape']

    if cfg.MODEL.BACKBONE.use_xyz:
        feats = torch.cat((feats_pc, coords_float), 1)
    voxel_feats = pointgroup_ops.voxelization(feats_pc, v2p_map, cfg.DATA_CONFIG.DATA_PROCESSOR.voxel_mode)
    # (M, C), float, cuda

    input_ = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)
    pc_output, pc_output_kl = model(input_, p2v_map, mode)   # (N, nClass), float

    pc_views = Realistic_Projection(trgt_data=cfg.DATA_CONFIG.TRGT_DATA, num_classes=cfg.COMMON_CLASSES.n_classes)
    get_img = pc_views.get_img
    # img: (B, W, H, 3), img_labels: (B, W, H, 1), img_logits: (B, W, H, 11)

    V = 8
    img_size = 256
    imgs = torch.zeros([batch_size*V, img_size, img_size, 3], device=coords_float.device)
    imgs_labels = torch.zeros([batch_size*V, img_size, img_size, 1], device=coords_float.device)
    pc2imgs_logits = torch.zeros([batch_size*V, img_size, img_size, cfg.COMMON_CLASSES.n_classes], device=coords_float.device)
    keep_idxs = []
    xs = []
    ys = []
    for b in range(batch_size):
        xyz_coords_float = coords_float[offsets[b]: offsets[b+1]].unsqueeze(0)
        xyz_labels = labels[offsets[b]: offsets[b+1]].unsqueeze(0)
        xyz_pc_output = pc_output[offsets[b]: offsets[b+1]].unsqueeze(0)
        img, img_labels, pc2img_logits, keep_idx, x, y = get_img(xyz_coords_float, xyz_labels, xyz_pc_output)
        imgs[b*V:(b+1)*V] = img
        imgs_labels[b*V:(b+1)*V] = img_labels
        pc2imgs_logits[b*V:(b+1)*V] = pc2img_logits
        keep_idxs.append(keep_idx)
        xs.append(x)
        ys.append(y)
    
    device = imgs.device
    B, W, H, _ = imgs.shape
    sfm = torch.nn.Softmax(dim=3)
    
    pc2imgs_probs = sfm((pc2imgs_logits).detach()) 
    pc2imgs_confs, pc2imgs_preds = pc2imgs_probs.max(3) 
    pc2imgs_preds[pc2imgs_confs < 0.091] = 255

    points = select_prompt_points(pc2imgs_confs, None, step, B, W, H, device)

    batch_indices = torch.arange(B, device=coords_float.device).view(B, 1).expand(B, points.shape[1])  
    points_label = pc2imgs_preds[batch_indices, points[:, :, 1], points[:, :, 0]].squeeze(-1)
    points_label_gt = imgs_labels[batch_indices, points[:, :, 1], points[:, :, 0]].squeeze(-1)

    idx = torch.where(points_label.view(-1) != 255)[0]
    acc = len(torch.where(points_label.view(-1)[idx] == points_label_gt.view(-1)[idx])[0])/len(idx)

    img_output, img_output_kl = sam(imgs, points, points_label, mode)
    img_preds = img_output.max(1)[1] 
    pc_preds = pc_output.max(1)[1]

    return img_output, img_output_kl, pc_output, pc_output_kl, pc_preds, imgs_labels, pc2imgs_logits, keep_idxs, xs, ys, acc


def model_fn_decorator(cfg, batch_size, test=False):

    # criterion
    ignore_label = cfg.DATA_CONFIG.DATA_CLASS.ignore_label
    n_classes = cfg.COMMON_CLASSES.n_classes
    if cfg.OPTIMIZATION.get('loss', 'cross_entropy') == 'cross_entropy':
        semantic_criterion = nn.CrossEntropyLoss(ignore_index=ignore_label)
        semantic_criterion_weight = nn.CrossEntropyLoss(ignore_index=ignore_label, reduction='none')
    elif cfg.OPTIMIZATION.get('loss', 'cross_entropy') == 'lovasz':
        semantic_criterion = lovasz_softmax_with_logit(ignore=ignore_label)
    else:
        raise NotImplementedError


    def model_fn(batch, model, sam, epoch, step, alpha, beta, mode = 'src', soft_label=False, loss_weight=None):
        # prepare input
        # batch {'locs': locs, 'voxel_locs': voxel_locs, 'p2v_map': p2v_map, 'v2p_map': v2p_map,
        # 'locs_float': locs_float, 'feats': feats, 'labels': labels,
        # 'offsets': batch_offsets, 'spatial_shape': spatial_shape, 'id': id}

        batch_size = batch['offsets'].size(0) - 1
        output, output_kl, pc_output, pc_output_kl, pc_preds, img_labels, pc2imgs_logits, keep_idx, xs, ys, acc = test_model_feat(cfg, batch_size, batch, model, sam, epoch, step, mode, beta)
        labels = batch['labels'].squeeze(-1).long().cuda(non_blocking=True)
        device = labels.device
        img_preds = output.max(1)[1] 
        img_labels = img_labels.squeeze(-1) #(B, W, H)

        #### img to pc
        img2pc_logits = []
        img2pc_logits_kl = []

        for i in range(batch_size):
            V, P = keep_idx[i].shape
            img2pc_pred = torch.zeros([P, V, cfg.COMMON_CLASSES.n_classes]).to(device)
            img2pc_pred_kl = torch.zeros([P, V, cfg.COMMON_CLASSES.n_classes]).to(device)
            for v in range(V):
                img2pc_pred[keep_idx[i][v].long(), v, :] = output[i*V+v, :, xs[i][v].long(), ys[i][v].long()].permute(1, 0) # xs (B, N) pred_points (N, 11)
                img2pc_pred_kl[keep_idx[i][v].long(), v, :] = output_kl[i*V+v, :, xs[i][v].long(), ys[i][v].long()].permute(1, 0)
            img2pc_logits.append(img2pc_pred.sum(dim=1))
            img2pc_logits_kl.append(img2pc_pred_kl.sum(dim=1))
        img2pc_logits = torch.cat(img2pc_logits, 0) #(N, 11)
        img2pc_preds = img2pc_logits.argmax(dim = 1) #(N)
        img2pc_logits_kl = torch.cat(img2pc_logits_kl, 0) #(N, 11)

        all_preds = (img2pc_logits + pc_output).max(1)[1] 
        T_img = beta
        # compute loss
        if mode == 'src':
            loss_img = semantic_criterion(output, img_labels.long())
            loss_img2pc = semantic_criterion(img2pc_logits, labels)
            loss_pc = semantic_criterion(pc_output, labels)
            

            criterion_aign = nn.KLDivLoss(reduction="batchmean", log_target=True)
            loss_align_pc = criterion_aign(F.log_softmax(pc_output_kl, dim=1), F.log_softmax(img2pc_logits.detach(), dim=1))
            loss_align_img2pc = criterion_aign(F.log_softmax(img2pc_logits_kl, dim=1), F.log_softmax(pc_output.detach(), dim=1))

            criterion_aign_img = nn.KLDivLoss(reduction="none", log_target=True)
            loss_align_img = criterion_aign_img(F.log_softmax(output_kl.permute(0, 2, 3, 1) / T_img, dim=3), F.log_softmax((pc2imgs_logits.detach() / T_img), dim=3))
            loss_align_img = (loss_align_img * (pc2imgs_logits != 0)).sum() / len(torch.where(pc2imgs_logits != 0)[0])

            loss_all = loss_img + loss_pc + alpha * loss_img2pc + loss_align_pc + loss_align_img + alpha * loss_align_img2pc
        else:

            criterion_aign = nn.KLDivLoss(reduction="batchmean", log_target=True)
            loss_align_pc = criterion_aign(F.log_softmax(pc_output_kl, dim=1), F.log_softmax(img2pc_logits.detach(), dim=1))
            loss_align_img2pc = criterion_aign(F.log_softmax(img2pc_logits_kl, dim=1), F.log_softmax(pc_output.detach(), dim=1))

            criterion_aign_img = nn.KLDivLoss(reduction="none", log_target=True)
            loss_align_img = criterion_aign_img(F.log_softmax(output_kl.permute(0, 2, 3, 1) / T_img, dim=3), F.log_softmax((pc2imgs_logits.detach() / T_img), dim=3))
            loss_align_img = (loss_align_img * (pc2imgs_logits != 0)).sum() / len(torch.where(pc2imgs_logits != 0)[0])

            loss_all = loss_align_pc + loss_align_img + alpha * loss_align_img2pc
        ret = {'loss': loss_all, 'output': output, 'all_preds': all_preds, 'img_labels': img_labels, 'img_preds': img_preds, 'img2pc_preds': img2pc_preds, 'pc_labels': labels, 'pc_output': pc_output, 'pc_preds': pc_preds, 'acc': acc}
        return ret

    return model_fn
