import _init_path
import open3d as o3d
import os
import time
import random
import datetime
import numpy as np
import argparse
import subprocess
import glob
from tqdm import tqdm
from pathlib import Path
from tensorboardX import SummaryWriter
from functools import reduce

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.distributed as dist
import torch.optim
import torch.utils.data
from torch.optim.lr_scheduler import CosineAnnealingLR

import sys
sys.path.append('..')
from dataset import get_dataset
from util import common_utils, model_utils_st
from util.common_utils import DataReader, AverageMeter, update_meter, calc_metrics, get_logger
from util.model_utils_st import load_params_from_pretrain, load_params_from_ckpt, \
    load_metric_from_ckpt, save_params
from util.pseudo_labels_util_sam import set_pseudo_labels
from util.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from model.dsnorm import DSNorm, set_ds_source, set_ds_target
from segment_anything import SamPredictor, sam_model_registry


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--exp_name', type=str, default='da', help='experiment name')
    parser.add_argument('--src_dataset', type=str, default='front3d', help='source dataset')
    parser.add_argument('--trgt_dataset', type=str, default='s3dis', help='target dataset')
    parser.add_argument('--sam_path', type=str, default='../pretrained_model/sam_vit_b_01ec64.pth',help='pretrain sam path')
    parser.add_argument('--weight_pc', type=str, default=None,help='pc weight path')
    parser.add_argument('--weight_img', type=str, default=None,help='img weight path')
    parser.add_argument('--batch_size', type=int, default=None, required=False,
                        help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for pretrained experiment')
    parser.add_argument('--st_extra_tag', type=str, default='st', help='extra tag for this experiment')
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--resume', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--weight_ema', type=str, default=None, help='pretrained_model_2')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18867, help='tcp port for distrbuted training')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--reserve_old_ckpt', action='store_true', default=False,
                        help='whether to remove previously saved ckpt')
    parser.add_argument('--manual_seed', type=int, default=None, help='')
    parser.add_argument('--ckpt_save_freq', type=int, default=1, help='number of training epochs')
    parser.add_argument('--print_freq', type=int, default=10, help='printing log frequency')
    parser.add_argument('--pseudo_labels_freq', type=int, default=5, help='pseudo labels saving frequency')
    parser.add_argument('--preserve_pseudo_labels', action='store_true', default=False, help='pseudo labels')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--step', type=int, default=16, help='img grid step')
    parser.add_argument('--max_ckpt_save_num', type=int, default=30, help='max number of saved checkpoint')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--pin_memory', action='store_true', default=False, help='')

    args = parser.parse_args()

    args.cfg_file = '../cfgs/da_'+args.src_dataset+'_'+args.trgt_dataset+'/spconv_st.yaml'
    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = args.exp_name+'_'+args.src_dataset+'_'+args.trgt_dataset


    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)
    return args, cfg


def worker_init_fn(worker_id):
    random.seed(args.manual_seed + worker_id)


def update_split_sampler(train_loader, tar_tail_splits, tar_splits_class_ratio, num_c, rank, dist_train=False):
    def flatten(l):
        return [jj for ii in l for jj in ii]
    if dist_train:
        tar_tail_splits_total = reduce(lambda x, y: x + y, common_utils.all_gather_object(tar_tail_splits, rank))
        tar_splits_class_ratio = torch.from_numpy(tar_splits_class_ratio).cuda()
        dist.all_reduce(tar_splits_class_ratio)
        tar_splits_class_ratio = tar_splits_class_ratio.cpu().numpy()
    else:
        tar_tail_splits_total = tar_tail_splits
    # update sampler
    tar_tail_splits_total = [flatten(tar_tail_splits_total[i::num_c]) for i in range(num_c)]
    train_loader.dataset.dataset1.augmentor.split_sampler.update(tar_tail_splits_total)
    # update sampler class ratio
    if cfg.DATA_CONFIG_TAR.DATA_AUG.tacm.cuboid_queue.update_class_ratio:
        train_loader.dataset.dataset1.augmentor.split_sampler.update_class_ratio(tar_splits_class_ratio)


def train_epoch(source_reader, train_loader, model, sam, model_fn, optimizer_pc, optimizer_img, epoch, rank, dist_train):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    intersection_x_meter = AverageMeter()
    union_x_meter = AverageMeter()
    target_x_meter = AverageMeter()
    loss_meter = AverageMeter()
    loss_x_meter = AverageMeter()
    loss_u_meter = AverageMeter()
    intersection_u_meter = AverageMeter()
    union_u_meter = AverageMeter()
    target_u_meter = AverageMeter()

    model.train()
    sam.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)
    logger.info('lr: {:.7f} '.format(optimizer_pc.param_groups[0]['lr']))

    train_loader.dataset.set_training_mode(True)
    for i, batch in enumerate(train_loader):
        if (i + 1) == len(train_loader):  # manually drop last samples
            continue

        try:
            cur_lr = float(optimizer_pc.lr)
        except:
            cur_lr = optimizer_pc.param_groups[0]['lr']

        data_time.update(time.time() - end)

        # adjust learning rate
        common_utils.adjust_lr(cfg.OPTIMIZATION_PC, optimizer_pc, args.epochs, len(train_loader), epoch, i)
        optimizer_pc.zero_grad()
        optimizer_img.zero_grad()

        # source forward
        accuracy_x = 0.0
        loss = 0.0
        if cfg.SELF_TRAIN.SRC.use_data:
            if cfg.MODEL.get('dsnorm', False):
                model.apply(set_ds_source)
            source_batch = source_reader.read_data()
            ret = model_fn(source_batch, model, sam, epoch, args.step)
            preds_x, labels_x, loss_x = ret['all_preds'], ret['pc_labels'], ret['loss']

            loss_x *= cfg.SELF_TRAIN.SRC.get('loss_weight', 1.0)
            loss_x.backward()

            # update loss
            if dist_train:
                n = preds_x.size(0)
                loss_x *= n
                count = labels_x.new_tensor([n], dtype=torch.long).cuda()
                dist.all_reduce(loss_x), dist.all_reduce(count)
                n = count.item()
                loss_x /= n
            loss_x_meter.update(loss_x.item(), labels_x.size(0))
            loss += loss_x.item()

            # update meter
            intersection_x_meter, union_x_meter, target_x_meter, accuracy_x, intersection_x, union_x, target_x = \
                update_meter(intersection_x_meter, union_x_meter, target_x_meter, preds_x, labels_x, 
                cfg.COMMON_CLASSES.n_classes, cfg.DATA_CONFIG.DATA_CLASS.ignore_label, dist_train)

        # target forward
        if cfg.MODEL.get('dsnorm', False):
            model.apply(set_ds_target)
        ret = model_fn(batch, model, sam, epoch, args.step)
        logit_u, preds_u, labels_u, loss_u = ret['output'], ret['all_preds'], ret['pc_labels'], ret['loss']
        loss_u *= cfg.SELF_TRAIN.TAR.get('loss_weight', 1.0)
        loss_u.backward()

        # update loss
        if dist_train:
            n = labels_u.size(0)
            loss_u *= n
            count = labels_u.new_tensor([n], dtype=torch.long).cuda()
            dist.all_reduce(loss_u)
            dist.all_reduce(count)
            n = count.item()
            loss_u /= n
        loss_u_meter.update(loss_u.item(), labels_u.size(0))
        loss += loss_u.item()

        # update meter
        intersection_u_meter, union_u_meter, target_u_meter, accuracy_u, intersection_u, union_u, target_u = \
            update_meter(intersection_u_meter, union_u_meter, target_u_meter, preds_u, labels_u, 
            cfg.COMMON_CLASSES.n_classes, cfg.DATA_CONFIG_TAR.DATA_CLASS.ignore_label, dist_train)

        # update split queue
        split_sampler_cfg = cfg.DATA_CONFIG_TAR.DATA_AUG.tacm.cuboid_queue
        if split_sampler_cfg.enabled:
            update_split_sampler(
                train_loader, batch['tar_tail_splits'], batch['tar_splits_class_ratio'],
                split_sampler_cfg.num_class, rank, dist_train=dist_train
            )

        # backward
        if cfg.OPTIMIZATION_PC.get('clip_grad', False) and cfg.OPTIMIZATION_PC.clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
        optimizer_pc.step()
        optimizer_img.step()
        loss_meter.update(loss, labels_u.size(0))
        torch.cuda.empty_cache()

        # update time and print log
        batch_time.update(time.time() - end)
        end = time.time()
        current_iter = epoch * len(train_loader) + i + 1
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m),
                                                    int(t_s))
        if (i + 1) % args.print_freq == 0 or i == len(train_loader) - 1:
            logger.info(
                'Epoch: [{}/{}][{}/{}] '
                'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                'Remain {remain_time} '
                'Loss {loss_meter.val:.4f} '
                'Loss_x {loss_x_meter.val:.4f} '
                'Loss_u {loss_u_meter.val:.4f} '
                'SrcAccuracy {accuracy_x:.4f} '
                'TarAccuracy {accuracy_u:.4f}. '.format(
                                                        epoch + 1,
                                                        args.epochs,
                                                        i + 1,
                                                        len(train_loader),
                                                        batch_time=batch_time,
                                                        data_time=data_time,
                                                        remain_time=remain_time,
                                                        loss_meter=loss_meter,
                                                        loss_x_meter=loss_x_meter,
                                                        loss_u_meter=loss_u_meter,
                                                        accuracy_x=accuracy_x,
                                                        accuracy_u=accuracy_u))

        # record to writer
        if rank == 0:
            if cfg.SELF_TRAIN.SRC.use_data:
                writer.add_scalar('loss_x_train_batch', loss_x_meter.val, current_iter)
                writer.add_scalar('mIoU_x_train_batch', np.mean(intersection_x / (union_x + 1e-10)), current_iter)
                writer.add_scalar('mAcc_x_train_batch', np.mean(intersection_x / (target_x + 1e-10)), current_iter)
                writer.add_scalar('allAcc_x_train_batch', accuracy_x, current_iter)
            writer.add_scalar('loss_train_batch', loss_meter.val, current_iter)
            writer.add_scalar('loss_u_train_batch', loss_u_meter.val, current_iter)
            writer.add_scalar('mIoU_u_train_batch', np.mean(intersection_u / (union_u + 1e-10)), current_iter)
            writer.add_scalar('mAcc_u_train_batch', np.mean(intersection_u / (target_u + 1e-10)), current_iter)
            writer.add_scalar('allAcc_u_train_batch', accuracy_u, current_iter)
            writer.add_scalar('lr', cur_lr, current_iter)

    if cfg.SELF_TRAIN.SRC.use_data:
        mIoU_x, mAcc_x, allAcc_x, _, _ = calc_metrics(intersection_x_meter, union_x_meter, target_x_meter)
    else:
        mIoU_x, mAcc_x, allAcc_x = 0.0, 0.0, 0.0
    mIoU_u, mAcc_u, allAcc_u, _, _ = calc_metrics(intersection_u_meter, union_u_meter, target_u_meter)
    logger.info(
        'Train result at epoch [{}/{}]: Src mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}, Tar mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'
        .format(epoch + 1, args.epochs, mIoU_x, mAcc_x, allAcc_x, mIoU_u,
                mAcc_u, allAcc_u))
    if rank == 0:
        if cfg.SELF_TRAIN.SRC.use_data:
            writer.add_scalar('loss_x_train', loss_x_meter.avg, epoch + 1)
            writer.add_scalar('mIoU_train', mIoU_x, epoch + 1)
            writer.add_scalar('mAcc_train', mAcc_x, epoch + 1)
            writer.add_scalar('allAcc_train', allAcc_x, epoch + 1)
        writer.add_scalar('loss_train', loss_meter.avg, epoch + 1)
        writer.add_scalar('loss_train_x', loss_x_meter.avg, epoch + 1)
        writer.add_scalar('loss_train_u', loss_u_meter.avg, epoch + 1)
        writer.add_scalar('mIoU_train', mIoU_u, epoch + 1)
        writer.add_scalar('mAcc_train', mAcc_u, epoch + 1)
        writer.add_scalar('allAcc_train', allAcc_u, epoch + 1)
    return loss_meter.avg, mIoU_u, mAcc_u, allAcc_u


def validate_epoch(val_loader, model, sam, model_fn, epoch, rank, dist_train):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    img2pc_intersection_meter = AverageMeter()
    img2pc_union_meter = AverageMeter()
    img2pc_target_meter = AverageMeter()

    pc_intersection_meter = AverageMeter()
    pc_union_meter = AverageMeter()
    pc_target_meter = AverageMeter()


    model.eval()
    sam.eval()
    end = time.time()
    accs = 0.0
    if cfg.MODEL.get('dsnorm', False):
        model.apply(set_ds_target)
    for i, batch in enumerate(val_loader):
        data_time.update(time.time() - end)

        # forward
        with torch.no_grad():
            ret = model_fn(batch, model, sam, epoch, args.step)
            loss = ret['loss']
            pc_labels = ret['pc_labels']
            img2pc_preds = ret['img2pc_preds']
            pc_preds = ret['pc_preds']
            preds = ret['all_preds']
            acc = ret['acc']
            accs += acc

        # update loss
        if dist_train:
            n = preds.size(0)
            loss *= n
            count = pc_labels.new_tensor([n], dtype=torch.long).cuda()
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            loss /= n
        loss_meter.update(loss.item(), pc_labels.size(0))
        # update meter
        img2pc_intersection_meter, img2pc_union_meter, img2pc_target_meter, img2pc_accuracy, img2pc_intersection, img2pc_union, img2pc_target = \
            update_meter(img2pc_intersection_meter, img2pc_union_meter, img2pc_target_meter, img2pc_preds, pc_labels, 
            cfg.COMMON_CLASSES.n_classes, cfg.DATA_CONFIG.DATA_CLASS.ignore_label, dist_train)
        pc_intersection_meter, pc_union_meter, pc_target_meter, pc_accuracy, pc_intersection, pc_union, pc_target = \
            update_meter(pc_intersection_meter, pc_union_meter, pc_target_meter, pc_preds, pc_labels, 
            cfg.COMMON_CLASSES.n_classes, cfg.DATA_CONFIG.DATA_CLASS.ignore_label, dist_train)
        intersection_meter, union_meter, target_meter, accuracy, _, _, _ = \
            update_meter(intersection_meter, union_meter, target_meter, preds, pc_labels,
            cfg.COMMON_CLASSES.n_classes, cfg.DATA_CONFIG_TAR.DATA_CLASS.ignore_label, dist_train)

        # update time and print log
        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) % args.print_freq == 0:
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                        'Accuracy img2pc {img2pc:.4f} pc {pc:.4f} prompt {prompt:.4f} all {all:.4f}.'.format(
                            i + 1,
                            len(val_loader),
                            data_time=data_time,
                            batch_time=batch_time,
                            loss_meter=loss_meter,
                            img2pc=img2pc_accuracy,
                            pc=pc_accuracy,
                            prompt=accs/(i+1),
                            all=accuracy))

    img2pc_mIoU, img2pc_mAcc, img2pc_allAcc, img2pc_iou_class, img2pc_accuracy_class = \
        calc_metrics(img2pc_intersection_meter, img2pc_union_meter, img2pc_target_meter)
    pc_mIoU, pc_mAcc, pc_allAcc, pc_iou_class, pc_accuracy_class = \
        calc_metrics(pc_intersection_meter, pc_union_meter, pc_target_meter)
    mIoU, mAcc, allAcc, iou_class, accuracy_class = calc_metrics(intersection_meter, union_meter, target_meter)
    logger.info('Img2pc val result: mIoU/mPre/mAcc/allPre/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(img2pc_mIoU, img2pc_mAcc, img2pc_allAcc))
    logger.info('Pc val result: mIoU/mPre/mAcc/allPre/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(pc_mIoU, pc_mAcc, pc_allAcc))
    logger.info('All val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    n_classes = cfg.COMMON_CLASSES.n_classes
    class_names = cfg.COMMON_CLASSES.class_names
    for i in range(n_classes):
        logger.info('pc Class {} : iou/accuracy {:.4f}/{:.4f}.'.format(
            class_names[i], pc_iou_class[i], pc_accuracy_class[i]))
    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
    if rank == 0:
        writer.add_scalar('loss_val', loss_meter.avg, epoch + 1)
        writer.add_scalar('mIoU_val', mIoU, epoch + 1)
        writer.add_scalar('mAcc_val', mAcc, epoch + 1)
        writer.add_scalar('allAcc_val', allAcc, epoch + 1)
    return pc_mIoU


def train(model, sam, model_fn, test_model_fn, source_reader, train_loader, val_loader, optimizer_pc, optimizer_img, scheduler, 
          ckpt_dir, pseudo_labels_dir, rank, dist_train=False, train_sampler=None,
          best_mIoU=None, best_epoch=0):

    best_mIoU = best_mIoU if best_mIoU is not None else 0.0
    validate_epoch(val_loader, model, sam, test_model_fn, args.start_epoch, rank, dist_train)

    logger.info('thres: {}'.format(cfg.SELF_TRAIN.thres) if cfg.SELF_TRAIN.global_thres else
                    'thres ratio: {}'.format(cfg.SELF_TRAIN.thres_ratio))
    for epoch in range(args.start_epoch, args.epochs):
        if train_sampler is not None:  # compatible for pytorch1.1
            train_sampler.set_epoch(epoch)
            source_reader.set_cur_epoch(epoch)

        generate = set_pseudo_labels(
            args, cfg, logger, pseudo_labels_dir, train_loader, model, sam, test_model_fn, epoch, args.step, 
            rank=rank, dist=dist_train
        )
        split_sampler_cfg = cfg.DATA_CONFIG_TAR.DATA_AUG.tacm.cuboid_queue
        if split_sampler_cfg.enabled and \
            (generate or (not train_loader.dataset.dataset1.augmentor.split_sampler.init_finish())):
            class_ratio = np.loadtxt(str(pseudo_labels_dir / 'class_ratio.txt'))
            split_sampler_cfg = cfg.DATA_CONFIG_TAR.DATA_AUG.tacm.cuboid_queue
            split_sampler_cfg.class_ratio = class_ratio
            train_loader.dataset.dataset1.augmentor.split_sampler.init_class_ratio(split_sampler_cfg)
            train_loader.dataset.dataset1.augmentor.split_sampler.update_cfg(split_sampler_cfg)
        train_epoch(source_reader, train_loader, model, sam, model_fn, optimizer_pc, optimizer_img, epoch, rank, dist_train)
        epoch_log = epoch + 1

        # save ckpt
        if rank == 0 and epoch_log % args.ckpt_save_freq == 0:
            filename_img = ckpt_dir / ('img_train_epoch_' + str(epoch_log) + '.pth')
            filename_pc = ckpt_dir / ('pc_train_epoch_' + str(epoch_log) + '.pth')
            logger.info('Saving checkpoint to: ' + str(filename_img))
            save_params(filename_pc, model, optimizer_pc, None, epoch_log)
            save_params(filename_img, sam, optimizer_img, scheduler, epoch_log)
            if not args.reserve_old_ckpt:
                try:
                    os.remove(str(ckpt_dir / ('img_train_epoch_' + str(epoch_log - args.ckpt_save_freq * 2) + '.pth')))
                    os.remove(str(ckpt_dir / ('pc_train_epoch_' + str(epoch_log - args.ckpt_save_freq * 2) + '.pth')))
                except Exception:
                    pass

        if cfg.EVALUATION.evaluate and epoch_log % cfg.EVALUATION.eval_freq == 0:
            logger.info('Model Evaluation ..... ')
            time.sleep(3)
            mIoU_val = validate_epoch(val_loader, model, sam, test_model_fn, epoch, rank, dist_train)
            time.sleep(3)
            if rank == 0 and mIoU_val > best_mIoU:
                best_mIoU = mIoU_val
                best_epoch = epoch_log
                filename_img = ckpt_dir / 'img_best_train.pth'
                filename_pc = ckpt_dir / 'pc_best_train.pth'
                logger.info('Best Model Saving checkpoint to: ' + str(filename_img))
                save_params(filename_pc, model, optimizer_pc, None, epoch_log, metric=best_mIoU)
                save_params(filename_img, sam, optimizer_img, scheduler, epoch_log, metric=best_mIoU)

        # save queue
        if rank == 0 and cfg.DATA_CONFIG_TAR.DATA_AUG.tacm.cuboid_queue.enabled:
            train_loader.dataset.dataset1.augmentor.split_sampler.save_sampler(ckpt_dir / 'split_sampler.pth')

        scheduler.step()
        logger.info('Best epoch: {}, best mIoU: {}'.format(best_epoch, best_mIoU))

    # delete pseudo labels
    if not args.preserve_pseudo_labels:
        os.system('rm -rf {}'.format(pseudo_labels_dir))


def main():
    # ==================================== init ==============================================
    global args, cfg
    args, cfg = parse_config()
    if args.launcher == 'none':
        dist_train = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_train = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION_PC.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    args.epochs = cfg.OPTIMIZATION_PC.NUM_EPOCHS if args.epochs is None else args.epochs

    if args.manual_seed is not None:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)

    pretrain_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    output_dir = pretrain_dir / args.st_extra_tag
    ckpt_dir = output_dir / 'ckpt'
    pseudo_labels_dir = output_dir / 'pseudo_labels'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    pseudo_labels_dir.mkdir(parents=True, exist_ok=True)
    

    # log to file
    global logger
    log_file = output_dir / ('log_{}.txt'.format(datetime.datetime.now().strftime('%Y%m%d-%H%M%S')))
    logger = get_logger(log_file=log_file, rank=cfg.LOCAL_RANK)
    logger.info(
        '*********************************** Start Logging*********************************'
    )
    gpu_list = os.environ[
        "CUDA_VISIBLE_DEVICES"] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys(
        ) else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_train:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)
    if cfg.LOCAL_RANK == 0:
        os.system('cp %s %s' % (args.cfg_file, output_dir))
    global writer
    writer = SummaryWriter(str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None

    # ======================================= create network and dataset ==============================
    # network
    model, model_fn_decorator = model_utils_st.build_model(cfg)
    sam = sam_model_registry["vit_b"](checkpoint=args.sam_path, num_classes=cfg.COMMON_CLASSES.n_classes).cuda()
    model_fn = model_fn_decorator(cfg, args.batch_size)
    model_fn_test = model_fn_decorator(cfg, args.batch_size, test=True)
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    elif cfg.MODEL.get('dsnorm', False):
        model = DSNorm.convert_dsnorm(model)
    model.cuda()
    logger.info('#classifier parameters: {}'.format(sum([x.nelement() for x in model.parameters()])))
    logger.info('#seg classifier parameters: {}'.format(sum([x.nelement() for x in filter(lambda p: p.requires_grad, sam.parameters())])))

    optimizer_pc = common_utils.build_optimizer(cfg.OPTIMIZATION_PC, model)
    optimizer_img = common_utils.build_optimizer(cfg.OPTIMIZATION_IMG, sam)
    scheduler = CosineAnnealingLR(optimizer_img, args.epochs)

    best_mIoU = None
    best_epoch = 0
    if args.weight_pc and args.weight_img:
        model = load_params_from_pretrain(
            args.weight_pc, dist_train, model, logger=logger
        )
        sam = load_params_from_pretrain(
            args.weight_img, dist_train, sam, logger=logger
        )
    else:
        ckpt_list_img = glob.glob(str(ckpt_dir / 'img_train_epoch_*.pth'))
        ckpt_list_pc = glob.glob(str(ckpt_dir / 'pc_train_epoch_*.pth'))
        if len(ckpt_list_img) > 0:
            ckpt_list_img.sort(key=os.path.getmtime)
            model, optimizer_pc, _, args.start_epoch = load_params_from_ckpt(
                ckpt_list_img[-1], dist_train, model, optimizer=optimizer_pc, scheduler=None, logger=logger)
            sam, optimizer_img, scheduler, args.start_epoch = load_params_from_ckpt(
                ckpt_list_pc[-1], dist_train, sam, optimizer=optimizer_img, scheduler=scheduler, logger=logger)
    logger.info('optimizer LR: {}'.format(optimizer_pc.param_groups[0]['lr']))

    if dist_train:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()])
        sam = torch.nn.parallel.DistributedDataParallel(sam, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()], find_unused_parameters=True)

    # dataset
    src_train_loader, src_train_sampler, train_loader, train_sampler, val_loader, val_sampler = get_dataset(
        cfg, args, dist_train, logger
    )
    source_reader = DataReader(src_train_loader, src_train_sampler)
    source_reader.construct_iter()

    split_sampler_cfg = cfg.DATA_CONFIG_TAR.DATA_AUG.tacm.cuboid_queue
    if split_sampler_cfg.enabled and os.path.exists(str(ckpt_dir / 'split_sampler.pth')):
        train_loader.dataset.dataset1.augmentor.split_sampler.load_sampler(str(ckpt_dir / 'split_sampler.pth'))
        train_loader.dataset.dataset1.augmentor.split_sampler.update_cfg(split_sampler_cfg)
        logger.info('=> loaded split sampler \'{}\''.format(str(ckpt_dir / 'split_sampler.pth')))

    logger.info(
        '**********************Start training %s/%s(%s)**********************'
        % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag + '/' + args.st_extra_tag))
    train(model, sam, model_fn, model_fn_test, source_reader, train_loader, val_loader,
          optimizer_pc, optimizer_img, scheduler, ckpt_dir, pseudo_labels_dir, cfg.LOCAL_RANK, dist_train=dist_train,
          train_sampler=train_sampler, best_mIoU=best_mIoU, best_epoch=best_epoch)


if __name__ == '__main__':
    import gc
    
    gc.collect()
    main()
