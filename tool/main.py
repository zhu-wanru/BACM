


import _init_path
import os
import time
import random
import datetime
import numpy as np
import argparse
import glob
from pathlib import Path
from tensorboardX import SummaryWriter

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.distributed as dist
import torch.optim
import torch.utils.data
from torch.optim.lr_scheduler import CosineAnnealingLR
from itertools import islice

import sys
sys.path.append('..')
from dataset import get_src_train_dataset, get_val_dataset, get_tar_train_dataset
from util import common_utils
from util import model_utils
from util.common_utils import AverageMeter, update_meter, calc_metrics, get_logger
from util.model_utils import load_params_from_ckpt, load_params_from_pretrain, load_metric_from_ckpt, save_params
from util.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from model.dsnorm import DSNorm, set_ds_source, set_ds_target
from segment_anything import sam_model_registry


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--exp_name', type=str, default='da', help='experiment name')
    parser.add_argument('--src_dataset', type=str, default='front3d', help='source dataset')
    parser.add_argument('--trgt_dataset', type=str, default='scannet', help='target dataset')
    parser.add_argument('--sam_path', type=str, default='../pretrained_model/sam_vit_b_01ec64.pth',help='pretrain sam path')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--resume', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--weight', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='pytorch')
    parser.add_argument('--tcp_port', type=int, default=18869, help='tcp port for distrbuted training')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--reserve_old_ckpt', action='store_true', default=False, help='whether to remove previously saved ckpt')
    parser.add_argument('--manual_seed', type=int, default=1028, help='')
    parser.add_argument('--ckpt_save_freq', type=int, default=1, help='number of training epochs')
    parser.add_argument('--print_freq', type=int, default=20, help='printing log frequency')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--max_ckpt_save_num', type=int, default=30, help='max number of saved checkpoint')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')
    parser.add_argument('--pin_memory', action='store_true', default=False, help='')
    parser.add_argument('--alpha', type=float, default=0.25, help='')
    parser.add_argument('--beta', type=float, default=0.25, help='')
    parser.add_argument('--step', type=int, default=16, help='img grid step')
    args = parser.parse_args()

    args.cfg_file = '../cfgs/da_'+args.src_dataset+'_'+args.trgt_dataset+'/spconv.yaml'
    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = args.exp_name+'_'+args.src_dataset+'_'+args.trgt_dataset

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)
    return args, cfg


def worker_init_fn(worker_id):
    random.seed(args.manual_seed + worker_id)


def train_epoch(train_loader, trgt_train_loader, model, sam, model_fn, optimizer_img, optimizer_pc, scheduler_img, scheduler_pc, epoch, rank, dist_train):
    img2pc_intersection_meter = AverageMeter()
    img2pc_union_meter = AverageMeter()
    img2pc_target_meter = AverageMeter()

    pc_intersection_meter = AverageMeter()
    pc_union_meter = AverageMeter()
    pc_target_meter = AverageMeter()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    train_loader = iter(train_loader)
    trgt_train_loader = iter(trgt_train_loader)
    
    end = time.time()
    iter_num = max(len(trgt_train_loader), len(train_loader))
    min_iter = min(len(trgt_train_loader), len(train_loader))
    max_iter = args.epochs * iter_num
    logger.info('lr: {:.7f} '.format(optimizer_img.param_groups[0]['lr']))
    model.train()
    sam.train()
    accs = 0.0

    for i in range(iter_num-min_iter):

        data_time.update(time.time() - end)
        torch.cuda.empty_cache()

        # forward
        if cfg.MODEL.get('dsnorm', False):
            model.apply(set_ds_source)
        batch = next(train_loader)
        optimizer_img.zero_grad()
        optimizer_pc.zero_grad()
        ret = model_fn(batch, model, sam, epoch, args.step, args.alpha, args.beta, mode = 'src')
        loss = ret.get('loss', torch.tensor(0).float().cuda())
        loss.backward()
        optimizer_img.step()
        optimizer_pc.step()


        pc_labels = ret['pc_labels']
        img2pc_preds = ret['img2pc_preds']
        pc_preds = ret['pc_preds']
        preds = ret['all_preds']
        offsets = batch['offsets']
        acc = ret['acc']
        accs += acc

        # update loss
        if dist_train:
            n = img2pc_preds.size(0)
            loss *= n
            count = pc_labels.new_tensor([n], dtype=torch.long).cuda()
            dist.all_reduce(loss); dist.all_reduce(count)
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
        intersection_meter, union_meter, target_meter, accuracy, intersection, union, target = \
            update_meter(intersection_meter, union_meter, target_meter, preds, pc_labels, 
            cfg.COMMON_CLASSES.n_classes, cfg.DATA_CONFIG.DATA_CLASS.ignore_label, dist_train)

        # update time and print log
        batch_time.update(time.time() - end)
        end = time.time()
        current_iter = epoch * iter_num + i + 1
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))
        if (i + 1) % args.print_freq == 0 or i == iter_num - 1:
            logger.info('Epoch Src: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'Loss {loss_meter.val:.4f} '
                        'Accuracy img2pc {img2pc:.4f} '
                        'pc {pc:.4f} '
                        'prompt {prompt:.4f} '
                        'all {all:.4f}.'.format(epoch+1, args.epochs, i + 1, iter_num,
                                                          batch_time=batch_time, data_time=data_time,
                                                          remain_time=remain_time,
                                                          loss_meter=loss_meter,
                                                          img2pc=img2pc_accuracy,
                                                          pc=pc_accuracy,
                                                          prompt=accs/(i+1),
                                                          all=accuracy
                                                          ))

        # record to writer
        if rank == 0:
            writer.add_scalar('loss_train_batch', loss_meter.val, current_iter)
            writer.add_scalar('img2pc_mIoU_train_batch', np.mean(img2pc_intersection / (img2pc_union + 1e-10)), current_iter)
            writer.add_scalar('img2pc_mAcc_train_batch', np.mean(img2pc_intersection / (img2pc_target + 1e-10)), current_iter)
            writer.add_scalar('img2pc_allAcc_train_batch', img2pc_accuracy, current_iter)
            writer.add_scalar('pc_mIoU_train_batch', np.mean(pc_intersection / (pc_union + 1e-10)), current_iter)
            writer.add_scalar('pc_mAcc_train_batch', np.mean(pc_intersection / (pc_target + 1e-10)), current_iter)
            writer.add_scalar('pc_allAcc_train_batch', pc_accuracy, current_iter)
            writer.add_scalar('mIoU_train_batch', np.mean(intersection / (union + 1e-10)), current_iter)
            writer.add_scalar('mAcc_train_batch', np.mean(intersection / (target + 1e-10)), current_iter)
            writer.add_scalar('allAcc_train_batch', accuracy, current_iter)

    for i in range(iter_num-min_iter, iter_num):    
        data_time.update(time.time() - end)
        torch.cuda.empty_cache()

        # forward
        if cfg.MODEL.get('dsnorm', False):
            model.apply(set_ds_source)

        batch = next(train_loader)
        trgt_batch = next(trgt_train_loader)

        optimizer_img.zero_grad()
        optimizer_pc.zero_grad()
        ret = model_fn(batch, model, sam, epoch, args.step, args.alpha, args.beta, mode = 'src')
        loss = ret.get('loss', torch.tensor(0).float().cuda())
        loss.backward()
        optimizer_img.step()
        optimizer_pc.step()


        optimizer_img.zero_grad()
        optimizer_pc.zero_grad()
        ret_1 = model_fn(trgt_batch, model, sam, epoch, args.step, args.alpha, args.beta, mode = 'trgt')
        loss_1 = ret_1.get('loss', torch.tensor(0).float().cuda())
        loss_1.backward()
        optimizer_img.step()
        optimizer_pc.step()
        loss = loss + loss_1

        pc_labels = ret['pc_labels']
        img2pc_preds = ret['img2pc_preds']
        pc_preds = ret['pc_preds']
        preds = ret['all_preds']
        offsets = batch['offsets']
        acc = ret['acc']
        accs += acc

        # update loss
        if dist_train:
            n = img2pc_preds.size(0)
            loss *= n
            count = pc_labels.new_tensor([n], dtype=torch.long).cuda()
            dist.all_reduce(loss); dist.all_reduce(count)
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
        intersection_meter, union_meter, target_meter, accuracy, intersection, union, target = \
            update_meter(intersection_meter, union_meter, target_meter, preds, pc_labels, 
            cfg.COMMON_CLASSES.n_classes, cfg.DATA_CONFIG.DATA_CLASS.ignore_label, dist_train)

        # update time and print log
        batch_time.update(time.time() - end)
        end = time.time()
        current_iter = epoch * iter_num + i + 1
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))
        if (i + 1) % args.print_freq == 0 or i == iter_num - 1:
            logger.info('Epoch Trgt: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'Loss {loss_meter.val:.4f} '
                        'Accuracy img2pc {img2pc:.4f} '
                        'pc {pc:.4f} '
                        'prompt {prompt:.4f} '
                        'all {all:.4f}.'.format(epoch+1, args.epochs, i + 1, iter_num,
                                                          batch_time=batch_time, data_time=data_time,
                                                          remain_time=remain_time,
                                                          loss_meter=loss_meter,
                                                          img2pc=img2pc_accuracy,
                                                          pc=pc_accuracy,
                                                          prompt=accs/(i+1),
                                                          all=accuracy
                                                          ))


    img2pc_mIoU, img2pc_mAcc, img2pc_allAcc, img2pc_iou_class, img2pc_accuracy_class = \
        calc_metrics(img2pc_intersection_meter, img2pc_union_meter, img2pc_target_meter)
    pc_mIoU, pc_mAcc, pc_allAcc, pc_iou_class, pc_accuracy_class = \
        calc_metrics(pc_intersection_meter, pc_union_meter, pc_target_meter)
    mIoU, mAcc, allAcc, iou_class, accuracy_class = \
        calc_metrics(intersection_meter, union_meter, target_meter)
    logger.info('Img2pc train result at epoch [{}/{}]: mIoU/mPre/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format
        (epoch+1, args.epochs, img2pc_mIoU,  img2pc_mAcc, img2pc_allAcc))
    logger.info('Pc train result at epoch [{}/{}]: mIoU/mPre/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format
        (epoch+1, args.epochs, pc_mIoU,  pc_mAcc, pc_allAcc))
    logger.info('All train result at epoch [{}/{}]: mIoU/mPre/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format
        (epoch+1, args.epochs, mIoU,  mAcc, allAcc))
    
    if rank == 0:
        writer.add_scalar('loss_train', loss_meter.avg, epoch + 1)
        writer.add_scalar('img2pc_mIoU_train', img2pc_mIoU, epoch + 1)
        writer.add_scalar('img2pc_mAcc_train', img2pc_mAcc, epoch + 1)
        writer.add_scalar('img2pc_allAcc_train', img2pc_allAcc, epoch + 1)
        writer.add_scalar('pc_mIoU_train', pc_mIoU, epoch + 1)
        writer.add_scalar('pc_mAcc_train', pc_mAcc, epoch + 1)
        writer.add_scalar('pc_allAcc_train', pc_allAcc, epoch + 1)
        writer.add_scalar('mIoU_train', mIoU, epoch + 1)
        writer.add_scalar('mAcc_train', mAcc, epoch + 1)
        writer.add_scalar('allAcc_train', allAcc, epoch + 1)
    return


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
    num_gpus = common_utils.get_world_size()
    accs = 0.0
    if cfg.MODEL.get('dsnorm', False):
        model.apply(set_ds_target)
    for i, batch in enumerate(val_loader):
        data_time.update(time.time() - end)

        # forward
        with torch.no_grad():
            ret = model_fn(batch, model, sam, epoch, args.step, args.alpha, args.beta, mode = 'trgt')
            loss = ret['loss']
            pc_labels = ret['pc_labels']
            img2pc_preds = ret['img2pc_preds']
            pc_preds = ret['pc_preds']
            preds = ret['all_preds']
            acc = ret['acc']
            accs += acc
        
        if dist_train and (i * args.batch_size + (batch['id'].size(0) - 1)) * num_gpus + rank + 1> val_loader.dataset.__len__():
            img2pc_preds, pc_preds, preds, pc_labels = img2pc_preds[:batch['offsets'][-2]], pc_preds[:batch['offsets'][-2]], preds[:batch['offsets'][-2]], pc_labels[:batch['offsets'][-2]]
            batch['offsets'] = batch['offsets'][:-1]
            batch['id'] = batch['id'][:-1]

        # update loss
        if dist_train:
            n = pc_preds.size(0)
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
        intersection_meter, union_meter, target_meter, accuracy, intersection, union, target = \
            update_meter(intersection_meter, union_meter, target_meter, preds, pc_labels, 
            cfg.COMMON_CLASSES.n_classes, cfg.DATA_CONFIG.DATA_CLASS.ignore_label, dist_train)

        # update time and print log
        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) % args.print_freq == 0:
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                        'Accuracy img2pc {img2pc:.4f} pc {pc:.4f} prompt {prompt:.4f} all {all:.4f}.'.format(i + 1, len(val_loader),
                                                                                        data_time=data_time,
                                                                                        batch_time=batch_time,
                                                                                        loss_meter=loss_meter,
                                                                                        # img=img_accuracy,
                                                                                        img2pc=img2pc_accuracy,
                                                                                        pc=pc_accuracy,
                                                                                        prompt=accs/(i+1),
                                                                                        all=accuracy))
    img2pc_mIoU, img2pc_mAcc, img2pc_allAcc, img2pc_iou_class, img2pc_accuracy_class = \
        calc_metrics(img2pc_intersection_meter, img2pc_union_meter, img2pc_target_meter)
    pc_mIoU, pc_mAcc, pc_allAcc, pc_iou_class, pc_accuracy_class = \
        calc_metrics(pc_intersection_meter, pc_union_meter, pc_target_meter)
    mIoU, mAcc, allAcc, iou_class, accuracy_class = \
        calc_metrics(intersection_meter, union_meter, target_meter)

    logger.info('Img2pc val result: mIoU/mPre/mAcc/allPre/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(
        img2pc_mIoU, img2pc_mAcc, img2pc_allAcc))
    logger.info('Pc val result: mIoU/mPre/mAcc/allPre/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(
        pc_mIoU, pc_mAcc, pc_allAcc))
    logger.info('All val result: mIoU/mPre/mAcc/allPre/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(
        mIoU, mAcc, allAcc))
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
        writer.add_scalar('loss_train', loss_meter.avg, epoch + 1)
        writer.add_scalar('img2pc_mIoU_train', img2pc_mIoU, epoch + 1)
        writer.add_scalar('img2pc_mAcc_train', img2pc_mAcc, epoch + 1)
        writer.add_scalar('img2pc_allAcc_train', img2pc_allAcc, epoch + 1)
        writer.add_scalar('pc_mIoU_train', pc_mIoU, epoch + 1)
        writer.add_scalar('pc_mAcc_train', pc_mAcc, epoch + 1)
        writer.add_scalar('pc_allAcc_train', pc_allAcc, epoch + 1)
    return pc_mIoU


def train(
    model, sam, model_fn, model_fn_test, train_loader, trgt_train_loader, val_loader, optimizer_img, optimizer_pc, scheduler_img, scheduler_pc, ckpt_dir, rank,
    dist_train=False, train_sampler=None, best_mIoU=None, best_epoch=0
):

    best_mIoU = best_mIoU if best_mIoU is not None else 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if train_sampler is not None:  # compatible for pytorch1.1
            train_sampler.set_epoch(epoch)

        train_epoch(train_loader, trgt_train_loader, model, sam, model_fn, optimizer_img, optimizer_pc, scheduler_img, scheduler_pc, epoch, rank, dist_train)
        epoch_log = epoch + 1

        if rank == 0 and epoch_log % args.ckpt_save_freq == 0:
            img_filename = ckpt_dir / ('img_train_epoch_' + str(epoch_log) + '.pth')
            pc_filename = ckpt_dir / ('pc_train_epoch_' + str(epoch_log) + '.pth')
            logger.info('Saving checkpoint to: ' + str(img_filename))
            save_params(pc_filename, model, optimizer_pc, scheduler_pc, epoch_log)
            save_params(img_filename, sam, optimizer_img, scheduler_img, epoch_log)
            if not args.reserve_old_ckpt:
                try:
                    os.remove(str(ckpt_dir / ('img_train_epoch_' + str(epoch_log - args.ckpt_save_freq * 2) + '.pth')))
                    os.remove(str(ckpt_dir / ('pc_train_epoch_' + str(epoch_log - args.ckpt_save_freq * 2) + '.pth')))
                except Exception:
                    pass
        
        if cfg.EVALUATION.evaluate and epoch_log % cfg.EVALUATION.eval_freq == 0:
            mIoU_val = validate_epoch(val_loader, model, sam, model_fn_test, epoch, rank, dist_train)
            if rank == 0 and mIoU_val > best_mIoU:
                best_mIoU = mIoU_val
                best_epoch = epoch_log
                img_filename = ckpt_dir / 'img_best_train.pth'
                pc_filename = ckpt_dir / 'pc_best_train.pth'
                logger.info('Best Model Saving checkpoint to: ' + str(img_filename))
                save_params(img_filename, sam, optimizer_img, scheduler_img, epoch_log, metric=best_mIoU)
                save_params(pc_filename, model, optimizer_pc, scheduler_pc, epoch_log, metric=best_mIoU)

        scheduler_img.step()
        scheduler_pc.step()
        logger.info('Best epoch: {}, best mIoU: {}'.format(best_epoch, best_mIoU))


def main():
    # ==================================== init ==============================================
    global args
    args, cfg = parse_config()
    if args.launcher == 'none':
        dist_train = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl')
        dist_train = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs

    if args.manual_seed is not None:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    ckpt_dir = output_dir / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # log to file
    global logger
    log_file = output_dir / ('log_{}.txt'.format(datetime.datetime.now().strftime('%Y%m%d-%H%M%S')))
    logger = get_logger(log_file=log_file, rank=cfg.LOCAL_RANK)
    logger.info('*********************************** Start Logging*********************************')
    gpu_list = os.environ["CUDA_VISIBLE_DEVICES"] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
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

    # ================================= create network and dataset ==============================
    # network
    model, model_fn_decorator = model_utils.build_model(cfg)
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
    optimizer_img = common_utils.build_optimizer(cfg.OPTIMIZATION, sam)
    
    if args.trgt_dataset == 's3dis':
        scheduler_pc = CosineAnnealingLR(optimizer_pc, args.epochs, eta_min=1e-5)
    else:
        scheduler_pc = CosineAnnealingLR(optimizer_pc, args.epochs)
    scheduler_img = CosineAnnealingLR(optimizer_img, args.epochs)


    best_mIoU = None
    best_epoch = 0
    if args.weight:
        model = load_params_from_pretrain(
            args.weight, dist_train, model, logger=logger, strict=not args.pretrain_not_strict
        )
        sam = load_params_from_pretrain(
            args.weight, dist_train, sam, logger=logger, strict=not args.pretrain_not_strict
        )
    else:
        ckpt_list_img = glob.glob(str(ckpt_dir / 'img_train_epoch_*.pth'))
        ckpt_list_pc = glob.glob(str(ckpt_dir / 'pc_train_epoch_*.pth'))
        if len(ckpt_list_pc) > 0:
            ckpt_list_img.sort(key=os.path.getmtime)
            model, optimizer_pc, scheduler_pc, _ = load_params_from_ckpt(ckpt_list_pc[0], dist_train, model, optimizer=optimizer_pc, scheduler=scheduler_pc, logger=logger)
            sam, optimizer_img, scheduler_img, args.start_epoch = load_params_from_ckpt(ckpt_list_img[0], dist_train, sam, optimizer=optimizer_img, scheduler=scheduler_img, logger=logger)
    logger.info('optimizer LR: {}'.format(optimizer_img.param_groups[0]['lr']))


    

    if dist_train:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()], find_unused_parameters=True)
        sam = torch.nn.parallel.DistributedDataParallel(sam, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()], find_unused_parameters=True)

    # dataset
    # source train data
    train_data, train_loader, train_sampler = get_src_train_dataset(
        cfg, args, dist_train, logger, pin_memory=args.pin_memory
    )
    trgt_train_loader, trgt_train_sampler  = get_tar_train_dataset(
        cfg, args, dist_train, logger, pin_memory=args.pin_memory)

    # target val data
    val_loader, val_sampler = get_val_dataset(args, cfg.DATA_CONFIG_TAR, dist_train, logger)

    logger.info('**********************Start training %s/%s(%s)**********************'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
    train(
        model, sam, model_fn, model_fn_test, train_loader, trgt_train_loader, val_loader, optimizer_img, optimizer_pc, scheduler_img, scheduler_pc, ckpt_dir,
        cfg.LOCAL_RANK, dist_train=dist_train, train_sampler=train_sampler, best_mIoU=best_mIoU, best_epoch=best_epoch
    )

    logger.info(' ************************** Clean Shared Memory ***************************')
    if cfg.LOCAL_RANK == 0:
        train_data.destroy_shm()
        val_loader.dataset.destroy_shm()


if __name__ == '__main__':
    import gc
    gc.collect()
    main()

 

