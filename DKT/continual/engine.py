# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import json
import os
import math
from typing import Iterable, Optional

import torch
from timm.utils import accuracy
from timm.loss import SoftTargetCrossEntropy
from torch.nn import functional as F

import continual.utils as utils
from continual.losses import DistillationLoss

CE = SoftTargetCrossEntropy()


def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, task_id: int, loss_scaler, max_norm: float = 0,
                    set_training_mode=True, debug=False, args=None,
                    teacher_model: torch.nn.Module = None,
                    model_without_ddp: torch.nn.Module = None,
                    loader_memory=None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Task: [{}] Epoch: [{}]'.format(task_id, epoch)
    print_freq = 10

    for batch_index, (samples, targets, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if batch_index == 0:
            print(f'Image size is {samples.shape}.')

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=not args.no_amp):
            loss_tuple = forward(samples, targets, model, teacher_model, criterion, args)

        loss = sum(filter(lambda x: x is not None, loss_tuple))
        internal_losses = model_without_ddp.get_internal_losses(loss)
        for internal_loss_value in internal_losses.values():
            loss += internal_loss_value

        check_loss(loss)

        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order

        loss_scaler(loss, optimizer, model_without_ddp, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()

        metric_logger.update_dict(internal_losses)
        metric_logger.update(loss=loss_tuple[0])
        metric_logger.update(kd=loss_tuple[1])
        metric_logger.update(feature=loss_tuple[2])
        metric_logger.update(cs=loss_tuple[3])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if debug:
            print('Debug, only doing one epoch!')
            break

    if hasattr(model_without_ddp, 'hook_after_epoch'):
        model_without_ddp.hook_after_epoch()

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def check_loss(loss):
    if not math.isfinite(loss.item()):
        raise Exception('Loss is {}, stopping training'.format(loss.item()))


def forward(samples, targets, model, teacher_model, criterion, args):

    outputs = model(samples)
    if isinstance(outputs, dict):
        main_output = outputs['logits']
        feature_output = outputs['tokens']
    else:
        main_output = outputs

    loss = criterion(main_output, targets)

    if teacher_model is not None:
        with torch.no_grad():
            main_output_old = None
            teacher_outputs = teacher_model(samples)

        if isinstance(outputs, dict):
            main_output_old = teacher_outputs['logits']
            feature_output_old = teacher_outputs['tokens']
        else:
            main_output_old = teacher_outputs

    kd_loss = None
    fe_loss = None
    cs_loss = None
    if teacher_model is not None:
        logits_for_distil = main_output[:, :main_output_old.shape[1]]

        kd_loss = 0.
        fe_loss = 0.
        if args.auto_kd:
            lbd = main_output_old.shape[1] / main_output.shape[1]
            loss = (1 - lbd) * loss
            kd_factor = lbd

            tau = args.distillation_tau

            _kd_loss = F.kl_div(
                F.log_softmax(logits_for_distil / tau, dim=1),
                F.log_softmax(main_output_old / tau, dim=1),
                reduction='mean',
                log_target=True
            ) * (tau ** 2)
            kd_loss += loss + kd_factor * _kd_loss
            targets_unsqueezed = targets.unsqueeze(1)
            aa = targets_unsqueezed
            ba = targets_unsqueezed.T
            indexes = (aa == ba).to(torch.int)
            div_targets = torch.clone(targets)
            nb_classes = main_output.shape[1]
            nb_new_classes = args.increment
            nb_old_classes = nb_classes - nb_new_classes
            mask_old_cls = div_targets < nb_old_classes
            mask_old_cls_un = mask_old_cls.unsqueeze(1)
            mask_old_cls_vn = mask_old_cls.unsqueeze(1).T
            cc = (mask_old_cls_un ^ mask_old_cls_vn)
            indexes[indexes == 1] = 4
            indexes[cc] = -4
            indexes[indexes == 0] = -1
            computed_similarity = sim_matrix(feature_output, feature_output_old).flatten()
            _cs_loss = 1 - computed_similarity
            _cs_loss *= indexes.flatten()
            cs_loss = 0.003 * _cs_loss.mean()
            fe_loss = 0  # the feature distillation, which is abandoned

    return loss, kd_loss, fe_loss, cs_loss


@torch.no_grad()
def evaluate(data_loader, model, device, logger):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    model.eval()

    for images, target, task_ids in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            output = model(images)
            if isinstance(output, dict):
                output = output['logits']
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, min(5, output.shape[1])))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

        logger.add([output.cpu().argmax(dim=1), target.cpu(), task_ids], subset='test')

    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f}  loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def eval_and_log(args, output_dir, model, model_without_ddp, optimizer, lr_scheduler,
                 epoch, task_id, loss_scaler, max_accuracy, accuracy_list,
                 n_parameters, device, data_loader_val, train_stats, log_store, log_path, logger,
                 model_log, skipped_task=False):
    if args.output_dir:
        if os.path.isdir(args.resume):
            checkpoint_paths = [os.path.join(args.resume, f'checkpoint_{task_id}.pth')]
        else:
            checkpoint_paths = [output_dir / f'checkpoint_{task_id}.pth']
        for checkpoint_path in checkpoint_paths:
            if skipped_task:
                continue

            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'task_id': task_id,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }, checkpoint_path)

    test_stats = evaluate(data_loader_val, model, device, logger)
    print(f"Accuracy of the network on the {len(data_loader_val.dataset)} test images: {test_stats['acc1']:.1f}%")
    max_accuracy = max(max_accuracy, test_stats["acc1"])
    print(f'Max accuracy: {max_accuracy:.2f}%')
    accuracy_list.append(test_stats['acc1'])

    log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                 **{f'test_{k}': v for k, v in test_stats.items()},
                 'epoch': epoch,
                 'n_parameters': n_parameters}

    mean_acc5 = -1.0
    if log_store is not None:
        log_store['results'][task_id] = log_stats
        all_acc5 = [task_log['test_acc5'] for task_log in log_store['results'].values()]
        mean_acc5 = sum(all_acc5) / len(all_acc5)

    if log_path is not None and utils.is_main_process():
        with open(log_path, 'a+') as f:
            f.write(json.dumps({
                'task': task_id,
                'epoch': epoch,
                'acc': round(100 * logger.accuracy, 2),
                'avg_acc': round(100 * logger.average_incremental_accuracy, 2),
                'forgetting': round(100 * logger.forgetting, 6),
                'acc_per_task': [round(100 * acc_t, 2) for acc_t in logger.accuracy_per_task],
                'train_lr': log_stats.get('train_lr', 0.),
                'bwt': round(100 * logger.backward_transfer, 2),
                'fwt': round(100 * logger.forward_transfer, 2),
                'test_acc1': round(log_stats['test_acc1'], 2),
                'test_acc5': round(log_stats['test_acc5'], 2),
                'mean_acc5': round(mean_acc5, 2),
                'train_loss': round(log_stats.get('train_loss', 0.), 5),
                'test_loss': round(log_stats['test_loss'], 5),
                **model_log
            }) + '\n')
    if args.output_dir and utils.is_main_process():
        with (output_dir / "log.txt").open("a") as f:
            f.write(json.dumps(log_stats) + "\n")

    return max_accuracy


def indexes_task_outputs(logits, targets, increment_per_task):
    if increment_per_task[0] != increment_per_task[1]:
        raise NotImplementedError(f'Not supported yet for non equal task size')

    inc = increment_per_task[0]
    indexes = torch.zeros(len(logits), inc).long()

    for r in range(indexes.shape[0]):
        for c in range(indexes.shape[1]):
            indexes[r, c] = (targets[r] // inc) * inc + r * logits.shape[1] + c

    indexed_logits = logits.view(-1)[indexes.view(-1)].view(len(logits), inc)
    indexed_targets = targets % inc

    return indexed_logits, indexed_targets


def sim_matrix(a, b, eps=1e-8):
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt
