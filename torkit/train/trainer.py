import time
import warnings
from typing import Dict, Optional

import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data.dataloader import DataLoader

from yacs.config import CfgNode
from .model import GenericModel
from torkit.utils.checkpoint import Checkpointer
from torkit.utils.metric_logger import MetricLogger
from torkit.utils import comm


class IterationBasedTrainer(object):
    """Iteration-based training engine.

    The user can inherit this class in the training script to support more features.
    More complicated training pipeline, e.g., detection, can be modified based on this.
    """

    def __init__(self,
                 cfg: CfgNode,
                 model: GenericModel,
                 optimizer: Optimizer,
                 train_dataloader: DataLoader,
                 lr_scheduler: Optional[_LRScheduler] = None,
                 val_dataloader: Optional[DataLoader] = None,
                 data_parallel=False,
                 distributed=False,
                 gpu=None,
                 ):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # ---------------------------------------------------------------------------- #
        # It is normal that you might modify the following section,
        # as it is usually task-specific.
        # ---------------------------------------------------------------------------- #
        # sanity check
        assert hasattr(model, 'compute_losses')
        assert hasattr(model, 'get_metrics')

        if data_parallel:
            self.model_parallel = nn.DataParallel(self.model)
        else:
            self.model_parallel = self.model

        self.distributed = distributed
        if distributed:
            self.model_parallel = nn.parallel.DistributedDataParallel(
                model, device_ids=[gpu], find_unused_parameters=True)
        else:
            self.model_parallel = self.model

    def train(self,
              start_iter=0,
              logger=None,
              checkpointer: Optional[Checkpointer] = None,
              checkpoint_data: Optional[Dict] = None,
              summary_writer=None,
              validate=False):
        # aliases
        cfg = self.cfg
        model_parallel = self.model_parallel
        model = self.model
        optimizer = self.optimizer
        lr_scheduler = self.lr_scheduler
        train_dataloader = self.train_dataloader
        max_iter = cfg.TRAIN.MAX_ITER
        is_main_process = comm.is_main_process()

        # logging
        if logger is None:
            warnings.warn('Logger is not provided. The default loguru logger is used.')
            from loguru import logger

        # metrics
        train_meters = MetricLogger(delimiter='  ')
        train_meters.metrics.update(model.get_metrics(training=True))

        # checkpoint
        if checkpoint_data is None:
            checkpoint_data = dict()

        # best validation
        best_metric_name = 'best_{}'.format(cfg.VAL.METRIC)
        best_metric = checkpoint_data.get(best_metric_name, None)

        # ---------------------------------------------------------------------------- #
        # Training begins.
        # ---------------------------------------------------------------------------- #
        logger.info('Start training from iteration {}'.format(start_iter))
        tic = time.time()
        for iteration, data_batch in enumerate(train_dataloader, start_iter):
            cur_iter = iteration + 1  # 1-index
            if cur_iter > max_iter:  # in case len(train_dataloader) >= max_iter
                break
            data_time = time.time() - tic

            # Copy data from cpu to gpu
            data_batch = {k: v.cuda(non_blocking=True) for k, v in data_batch.items()}

            # Forward
            pred_dict = model_parallel(data_batch)
            # Compute losses
            loss_dict = model.compute_losses(pred_dict, data_batch)
            total_loss = sum(loss_dict.values())

            # It is slightly faster to update metrics between forward and backward.
            optimizer.zero_grad()
            with torch.no_grad():
                train_meters.update(total_loss=total_loss, **loss_dict)
                model.update_metrics(pred_dict, data_batch, train_meters.metrics, training=True)

            # Backward
            total_loss.backward()
            if cfg.OPTIMIZER.MAX_GRAD_NORM > 0:
                # CAUTION: built-in clip_grad_norm_ clips the total norm.
                clip_grad_norm_(model.parameters(), max_norm=cfg.OPTIMIZER.MAX_GRAD_NORM)
            optimizer.step()

            batch_time = time.time() - tic
            train_meters.update(time=batch_time, data=data_time)

            # Logging
            log_period = cfg.TRAIN.LOG_PERIOD
            if log_period > 0 and (cur_iter % log_period == 0 or cur_iter == 1) and is_main_process:
                logger.info(
                    train_meters.delimiter.join(
                        [
                            'iter: {iter:4d}',
                            '{meters}',
                            'lr: {lr:.2e}',
                            'max mem: {memory:.0f}',
                        ]
                    ).format(
                        iter=cur_iter,
                        meters=str(train_meters),
                        lr=optimizer.param_groups[0]['lr'],
                        memory=torch.cuda.max_memory_allocated() / (1024.0 ** 2),
                    )
                )
                torch.cuda.reset_max_memory_allocated()

            # Summary
            summary_period = cfg.TRAIN.SUMMARY_PERIOD
            if summary_writer is not None and summary_period > 0 and cur_iter % summary_period == 0 and is_main_process:
                for name, metric in train_meters.metrics.items():
                    summary_writer.add_scalar('train/' + name, metric.result, global_step=cur_iter)
                    if hasattr(model, 'summarize'):
                        model.summarize(pred_dict, data_batch, train_meters.metrics,
                                        summary_writer, cur_iter, training=True)

            if validate and (cur_iter % cfg.VAL.PERIOD == 0 or cur_iter == max_iter):
                best_metric = self.validate(cur_iter, best_metric=best_metric,
                                            logger=logger, summary_writer=summary_writer,
                                            checkpointer=checkpointer, checkpoint_data=checkpoint_data)
                # after validation
                model_parallel.train()
                train_meters.reset()

            # Checkpoint
            ckpt_period = cfg.TRAIN.CHECKPOINT_PERIOD
            if ((ckpt_period > 0 and cur_iter % ckpt_period == 0) or cur_iter == max_iter) and is_main_process:
                checkpoint_data['iteration'] = cur_iter
                if best_metric is not None:
                    checkpoint_data[best_metric_name] = best_metric
                checkpointer.save('model_{:06d}'.format(cur_iter), **checkpoint_data)

            # Finalize one training step
            # Since pytorch v1.1.0, lr_scheduler is called after optimization.
            if lr_scheduler is not None:
                lr_scheduler.step()
            tic = time.time()

        # END: training loop
        if validate and cfg.VAL.METRIC:
            logger.info('{} = {}'.format(best_metric_name, best_metric))

    def validate(self,
                 global_step,
                 best_metric=None,
                 logger=None,
                 checkpointer: Optional[Checkpointer] = None,
                 checkpoint_data: Optional[Dict] = None,
                 summary_writer=None,
                 ):
        # aliases
        cfg = self.cfg
        model_parallel = self.model_parallel
        model = self.model
        val_dataloader = self.val_dataloader
        is_main_process = comm.is_main_process()

        # Prepare for validation
        model_parallel.eval()

        # logging
        val_meters = MetricLogger(delimiter='  ')
        val_meters.metrics.update(model.get_metrics(training=False))

        if is_main_process:
            logger.info('Validation begins at iteration {}.'.format(global_step))

        start_time = time.time()
        tic = time.time()
        for iteration, data_batch in enumerate(val_dataloader, 1):
            data_time = time.time() - tic

            # copy data from cpu to gpu
            data_batch = {k: v.cuda(non_blocking=True) for k, v in data_batch.items()}

            # Forward
            with torch.no_grad():
                pred_dict = model_parallel(data_batch)
                loss_dict = model.compute_losses(pred_dict, data_batch)
                total_loss = sum(loss_dict.values())

                val_meters.update(total_loss=total_loss.item(), **loss_dict)
                model.update_metrics(pred_dict, data_batch, val_meters.metrics, training=False)

            batch_time = time.time() - tic
            val_meters.update(time=batch_time, data=data_time)

            # Logging
            if cfg.VAL.LOG_PERIOD > 0 and iteration % cfg.VAL.LOG_PERIOD == 0 and is_main_process:
                logger.info(
                    val_meters.delimiter.join(
                        [
                            'iter: {iter:4d}',
                            '{meters}',
                            'max mem: {memory:.0f}',
                        ]
                    ).format(
                        iter=iteration,
                        meters=str(val_meters),
                        memory=torch.cuda.max_memory_allocated() / (1024.0 ** 2),
                    )
                )

            # END: validation step
            tic = time.time()

        # END: validation loop
        elapsed_time = time.time() - start_time
        logger.info('Iteration[{}]-Val {}  total_time: {:.2f}s'.format(
            global_step, val_meters.summary_str, elapsed_time))

        # Summary
        if summary_writer is not None and is_main_process:
            for name, metric in val_meters.metrics.items():
                summary_writer.add_scalar('val/' + name, metric.result, global_step=global_step)
            if hasattr(model, 'summarize'):
                model.summarize(pred_dict, data_batch, val_meters.metrics,
                                summary_writer, global_step, training=False)

        # best validation
        if cfg.VAL.METRIC in val_meters.metrics and is_main_process:
            cur_metric = val_meters.metrics[cfg.VAL.METRIC].result
            best_metric_name = 'best_{}'.format(cfg.VAL.METRIC)
            if best_metric is None \
                    or (cfg.VAL.METRIC_ASCEND and cur_metric > best_metric) \
                    or (not cfg.VAL.METRIC_ASCEND and cur_metric < best_metric):
                best_metric = cur_metric

                if checkpointer is not None:
                    checkpoint_data['iteration'] = global_step
                    checkpoint_data[best_metric_name] = best_metric
                    checkpointer.save('model_best', tag=False, **checkpoint_data)

        return best_metric


def build_optimizer(cfg, model):
    name = cfg.OPTIMIZER.TYPE
    if name == '':
        warnings.warn('No optimizer is built.')
        return None
    elif hasattr(torch.optim, name):
        return getattr(torch.optim, name)(
            model.parameters(),
            lr=cfg.OPTIMIZER.LR,
            weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY,
            **cfg.OPTIMIZER.get(name, dict()),
        )
    else:
        raise ValueError(f'Unsupported optimizer: {name}.')


def build_lr_scheduler(cfg, optimizer):
    name = cfg.LR_SCHEDULER.TYPE
    if name == '':
        warnings.warn('No lr_scheduler is built.')
        return None
    elif hasattr(torch.optim.lr_scheduler, name):
        lr_scheduler = getattr(torch.optim.lr_scheduler, name)(
            optimizer,
            **cfg.LR_SCHEDULER.get(name, dict()),
        )
        return lr_scheduler
    else:
        raise ValueError(f'Unsupported lr_scheduler: {name}.')
