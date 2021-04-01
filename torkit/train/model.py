from typing import Dict

import torch
from torch.utils.tensorboard import SummaryWriter
from .metric import Metric


class GenericModel(torch.nn.Module):
    """A generic model class."""

    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pred_dict = dict()
        return pred_dict

    def compute_losses(self,
                       pred_dict: Dict[str, torch.Tensor],
                       data_batch: Dict[str, torch.Tensor],
                       ) -> Dict[str, torch.Tensor]:
        loss_dict = dict()
        return loss_dict

    def get_metrics(self, training=True) -> Dict[str, Metric]:
        metrics = dict()
        return metrics

    def update_metrics(self,
                       pred_dict: Dict[str, torch.Tensor],
                       data_batch: Dict[str, torch.Tensor],
                       metrics: Dict[str, Metric],
                       training=True,
                       ):
        pass

    def summarize(self,
                  pred_dict: Dict[str, torch.Tensor],
                  data_batch: Dict[str, torch.Tensor],
                  metrics: Dict[str, Metric],
                  summary_writer: SummaryWriter,
                  global_step: int,
                  training=True):
        pass
