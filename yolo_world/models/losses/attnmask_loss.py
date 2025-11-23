# Copyright (c) Tencent Inc. All rights reserved.
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from mmdet.models.losses.mse_loss import mse_loss
from mmyolo.registry import MODELS

@MODELS.register_module()
class BCELoss(nn.Module):
    """BCELoss.

    This class wraps the PyTorch binary cross-entropy loss function.

    Args:
        reduction (str): Options are "none", "mean", and "sum".
            Defaults to "mean".
        loss_weight (float): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self,
                 reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        assert self.reduction in (None, 'none', 'mean', 'sum')

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Forward function.

        Args:
            pred (torch.Tensor): The predictions. Shape (N, *).
            target (torch.Tensor): The targets. Shape (N, *).
            
        Returns:
            torch.Tensor: The calculated loss.
        """

        loss = F.binary_cross_entropy(pred, target, reduction=self.reduction)
        
        return loss
