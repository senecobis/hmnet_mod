
import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base.blocks import BlockBase
from ..base.init import init_transformer
from ..base.backbone.builder import build_backbone
from ..base.neck.builder import build_neck
from ..base.head.builder import build_head

from .hmseg import HMSeg
from hmnet.utils.common import Timer

from torch import Tensor
from typing import Tuple, List, Optional, Dict

class HMSegVQ(HMSeg):
    def __init__(self, backbone, neck, seg_head, aux_head, devices, test_aug=None) -> None:
        super().__init__(backbone, neck, seg_head, aux_head, devices, test_aug=test_aug)
        self.backbone = build_backbone(backbone)
    
    def forward(self, list_events, list_images, list_image_metas, list_gt, init_states=True) -> Tensor:
        """Here we override the forward function to return quantization losses as well
        """
        if init_states:
            self.idx_offset = 0

        # Gather gts for loss calculation
        gather_indices = self._identify_required_outputs_batch(list_gt, self.idx_offset)
        out_image_metas = self._gather(list_image_metas, gather_indices)
        out_gt = self._gather(list_gt, gather_indices)
        out_gt = torch.stack(out_gt)

        # Backbone
        outputs, quant_loss = self.backbone(list_events, 
                                list_image_metas, 
                                gather_indices, 
                                list_images=list_images, 
                                init_states=init_states, 
                                detach=True, 
                                fast_training=True
                                )

        # Neck, Head
        loss, log_vars = self._forward_head(outputs, out_image_metas, out_gt, gather_indices['batch'])

        # Make sure all parameters are involved in loss calculation
        # e.g., parameters for event embedding will not be used when len(events)==0
        loss = loss + sum([ 0. * params.sum() for params in self.parameters() ]) + quant_loss

        outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(out_image_metas))

        self.idx_offset += len(list_events)

        return outputs
    
