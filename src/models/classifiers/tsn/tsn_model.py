
import numpy as np
import torch
import torch.nn as nn
from typing import Dict
from .tsn_modules import SimpleClsHead, SimpleSTModule
from ...train_step_mixin import TrainStepMixin
from ....builder import MODELS, build_backbone


@MODELS.register_module()
class TSN(nn.Module, TrainStepMixin):
    """ TSN action recognition model.
    Mainly ported from https://github.com/open-mmlab/mmaction project.
    """

    def __init__(self,
                 backbone: dict,
                 st_module: dict,
                 cls_head: dict,
                 linear_probe=False):
        super(TSN, self).__init__()
        self.backbone = build_backbone(backbone)
        self.st_module = SimpleSTModule(**st_module)
        self.cls_head = SimpleClsHead(**cls_head)
        self.init_weights()
        self.linear_probe = linear_probe

    def init_weights(self):
        self.backbone.init_weights()
        if hasattr(self, 'st_module'):
            self.st_module.init_weights()
        if hasattr(self, 'cls_head'):
            self.cls_head.init_weights()

    @torch.no_grad()
    def _forward_retrieve(self, imgs):
        batch_size = imgs.size(0)
        num_segs = imgs.size(1)
        # unsqueeze the first dimension
        imgs = imgs.view((-1,) + imgs.shape[2:])
        # backbone network
        feats = self.backbone(imgs)  # [NM, C, T, H, W]
        if isinstance(feats, (tuple, list)):
            feats = feats[-1]

        feats = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))(
            feats)  # nn.AdaptiveAvgPool3d((1, 1, 1))(feats).squeeze() #
        return feats.view(batch_size, num_segs, -1).mean(dim=1)

    def forward(self, return_loss=True, retrieve_mode=False, *args, **kwargs):
        if retrieve_mode:
            return self._forward_retrieve(*args, **kwargs)
        elif return_loss:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_test(*args, **kwargs)

    def _forward(self, imgs: torch.Tensor):
        """ Predict the classification results of the given video clip.
        Args:
            imgs (torch.Tensor): RGB image data in shape of [N, M, C, T, H, W]
        Returns:
            cls_logits (torch.Tensor): classification results,
                in shape of [N, M, num_class]
        """
        batch_size = imgs.size(0)
        num_segs = imgs.size(1)
        # unsqueeze the first dimension
        imgs = imgs.view((-1, ) + imgs.shape[2:])

        # backbone network
        if self.linear_probe:
            with torch.no_grad():
                feats = self.backbone(imgs)  # [NM, C, T, H, W]
        else:
            feats = self.backbone(imgs)  # [NM, C, T, H, W]

        if isinstance(feats, (tuple, list)):
            feats = feats[-1]
        if self.st_module is not None:
            feats = self.st_module(feats)  # [NM, C, 1, 1, 1]

        if self.linear_probe:
            feats = nn.functional.normalize(feats, dim=1, p=2)
        cls_logits = self.cls_head(feats)
        cls_logits = cls_logits.view(batch_size, num_segs, -1)
        return cls_logits

    def forward_train(self,
                      imgs: torch.Tensor,
                      gt_labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """ Forward 3D-Net and then return the losses
        Args:
            imgs (torch.Tensor): RGB image data in shape of [N, M, C, T, H, W]
            gt_labels (torch.Tensor): ground-truth label in shape of [N, 1]
        """
        cls_logits = self._forward(imgs)
        gt_labels = gt_labels.view(-1)
        losses = self.cls_head.loss(cls_logits, gt_labels)
        return losses

    def forward_test(self, imgs: torch.Tensor) -> np.ndarray:
        """ Forward 3D-Net and then return the classification results

        Args:
            imgs (torch.Tensor): RGB image data in shape of [N, M, C, T, H, W]

        Returns:
            cls_scores (np.ndarray): in shape of [N, num_cls]

        """
        with torch.no_grad():
            cls_logits = self._forward(imgs)
            # average the classification logit
            cls_logits = cls_logits.mean(dim=1)
            cls_scores = torch.nn.functional.softmax(cls_logits, dim=1)
            cls_scores = cls_scores.cpu().numpy()
        return cls_scores
