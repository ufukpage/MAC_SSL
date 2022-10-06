import torch
import torch.nn as nn
from ...train_step_mixin import TrainStepMixin
from ....builder import build_backbone, MODELS
from ...classifiers.tsn.tsn_modules import SimpleClsHead, SimpleSTModule
import numpy as np
import torch.nn.functional as F


@MODELS.register_module()
class CoT2RegMoco(nn.Module, TrainStepMixin):

    def __init__(self,
                 backbone: dict,
                 st_module: dict,
                 cls_head: dict,
                 in_channels: int,
                 out_channels: int,
                 queue_size: int = 65536,
                 momentum: float = 0.999,
                 temperature: float = 0.07,
                 mlp: bool = False,
                 nce_loss_weight=.5,
                 cls_loss_weight=1.):
        super(CoT2RegMoco, self).__init__()
        self.backbone = build_backbone(backbone)  # q encoder
        self.st_module = SimpleSTModule(**st_module)
        self.cls_head = SimpleClsHead(**cls_head)
        self.init_weights()
        if nce_loss_weight > 0.:
            self.K = queue_size
            self.m = momentum
            self.T = temperature

            # create the encoders
            # num_classes is the output fc dimension

            self.key_encoder = build_backbone(backbone)  # key encoder

            if mlp:
                self.fc = nn.Sequential(
                    nn.Linear(in_channels, in_channels),
                    nn.ReLU(),
                    nn.Linear(in_channels, out_channels)
                )
                self.key_fc = nn.Sequential(
                    nn.Linear(in_channels, in_channels),
                    nn.ReLU(),
                    nn.Linear(in_channels, out_channels)
                )
            else:
                self.fc = nn.Linear(in_channels, out_channels)
                self.key_fc = nn.Linear(in_channels, out_channels)

            for param_q, param_k in zip(self.backbone.parameters(),
                                        self.key_encoder.parameters()):
                param_k.data.copy_(param_q.data)  # initialize
                param_k.requires_grad = False  # not update by gradient
            for param_q, param_k in zip(self.fc.parameters(),
                                        self.key_fc.parameters()):
                param_k.data.copy_(param_q.data)  # initialize
                param_k.requires_grad = False  # not update by gradient

            self.register_buffer("queue", torch.randn(out_channels, queue_size))
            self.queue = nn.functional.normalize(self.queue, dim=0)

            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.cls_loss_weight = cls_loss_weight
        self.nce_loss_weight = nce_loss_weight

    def init_weights(self):
        self.backbone.init_weights()
        if hasattr(self, 'st_module'):
            self.st_module.init_weights()
        if hasattr(self, 'cls_head'):
            self.cls_head.init_weights()

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.backbone.parameters(),
                                    self.key_encoder.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.fc.parameters(),
                                    self.key_fc.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.transpose(0, 1)
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) pt_model_3. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) pt_model_3. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def loss(self,
             output,
             labels):
        if output.dim() == 3 and labels.dim() == 1:
            batch_size, num_segs, _ = output.size()
            assert batch_size  * 2 == labels.size(0)
            labels = labels.view(batch_size, 2)
            labels = labels.repeat([1, num_segs]).contiguous()
            output = output.view(batch_size * num_segs, -1)

        losses = dict()
        losses['loss_reg'] = F.mse_loss(torch.sigmoid(output), labels)
        return losses

    def _forward_reg(self, feats: torch.Tensor):
        # backbone network
        if isinstance(feats, (tuple, list)):
            feats = feats[-1]
        if self.st_module is not None:
            feats = self.st_module(feats)  # [NM, C, 1, 1, 1]
        cls_logits = self.cls_head(feats)

        return cls_logits

    def forward_reg(self,
                      feat: torch.Tensor,
                      gt_labels: torch.Tensor):
        cls_logits = self._forward_reg(feat)
        losses = self.loss(cls_logits, gt_labels)
        return losses

    def forward(self, return_loss=True, *args, **kwargs):
        if return_loss:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_test(*args, **kwargs)

    def forward_train(self, imgs, imgs_k, gt_labels):

        q = self.backbone(imgs)  # [N, C, t, h, w]
        losses = self.forward_reg(q, gt_labels)

        q = self.fc(q.view(q.shape[0:2] + (-1,)).mean(dim=-1))
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(imgs_k)

            k = self.key_encoder(im_k)  # keys: NxC
            k = self.key_fc(k.view(k.shape[0:2] + (-1,)).mean(dim=-1))
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        nce_loss = nn.functional.cross_entropy(logits, labels)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)
        losses["nce_loss"] = nce_loss * self.nce_loss_weight
        return losses

    def _forward(self, imgs: torch.Tensor):
        feats = self.backbone(imgs)  # [NM, C, T, H, W]
        return self._forward_cls(feats)

    def forward_test(self, imgs: torch.Tensor, imgs_k) -> np.ndarray:
        with torch.no_grad():
            cls_logits = self._forward(imgs)
            cls_scores = torch.sigmoid(cls_logits)
            cls_scores = cls_scores.cpu().numpy()
        return cls_scores

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


