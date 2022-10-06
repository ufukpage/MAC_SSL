import torch
import torch.nn as nn
from ...train_step_mixin import TrainStepMixin
from ....builder import build_backbone, MODELS
from ...classifiers.tsn.tsn_modules import SimpleClsHead, SimpleSTModule
import numpy as np


@MODELS.register_module()
class CoT2Moco(nn.Module, TrainStepMixin):

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
                 cls_loss_weight=.5,
                 symmetric_loss=False):
        super(CoT2Moco, self).__init__()
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

        self.symmetric_loss = symmetric_loss

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
        *** Only support DistributedDataParallel (DDP) model. ***
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
        *** Only support DistributedDataParallel (DDP) model. ***
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

    def _forward_cls(self, feats: torch.Tensor):
        # backbone network
        if isinstance(feats, (tuple, list)):
            feats = feats[-1]
        if self.st_module is not None:
            feats = self.st_module(feats)  # [NM, C, 1, 1, 1]
        cls_logits = self.cls_head(feats)

        return cls_logits

    def forward_cls(self, feat: torch.Tensor, gt_labels: torch.Tensor):
        cls_logits = self._forward_cls(feat)
        gt_labels = gt_labels.view(-1)
        losses = self.cls_head.loss(cls_logits, gt_labels)
        return losses

    def forward(self, return_loss=True, *args, **kwargs):
        if return_loss:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_test(*args, **kwargs)

    def forward_train(self, imgs, imgs_k, gt_labels):
        if self.nce_loss_weight > 0.:
            if self.symmetric_loss:
                # https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb
                loss_12, q, k2 = self._forward_moco(imgs, imgs_k)
                loss_21, _, k1 = self._forward_moco(imgs_k, imgs)
                nce_loss = loss_12 + loss_21
                k = torch.cat([k1, k2], dim=0)
            else:
                nce_loss, q, k = self._forward_moco(imgs, imgs_k)
        else:
            q = self.backbone(imgs)

        losses = self.forward_cls(q, gt_labels)

        if self.nce_loss_weight > 0:
            # dequeue and enqueue
            self._dequeue_and_enqueue(k)
            losses["nce_loss"] = nce_loss * self.nce_loss_weight

        losses['loss_cls'] = losses['loss_cls'] * self.cls_loss_weight

        return losses

    def _forward_moco(self, imgs, imgs_k):
        q_1 = self.backbone(imgs)  # [N, C, t, h, w]

        q = self.fc(q_1.view(q_1.shape[0:2] + (-1,)).mean(dim=-1))
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

        return nce_loss, q_1, k

    def _forward(self, imgs: torch.Tensor):
        feats = self.backbone(imgs)  # [NM, C, T, H, W]
        return self._forward_cls(feats)

    def forward_test(self, imgs: torch.Tensor, imgs_k) -> np.ndarray:
        with torch.no_grad():
            cls_logits = self._forward(imgs)
            cls_scores = torch.nn.functional.softmax(cls_logits, dim=1)
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


@MODELS.register_module()
class CoTMocoSeparate(CoT2Moco):

    def __init__(self,
                 cls_head_2: dict,
                 *args, **kwargs):
        super(CoT2Moco, self).__init__(*args, **kwargs)

        self.cls_head_2 = SimpleClsHead(**cls_head_2)
        self.cls_head_2.init_weights()

    def _forward_cls_2(self, feats: torch.Tensor):
        # backbone network
        if isinstance(feats, (tuple, list)):
            feats = feats[-1]
        if self.st_module is not None:
            feats = self.st_module(feats)  # [NM, C, 1, 1, 1]
        cls_logits = self.cls_head_2(feats)

        return cls_logits

    def forward_cls(self, feat: torch.Tensor, gt_labels: torch.Tensor):
        gt_labels = gt_labels.view(-1)

        classes = torch.Tensor([4]).to(gt_labels.device)
        gt_label1 = gt_labels // classes
        gt_label2 = gt_labels % classes

        cls_logits = self._forward_cls(feat)
        losses = self.cls_head.loss(cls_logits, gt_label1.long())
        clsloss1 = losses['loss_cls']

        cls_logits2 = self._forward_cls_2(feat)
        losses = self.cls_head_2.loss(cls_logits2, gt_label2.long())
        clsloss2 = losses['loss_cls']

        losses['loss_cls'] = (clsloss1 + clsloss2)
        return losses