
import torch
import torch.nn as nn
from ...train_step_mixin import TrainStepMixin
from ....builder import build_backbone, MODELS
from ..moco.moco_model import MoCo


@MODELS.register_module()
class BeMoCo(MoCo):
    """
    Build a MoCo pt_model_3 with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722

    The code is mainly ported from the official repo:
    https://github.com/facebookresearch/moco

    """
    def __init__(self, *args, **kwargs):
        super(BeMoCo, self).__init__(*args, **kwargs)

    def forward(self, imgs_q, imgs_k, imgs_n=None, imgs=None):

        q = self.backbone(imgs_q)  # [N, C, t, h, w]
        # [N, out_channels]
        q = self.fc(q.view(q.shape[0:2] + (-1,)).mean(dim=-1))
        q = nn.functional.normalize(q, dim=1)
        if imgs_n is not None:
            n = self.backbone(imgs_n)  # [N, C, t, h, w]
            # [N, out_channels]
            n = self.fc(n.view(n.shape[0:2] + (-1,)).mean(dim=-1))
            n = nn.functional.normalize(n, dim=1)

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

        if imgs_n is not None:
            hard_l_neg = torch.einsum('nc,nc->n', [q, n]).unsqueeze(-1)
            logits = torch.cat([l_pos, l_neg, hard_l_neg], dim=1)
        else:
            # logits: Nx(1+K)
            logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        nce_loss = nn.functional.cross_entropy(logits, labels)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return dict(nce_loss=nce_loss)


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
