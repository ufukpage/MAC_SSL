import torch
import torch.nn as nn
from ....builder import MODELS
import torch.nn.functional as F
from ..mac2reg_moco.mac2reg_moco_model import CoT2RegMoco


@MODELS.register_module()
class CoT2nRegMoco(CoT2RegMoco):

    def __init__(self,
                 *args, **kwargs):
        super(CoT2nRegMoco, self).__init__(*args, **kwargs)

    def forward_train(self, imgs, imgs_k, gt_labels):

        q = self.backbone(imgs)  # [N, C, t, h, w]

        losses = self.forward_reg(q, gt_labels)
        if self.nce_loss_weight > 0:
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


