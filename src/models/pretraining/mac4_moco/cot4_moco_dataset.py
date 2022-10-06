import torch
import logging
from mmcv.parallel import DataContainer

from ....builder import DATASETS
from ...classifiers.tsn import TSNDataset
from ....datasets.transforms.tensor import GroupToTensor
import numpy as np


@DATASETS.register_module()
class CoT4MocoDataset(TSNDataset):

    def __init__(self, *args, **kwargs):
        super(CoT4MocoDataset, self).__init__(*args, **kwargs)

        self.maskToTensor = GroupToTensor(switch_rgb_channels=True, div255=True, mean=(0, 0, 0), std=(1, 1, 1))
        # find the index of rotation transformation in the list
        # of image transformations, because we need to get the rotation
        # degree as the ground-truth.
        try:
            self.mix_alpha_trans_index = \
                next(i for i, trans in enumerate(self.img_transform.transforms)
                     if trans.__class__.__name__ == 'GroupMask4Calculation')
        except Exception:
            logger = logging.getLogger()
            logger.error("Cannot find 'GroupMask2Calculation' or 'GroupRetinaMask2Calculation' in "
                         "the image transformation configuration."
                         "It is necessary for CoT2Moco task.")
            raise ValueError

    def get_single_clip(self, storage_obj):
        frame_inds = self.frame_sampler.sample(len(storage_obj))
        num_segs, clip_len = frame_inds.shape
        assert num_segs == 1
        img_list = storage_obj.get_frame(frame_inds.reshape(-1))
        img_tensor = self.img_transform.apply_image(img_list)
        img_tensor = img_tensor.permute(1, 0, 2, 3).contiguous()
        return img_tensor

    def __getitem__(self, idx):
        video_info = self.data_source[idx]
        storage_obj = self.backend.open(video_info)
        frame_inds = self.frame_sampler.sample(len(storage_obj))
        num_segs, clip_len = frame_inds.shape
        assert num_segs == 1, f'support num_segs==1 only, got {num_segs}'

        img_list = storage_obj.get_frame(frame_inds.reshape(-1))
        img_tensor, trans_params = \
            self.img_transform.apply_image(img_list,
                                           return_transform_param=True)

        img_tensor = img_tensor.permute(1, 0, 2, 3).contiguous()

        masks = trans_params[self.mix_alpha_trans_index]['masks']

        mask_tensor = self.maskToTensor.apply_image(masks)

        mask_tensor = mask_tensor.permute(1, 0, 2, 3).contiguous()

        gt_alphas = trans_params[self.mix_alpha_trans_index]['gt_alphas']

        reverse_ground = trans_params[self.mix_alpha_trans_index]["reverse_ground"]

        quad_size = int(clip_len/4)

        for i in range(4):
            img = img_tensor[:, i * quad_size:(i + 1) * quad_size, :, :]
            mask = mask_tensor[:, i * quad_size:(i + 1) * quad_size, :, :]
            if not reverse_ground:
                img_tensor[:, i*quad_size:(i+1)*quad_size, :, :] = img * mask * gt_alphas[i] + img * (1-mask)
            else:
                img_tensor[:, i*quad_size:(i+1)*quad_size, :, :] = img * mask  + img * (1-mask) * gt_alphas[i]


        data = dict(
            imgs=DataContainer(img_tensor,
                               stack=True,
                               pad_dims=2,
                               cpu_only=False),
        )

        if not self.test_mode:

            # imgs_k = self.get_single_clip(storage_obj)

            imgs_k = self.img_transform.apply_image(img_list)
            imgs_k = imgs_k.permute(1, 0, 2, 3).contiguous()

            data['imgs_k'] = DataContainer(imgs_k,
                                           stack=True,
                                           pad_dims=None,
                                           cpu_only=False)
            gt_label = torch.LongTensor([trans_params[self.mix_alpha_trans_index]['gt']])
            data['gt_labels'] = DataContainer(gt_label,
                                              stack=True,
                                              pad_dims=None,
                                              cpu_only=False)

        storage_obj.close()
        return data


    def evaluate_ssl(self, results, labels, logger=None):
        if isinstance(results, list):
            if results[0].ndim == 1:
                results = [r[np.newaxis, ...] for r in results]
            results = np.concatenate(results, axis=0)
        assert len(results) == len(labels), \
            f'The results should have same size as gts. But' \
            f' got {len(results)} and {len(labels)}'

        sort_inds = results.argsort(axis=1)[:, ::-1]

        acc_dict = dict()
        for k in [1, 3]:
            top_k_inds = sort_inds[:, :k]
            correct = (top_k_inds.astype(np.long) ==
                       labels.reshape(len(labels), 1))
            correct_count = np.any(correct, axis=1).astype(np.float32).sum()
            acc = correct_count / len(labels)
            acc_dict[f'top_{k}_acc'] = acc
            if logger is not None:
                logger.info(f'top_{k}_acc: {acc*100}%')

        # mean class accuracy
        per_class_acc = dict()
        for i in range(len(results)):
            class_id = int(labels[i])
            if class_id not in per_class_acc:
                per_class_acc[class_id] = []
            if sort_inds[i, 0] == class_id:
                per_class_acc[class_id].append(1.0)
            else:
                per_class_acc[class_id].append(0.0)
        per_class_acc_list = []
        for k, v in per_class_acc.items():
            per_class_acc_list.append(sum(v) / len(v))
        acc_dict[f'mean_class_acc'] = sum(per_class_acc_list) / \
                                      len(per_class_acc_list)

        return acc_dict

