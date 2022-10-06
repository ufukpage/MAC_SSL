import torch
import logging
from mmcv.parallel import DataContainer

from ....builder import DATASETS
from ...classifiers.tsn import TSNDataset
from ....datasets.transforms.tensor import GroupToTensor
import numpy as np


@DATASETS.register_module()
class CoT2MocoDataset(TSNDataset):

    def __init__(self, *args, **kwargs):
        super(CoT2MocoDataset, self).__init__(*args, **kwargs)

        self.maskToTensor = GroupToTensor(switch_rgb_channels=True, div255=True, mean=(0, 0, 0), std=(1, 1, 1))
        # find the index of rotation transformation in the list
        # of image transformations, because we need to get the rotation
        # degree as the ground-truth.
        try:
            self.mix_alpha_trans_index = \
                next(i for i, trans in enumerate(self.img_transform.transforms)
                     if trans.__class__.__name__ == 'GroupMask2Calculation' or
                     trans.__class__.__name__ == 'GroupMask2NCalculation')
        except Exception:
            logger = logging.getLogger()
            logger.error("Cannot find 'GroupMask2Calculation' or 'GroupMask2NCalculation' in "
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

        alpha1 = trans_params[self.mix_alpha_trans_index]['alpha_1']
        alpha2 = trans_params[self.mix_alpha_trans_index]['alpha_2']

        reverse_ground = trans_params[self.mix_alpha_trans_index]["reverse_ground"]

        half_size = int(clip_len/2)
        if reverse_ground:
            img_tensor[:, 0:half_size, :, :] = \
                (img_tensor[:, 0:half_size, :, :] * mask_tensor[:, 0:half_size, :, :]) + \
                img_tensor[:, 0:half_size, :, :] * (1 - mask_tensor[:, 0:half_size, :, :]) * alpha1
            img_tensor[:, half_size:clip_len, :, :] = \
                (img_tensor[:, half_size:clip_len, :, :] * mask_tensor[:, half_size:clip_len, :, :])  \
                + img_tensor[:, half_size:clip_len, :, :] * (1 - mask_tensor[:, half_size:clip_len, :, :]) * alpha2
        else:
            img_tensor[:, 0:half_size, :, :] = \
                (img_tensor[:, 0:half_size, :, :] * mask_tensor[:, 0:half_size, :, :]) * alpha1 + \
                img_tensor[:, 0:half_size, :, :] * (1 - mask_tensor[:, 0:half_size, :, :])
            img_tensor[:, half_size:clip_len, :, :] = \
                (img_tensor[:, half_size:clip_len, :, :] * mask_tensor[:, half_size:clip_len, :, :]) * alpha2 \
                + img_tensor[:, half_size:clip_len, :, :] * (1 - mask_tensor[:, half_size:clip_len, :, :])

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


@DATASETS.register_module()
class CoT2MocoMiniK400BBoxDataset(CoT2MocoDataset):

    def __init__(self, *args, **kwargs):
        super(CoT2MocoMiniK400BBoxDataset, self).__init__(*args, **kwargs)

    def __getitem__(self, idx):
        video_info = self.data_source[idx]
        storage_obj = self.backend.open(video_info)

        frame_inds = self.frame_sampler.sample(len(storage_obj))
        num_segs, clip_len = frame_inds.shape
        assert num_segs == 1, f'support num_segs==1 only, got {num_segs}'

        all_bboxes = storage_obj.get_bbox()
        bbox_list = []
        for ind in frame_inds[0]:
            if (ind+1) in all_bboxes:
                bbox_list.append(all_bboxes[ind+1])
            else:
                bbox_list.append(-1)

        try:
            img_list = storage_obj.get_frame(frame_inds.reshape(-1))
        except:
            raise Exception('video_info[name]:' + video_info['name'] + " len(frames):" + str(video_info['boxes']) +
                            ' ordered_boxes:', list(all_bboxes.keys()))

        fg_masks = list()
        bg_masks = list()
        for ind, bbox in enumerate(bbox_list):

            img = img_list[ind]
            fg_mask = np.zeros(img.shape, img.dtype)
            if bbox == -1: # no foreground found
                fg_masks.append(fg_mask)
                bg_masks.append(img)
                continue

            for box_ in bbox:
                box = np.array(box_).astype(int)
                roi = img[box[1]: box[3], box[0]: box[2]]

                fg_mask[box[1]: box[3], box[0]: box[2], :] = roi
            # roi = img[y:y + h, x:x + w]

            bg_mask = img - fg_mask
            fg_masks.append(fg_mask)
            bg_masks.append(bg_mask)
            """"
            # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 1)
            # cv2.imshow('{}.png'.format(ind), roi)
            cv2.imshow('FD', fg_mask)
            cv2.imshow('BG', bg_mask)
            cv2.waitKey(int(1000 / 1))
            #"""

        fg_tensor, trans_params = self.img_transform.apply_image(fg_masks, return_transform_param=True)
        reverse_ground = trans_params[self.mix_alpha_trans_index]["reverse_ground"]

        bg_tensor = bg_masks.copy()
        for ind, transform in enumerate(self.img_transform.transforms):
            if ind != self.mix_alpha_trans_index:
                bg_tensor = transform.apply_image(bg_tensor, trans_params[ind])

        alpha1 = trans_params[self.mix_alpha_trans_index]['alpha_1']
        alpha2 = trans_params[self.mix_alpha_trans_index]['alpha_2']

        # bg_tensor = bg_tensor.view((num_segs, clip_len) + bg_tensor.shape[1:])
        bg_tensor = bg_tensor.permute(1, 0, 2, 3).contiguous()
        # fg_tensor = fg_tensor.view((num_segs, clip_len) + fg_tensor.shape[1:])
        fg_tensor = fg_tensor.permute(1, 0, 2, 3).contiguous()

        half_size = int(clip_len / 2)

        if not reverse_ground:
            bg_tensor[:, 0:half_size, :, :] = \
                fg_tensor[:, 0:half_size, :, :] * alpha1 + bg_tensor[:, 0:half_size, :, :]
            bg_tensor[:, half_size:clip_len, :, :] = \
                fg_tensor[:, half_size:clip_len, :, :] * alpha2 + bg_tensor[:, half_size:clip_len, :, :]
        else:
            bg_tensor[:, 0:half_size, :, :] = \
                fg_tensor[:, 0:half_size, :, :] + bg_tensor[:, 0:half_size, :, :] * alpha1
            bg_tensor[:, half_size:clip_len, :, :] = \
                fg_tensor[:, half_size:clip_len, :, :] + bg_tensor[:, half_size:clip_len, :, :] * alpha2

        img_tensor = bg_tensor

        """"
        for index in range(img_tensor.shape[2]):
            img = img_tensor[0, index, :, :]
            #plt.imshow(img.permute(1, 2, 0).to(torch.uint8))
            #plt.imshow(img.numpy()) #.astype(np.uint8))
            #plt.imshow(mask_tensor[0, 0, index, :, :].numpy())
            cv2.imshow('FG', cv2.resize(fg_tensor[0, index, :, :].numpy(), (256, 256)))
            cv2.imshow('MAC', cv2.resize(img.numpy(), (256, 256)))
            cv2.waitKey(int(100 / 1))
            #plt.show()
        # """
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