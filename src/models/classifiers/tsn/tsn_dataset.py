
import torch
import numpy as np
from torch.utils.data import Dataset
from mmcv.parallel import DataContainer

from ....datasets.transforms import Compose
from ....datasets import builder
from ....builder import DATASETS
import mmcv
import os
from mmcv.runner import get_dist_info
import cv2
import collections


@DATASETS.register_module()
class TSNDataset(Dataset):

    def __init__(self,
                 data_dir: str,
                 data_source: dict,
                 backend: dict,
                 frame_sampler: dict,
                 transform_cfg: list,
                 test_mode: bool,
                 name: str = None):
        if name is None:
            name = 'undefined_dataset'
        self.name = name
        self.data_dir = data_dir
        self.data_source = builder.build_data_source(data_source,
                                                     dict(data_dir=data_dir))
        self.backend = builder.build_backend(backend,
                                             dict(data_dir=data_dir))
        self.frame_sampler = builder.build_frame_sampler(frame_sampler)
        self.img_transform = Compose(transform_cfg)
        self.test_mode = test_mode
        self.retrieve_mode = None

    def __len__(self):
        return len(self.data_source)

    def set_retrieve_mode(self, retrieve_mode):
        self.retrieve_mode = retrieve_mode

    def __getitem__(self, idx):
        video_info = self.data_source[idx]
        storage_obj = self.backend.open(video_info)

        frame_inds = self.frame_sampler.sample(len(storage_obj))
        num_segs, clip_len = frame_inds.shape

        img_list = storage_obj.get_frame(frame_inds.reshape(-1))
        img_tensor_list = []
        for i in range(num_segs):
            raw_imgs = img_list[i*clip_len:(i+1)*clip_len]
            img_tensor = self.img_transform.apply_image(raw_imgs)
            img_tensor_list.append(img_tensor)

        img_tensor = torch.cat(img_tensor_list, dim=0)
        # img_tensor: (M, C, H, W) M = N_seg * L
        img_tensor = img_tensor.view((num_segs, clip_len) +
                                     img_tensor.shape[1:])
        img_tensor = img_tensor.permute(0, 2, 1, 3, 4).contiguous()
        # img_tensor: [N_seg, 3, L, H, W]
        data = dict(
            imgs=DataContainer(img_tensor, stack=True, cpu_only=False)
        )
        if not self.test_mode:
            gt_label = torch.LongTensor([video_info['label']]) - 1
            data['gt_labels'] = DataContainer(gt_label,
                                              stack=True,
                                              pad_dims=None,
                                              cpu_only=False)
        if self.retrieve_mode:
            data['index'] = DataContainer(torch.LongTensor([idx]),
                                                            stack=True,
                                                            pad_dims=None,
                                                            cpu_only=False)

        return data

    def evaluate(self, results, logger=None, output_dir=None):
        if isinstance(results, list):
            if results[0].ndim == 1:
                results = [r[np.newaxis, ...] for r in results]
            results = np.concatenate(results, axis=0)
        assert len(results) == len(self), \
            f'The results should have same size as gts. But' \
            f' got {len(results)} and {len(self)}'
        labels = np.array([int(self.data_source[_]['label']) - 1
                           for _ in range(len(self))], np.long)
        sort_inds = results.argsort(axis=1)[:, ::-1]

        acc_dict = dict()
        for k in [1, 5]:
            top_k_inds = sort_inds[:, :k]
            correct = (top_k_inds.astype(np.long) ==
                       labels.reshape(len(self), 1))
            correct_count = np.any(correct, axis=1).astype(np.float32).sum()
            acc = correct_count / len(self)
            acc_dict[f'top_{k}_acc'] = acc
            if logger is not None:
                logger.info(f'top_{k}_acc: {acc * 100}%')

        nb_classes = results.shape[1]

        confusion_matrix = torch.zeros(nb_classes, nb_classes)
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
            confusion_matrix[class_id, sort_inds[i, 0]] += 1
        per_class_acc_list = []

        for k, v in per_class_acc.items():
            per_class_acc_list.append(sum(v) / len(v))
        acc_dict[f'mean_class_acc'] = sum(per_class_acc_list) / \
                                      len(per_class_acc_list)

        if output_dir:
            rank, world_size = get_dist_info()
            if rank == 0:
                print(confusion_matrix)
                mmcv.dump(confusion_matrix, os.path.join(output_dir, f'confusion_matrix.pkl'))
                mmcv.dump(per_class_acc_list, os.path.join(output_dir, f'per_class_acc_list.pkl'))
                mmcv.dump(per_class_acc_list, os.path.join(output_dir,
                                                     f'per_class_acc_list.json'))

        return acc_dict


@DATASETS.register_module()
class TSNBiasDataset(TSNDataset):

    def __init__(self, vid_len_divider=3, sampler_divider=4, *args, **kwargs):
        super(TSNBiasDataset, self).__init__(*args, **kwargs)
        self.vid_len_divider = vid_len_divider
        self.sampler_divider = sampler_divider

    def __getitem__(self, idx):
        video_info = self.data_source[idx]
        storage_obj = self.backend.open(video_info)

        frame_inds = self.frame_sampler.sample(int(len(storage_obj)//self.vid_len_divider))
        num_segs, clip_len = frame_inds.shape

        img_list = storage_obj.get_frame(frame_inds.reshape(-1))
        img_tensor_list = []
        for i in range(num_segs):
            raw_imgs = img_list[i*clip_len:(i+1)*clip_len]
            img_tensor = self.img_transform.apply_image(raw_imgs)
            t, c, h, w = img_tensor.size()
            part_t = t // self.sampler_divider
            for ind in range(part_t):
                stat_image = img_tensor[ind*self.sampler_divider]
                img_tensor[ind * self.sampler_divider:(ind+1) * self.sampler_divider] = stat_image
            img_tensor_list.append(img_tensor)

        img_tensor = torch.cat(img_tensor_list, dim=0)
        # img_tensor: (M, C, H, W) M = N_seg * L
        img_tensor = img_tensor.view((num_segs, clip_len) +
                                     img_tensor.shape[1:])
        img_tensor = img_tensor.permute(0, 2, 1, 3, 4).contiguous()
        # img_tensor: [N_seg, 3, L, H, W]
        data = dict(
            imgs=DataContainer(img_tensor, stack=True, cpu_only=False)
        )
        if not self.test_mode:
            gt_label = torch.LongTensor([video_info['label']]) - 1
            data['gt_labels'] = DataContainer(gt_label,
                                              stack=True,
                                              pad_dims=None,
                                              cpu_only=False)
        if self.retrieve_mode:
            data['index'] = DataContainer(torch.LongTensor([idx]),
                                                            stack=True,
                                                            pad_dims=None,
                                                            cpu_only=False)

        return data


@DATASETS.register_module()
class TSNActorDataset(TSNDataset):

    def __init__(self, *args, **kwargs):
        super(TSNActorDataset, self).__init__(*args, **kwargs)

    def __getitem__(self, idx):
        video_info = self.data_source[idx]
        storage_obj = self.backend.open(video_info)

        frame_inds = self.frame_sampler.sample(len(storage_obj))
        num_segs, clip_len = frame_inds.shape

        all_bboxes = storage_obj.get_bbox()

        bbox_list = []
        for seg in range(num_segs):
            seg_ind = frame_inds[seg]
            seg_bbox_list = []
            for ind in seg_ind:
                if (ind + 1) in all_bboxes:
                    bb_list = all_bboxes[(ind + 1)]
                    bbox = np.zeros(4)
                    bbox[0] = np.inf
                    bbox[1] = np.inf
                    for bb in bb_list:
                        if bb[0] < bbox[0]:
                            bbox[0] = bb[0]
                        if bb[1] < bbox[1]:
                            bbox[1] = bb[1]
                        if bb[2] > bbox[2]:
                            bbox[2] = bb[2]
                        if bb[3] > bbox[3]:
                            bbox[3] = bb[3]
                    seg_bbox_list.append(bbox)
                else:
                    seg_bbox_list.append(-1)
            bbox_list.append(seg_bbox_list)

        img_list = []
        for seg in range(num_segs):
            img_list.extend(storage_obj.get_frame(frame_inds[seg]))
        img_tensor_list = []
        for i in range(num_segs):
            raw_imgs = img_list[i*clip_len:(i+1)*clip_len]
            for ind in range(len(raw_imgs)):
                raw_image = raw_imgs[ind]
                actor_image = raw_image
                box = bbox_list[i][ind]
                if isinstance(box, np.ndarray):
                    box = box.astype(int)
                    actor_image = raw_image[box[1]: box[3], box[0]: box[2], :]

                if not np.array(actor_image.shape).all(): # check covering bounding box, some are greater than image size
                    actor_image = raw_image
                try:
                    raw_imgs[ind] = cv2.resize(actor_image, dsize=(raw_image.shape[1], raw_image.shape[0]),
                                                interpolation=cv2.INTER_CUBIC)
                except Exception as e:
                    print('EXCEPTION: ' + str(e))
                    raise Exception(
                        'video_info[name]:' + video_info['name'] + " len(frames):" + str(video_info['boxes']) +
                        ' raw_image.shape:', list(raw_image.shape), ' actor_image.shape:', list(actor_image.shape),
                        'bbox:', box)
                """"
                cv2.imshow('raw', raw_image)
                cv2.imshow('actor', raw_imgs[ind])
                cv2.waitKey(int(1000 / 5))
                #"""

            img_tensor = self.img_transform.apply_image(raw_imgs)
            img_tensor_list.append(img_tensor)

        img_tensor = torch.cat(img_tensor_list, dim=0)
        # img_tensor: (M, C, H, W) M = N_seg * L
        img_tensor = img_tensor.view((num_segs, clip_len) +
                                     img_tensor.shape[1:])
        img_tensor = img_tensor.permute(0, 2, 1, 3, 4).contiguous()
        # img_tensor: [N_seg, 3, L, H, W]
        data = dict(
            imgs=DataContainer(img_tensor, stack=True, cpu_only=False)
        )
        if not self.test_mode:
            gt_label = torch.LongTensor([video_info['label']]) - 1
            data['gt_labels'] = DataContainer(gt_label,
                                              stack=True,
                                              pad_dims=None,
                                              cpu_only=False)
        if self.retrieve_mode:
            data['index'] = DataContainer(torch.LongTensor([idx]),
                                                            stack=True,
                                                            pad_dims=None,
                                                            cpu_only=False)

        return data