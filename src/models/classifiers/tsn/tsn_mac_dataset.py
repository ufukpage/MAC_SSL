import torch
import numpy as np
from torch.utils.data import Dataset
from mmcv.parallel import DataContainer

from ....datasets.transforms import Compose
from ....datasets import builder
from ....builder import DATASETS
from .tsn_dataset import TSNDataset


@DATASETS.register_module()
class TSNMACDataset(TSNDataset):

    def __init__(self,
                 data_dir: str,
                 data_source: dict,
                 backend: dict,
                 frame_sampler: dict,
                 transform_cfg: list,
                 test_mode: bool,
                 name: str = None):
        super(TSNDataset, self).__init__(data_dir=data_dir, data_source=data_source, backend=backend,
                                         frame_sampler=frame_sampler, transform_cfg=transform_cfg, test_mode=test_mode,
                                         name=name)

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
        alpha = trans_params[self.mix_alpha_trans_index]['alpha']
        img_tensor = (img_tensor * mask_tensor) * alpha + img_tensor * (1-mask_tensor)

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