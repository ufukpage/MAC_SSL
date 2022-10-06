import numpy as np
import random
from typing import List

from ....builder import TRANSFORMS
from ..mac2_moco.mac_moco_transforms import GroupMask2Calculation

def normalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data) + np.finfo(float).eps)


def cvt_uint8(img, is_bgr=True):
    """ convert data type from numpy.float32 to numpy.uint8 """
    nimg = np.round(np.clip(img, 0, 255)).astype(np.uint8)
    if not is_bgr:
        nimg[:, :, 0] = np.clip(nimg[:, :, 0], 0, 179)
    return nimg


def clamp_normal(value, max=1., clamp=1.5):
    if value > max:
        return abs(value - clamp)
    return value


@TRANSFORMS.register_module()
class GroupMask2nRegCalculation(GroupMask2Calculation):

    def __init__(self, dist_type='uniform', *args, **kwargs):
        super(GroupMask2nRegCalculation, self).__init__(*args, **kwargs)
        self.dist_type = dist_type
        self.mu = 1.
        self.sigma = 0.5

    def get_transform_param(self, data,  *args, **kwargs) -> dict:
        if self.dist_type == 'uniform':
            flag_1 = np.random.uniform()
            np.random.seed(random.randint(0, 167384000))
            flag_2 = np.random.uniform()
        else:
            flag_1 = clamp_normal(np.random.normal(self.mu, self.sigma))
            np.random.seed(random.randint(0, 167384000))
            flag_2 = clamp_normal(np.random.normal(self.mu, self.sigma))

        drop_out_flags = [(0 < np.random.rand() < self.drop_prob)]
        for ind, _ in enumerate(data):
            if ind == 1: continue

            if drop_out_flags[ind - 1] is not True:
                d_flag = 0 < np.random.rand() < self.drop_prob
                drop_out_flags.append(d_flag)
            else:
                drop_out_flags.append(False)
        return dict(alpha_1=flag_1, alpha_2=flag_2, img_shape=data[0].shape,
                    drop_out_flags=drop_out_flags, reverse_ground=self.reverse_ground)

    def apply_boxes(self,
                    boxes: np.ndarray,
                    transform_param: dict):
        raise NotImplementedError

    def apply_flow(self,
                   flows: List[np.ndarray],
                   transform_param: dict):
        raise NotImplementedError