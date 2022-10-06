import numpy as np
import random
import cv2
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


@TRANSFORMS.register_module()
class GroupMask4Calculation(GroupMask2Calculation):

    def __init__(self, *args, **kwargs):
        super(GroupMask4Calculation, self).__init__(*args, **kwargs)

    def get_transform_param(self, data, *args, **kwargs) -> dict:
        len_cls = len(self.alphas)
        flag = random.randint(0, (len_cls**4 - 1))
        flag1 = flag // (len_cls**2)
        flag1_1 = flag1 // len_cls     # flag_1 = 7 // 4 = 1  gt of first 8 frames
        flag1_2 = flag1 % len_cls     # flag_2 = 7 %  4 = 3  gt of last 8 frames

        flag2 = flag % (len_cls**2)
        flag2_1 = flag2 // len_cls    # flag_1 = 7 // 4 = 1  gt of first 8 frames
        flag2_2 = flag2 % len_cls     # flag_2 = 7 %  4 = 3  gt of last 8 frames

        drop_out_flags = [(0 < np.random.rand() < self.drop_prob)]
        for ind, _ in enumerate(data):
            if ind == 1: continue

            if drop_out_flags[ind - 1] is not True:
                d_flag = 0 < np.random.rand() < self.drop_prob
                drop_out_flags.append(d_flag)
            else:
                drop_out_flags.append(False)
        return dict(gt_alphas=[self.alphas[flag1_1], self.alphas[flag1_2], self.alphas[flag2_1], self.alphas[flag2_2]],
                    img_shape=data[0].shape, gt=flag, drop_out_flags=drop_out_flags, reverse_ground=self.reverse_ground)

    def _apply_image(self,
                     data: List[np.ndarray],
                     transform_param: dict):
        return self.calculate_mask(data, transform_param)

    def apply_boxes(self,
                    boxes: np.ndarray,
                    transform_param: dict):
        raise NotImplementedError

    def apply_flow(self,
                   flows: List[np.ndarray],
                   transform_param: dict):
        raise NotImplementedError
