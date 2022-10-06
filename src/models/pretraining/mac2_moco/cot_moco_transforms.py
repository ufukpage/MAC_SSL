import numpy as np
import random
import cv2
from typing import List

from ....datasets import BaseTransform
from ....builder import TRANSFORMS


def normalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data) + np.finfo(float).eps)


def cvt_uint8(img, is_bgr=True):
    """ convert data type from numpy.float32 to numpy.uint8 """
    nimg = np.round(np.clip(img, 0, 255)).astype(np.uint8)
    if not is_bgr:
        nimg[:, :, 0] = np.clip(nimg[:, :, 0], 0, 179)
    return nimg


@TRANSFORMS.register_module()
class GroupMask2Calculation(BaseTransform):

    def __init__(self, alphas, threshold=30, momentum=0.5, dilate_kernel_size=(2, 2), drop_prob=0.,
                 reverse_ground=False, disable_ht=False, uncertainty=None):
        super(GroupMask2Calculation).__init__()
        self.alphas = alphas
        self.threshold = threshold
        self.momentum = momentum
        self.dilate_kernel_size = dilate_kernel_size
        self.drop_prob = drop_prob
        self.reverse_ground = reverse_ground
        self.disable_ht = disable_ht
        self.uncertainty = uncertainty

    def get_transform_param(self, data, *args, **kwargs) -> dict:
        max_pos = len(self.alphas)**2
        flag = random.randint(0, max_pos-1)
        flag_1 = flag // len(self.alphas)     # flag_1 = 7 // 4 = 1  gt of first 8 frames
        flag_2 = flag % len(self.alphas)      # flag_2 = 7 %  4 = 3  gt of last 8 frames
        drop_out_flags = [(0 < np.random.rand() < self.drop_prob)]
        for ind, _ in enumerate(data):
            if ind == 1: continue

            if drop_out_flags[ind - 1] is not True:
                d_flag = 0 < np.random.rand() < self.drop_prob
                drop_out_flags.append(d_flag)
            else:
                drop_out_flags.append(False)
        return dict(alpha_1=self.alphas[flag_1], alpha_2=self.alphas[flag_2], img_shape=data[0].shape, gt=flag,
                    drop_out_flags=drop_out_flags, reverse_ground=self.reverse_ground)

    def _apply_image(self,
                     data: List[np.ndarray],
                     transform_param: dict):
        return self.calculate_mask(data, transform_param)

    def calculate_mask(self, data_: List[np.ndarray], transform_param: dict):
        beta_img = data_[0]
        beta_img = cv2.cvtColor(beta_img, cv2.COLOR_BGR2GRAY)
        masks = []
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.dilate_kernel_size)

        uc_mask = None
        if self.uncertainty is not None:
            data = np.array([np.dstack([cv2.cvtColor(cvt_uint8(img), cv2.COLOR_BGR2GRAY)]) for img in data_])
            uc_mask = np.std(data, axis=0).astype(dtype=np.uint8)
            uc_mask = (normalizeData(uc_mask) * 255).astype(np.uint8)
            uc_mask = np.repeat(uc_mask, 3, axis=2)

        for ind, img_ in enumerate(data_):
            if transform_param["drop_out_flags"][ind]:
                mask = (np.ones(img_.shape) * 255).astype(np.uint8)
            else:
                img = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)

                diff = cv2.absdiff(img, beta_img)
                diff[diff <= self.threshold] = 0
                if not self.disable_ht:
                    diff[diff > self.threshold] = 255
                else:
                    diff = (normalizeData(diff) * 255.).astype(np.uint8)
                mask = np.dstack([diff] * 3)

                beta_img = np.array(self.momentum * img + (1-self.momentum) * beta_img, dtype=np.uint8)

            mask = cv2.dilate(mask, kernel)
            if self.uncertainty is not None:
                mask = self.uncertainty * uc_mask + (1 - self.uncertainty) * mask
            masks.append(mask)

            """"
            alpha = 0.7
            blended_image = ((img_ * (mask/255)) * alpha).astype(np.uint8) + ((img_ * ((255-mask)/255).astype(np.uint8))) #* (1-alpha)).astype(np.uint8)

            cv2.namedWindow('Mask', 0)
            cv2.namedWindow('Img', 0)
            cv2.namedWindow('Blended', 0)
            cv2.namedWindow('0.4Img', 0)
            cv2.namedWindow('0.7Img', 0)

            cv2.imshow('Mask', mask)
            cv2.imshow('Img', img_)
            cv2.imshow('Blended',blended_image.astype(np.uint8))
            cv2.imshow('0.4Img', (img_*0.4).astype(np.uint8))
            cv2.imshow('0.7Img', (img_*0.7).astype(np.uint8))
            cv2.waitKey(int(100/1))
        cv2.destroyAllWindows()
            #"""

        masks[0] = masks[1] # copy mask2 to mask1 to prevent all zero mask
        transform_param["masks"] = np.array(masks)
        return data_

    def apply_boxes(self,
                    boxes: np.ndarray,
                    transform_param: dict):
        raise NotImplementedError

    def apply_flow(self,
                   flows: List[np.ndarray],
                   transform_param: dict):
        raise NotImplementedError


@TRANSFORMS.register_module()
class GroupMask2NCalculation(GroupMask2Calculation):

    def __init__(self, alpha_n, type='normal', *args, **kwargs):
        super(GroupMask2NCalculation, self).__init__(alphas=[], *args, **kwargs )
        self.step = 1 / alpha_n
        start_cluster_center = self.step / 2
        end_cluster_center = 1 - start_cluster_center
        self.alphas = np.linspace(start_cluster_center, end_cluster_center, num=alpha_n)
        self.type=type

    def get_transform_param(self, data, *args, **kwargs) -> dict:
        params = super(GroupMask2NCalculation, self).get_transform_param(data, *args, **kwargs)

        alpha1 = params['alpha_1']
        alpha2 = params['alpha_2']

        # according to z score, step_size/2 ~= 4 * std, aka 'empirical rule'
        # https://www.investopedia.com/terms/t/three-sigma-limits.asp
        if self.type == 'normal':
            params['alpha_1'] = np.clip(np.random.normal(alpha1, self.step / 8), alpha1 - self.step/4,
                                        alpha1 + self.step/4)
            params['alpha_2'] = np.clip(np.random.normal(alpha2, self.step / 8), alpha2 - self.step/4,
                                        alpha2 + self.step/4)
        else:
            params['alpha_1'] = np.random.uniform(alpha1 - self.step/4, alpha1 + self.step/4)
            params['alpha_2'] = np.random.uniform(alpha2 - self.step/4, alpha2 + self.step/4)

        return params

    def apply_boxes(self,
                    boxes: np.ndarray,
                    transform_param: dict):
        raise NotImplementedError

    def apply_flow(self,
                   flows: List[np.ndarray],
                   transform_param: dict):
        raise NotImplementedError

