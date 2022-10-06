
import numpy as np
import cv2
from typing import List
from .base_transform import BaseTransform
from .dynamic_utils import sample_key_frames, extend_key_frame_to_all
from ...builder import TRANSFORMS
# import imgaug.augmenters as iaa
from PIL import ImageOps, Image, ImageFilter
import random


@TRANSFORMS.register_module()
class RandomBrightness(BaseTransform):

    def __init__(self,
                 prob: float,
                 delta: float):
        self.brightness_prob = prob
        self.brightness_delta = delta

    def get_transform_param(self, *args, **kwargs) -> dict:
        flag = (0 < np.random.rand() < self.brightness_prob)
        delta = np.random.uniform(-self.brightness_delta, self.brightness_delta)
        return dict(flag=flag, delta=delta)

    def _apply_image(self,
                     data: List[np.ndarray],
                     transform_param: dict):
        if transform_param['flag']:
            delta = transform_param['delta']
            data = [np.clip(img + delta, a_min=0, a_max=255) for img in data]
        return data


@TRANSFORMS.register_module()
class GroupMaskBlend(BaseTransform):

    def __init__(self, alphas, threshold, momentum, dilate_kernel_size, blend_prob, drop_prob):
        super(GroupMaskBlend).__init__()
        self.alphas = alphas
        self.threshold = threshold
        self.momentum = momentum
        self.dilate_kernel_size = dilate_kernel_size
        self.drop_prob = drop_prob
        self.blend_prob = blend_prob

    def get_transform_param(self, data, *args, **kwargs) -> dict:
        prob_flag = (0 < np.random.rand() < self.blend_prob)
        flag = random.randint(0, 3)

        drop_out_flags = [(0 < np.random.rand() < self.drop_prob)]
        for ind, _ in enumerate(data):
            if ind == 1: continue

            if drop_out_flags[ind - 1] is not True:
                d_flag = 0 < np.random.rand() < self.drop_prob
                drop_out_flags.append(d_flag)
            else:
                drop_out_flags.append(False)
        return dict(alpha=self.alphas[flag], img_shape=data[0].shape, gt=flag,
                    prob_flag=prob_flag, drop_out_flags=drop_out_flags)

    def _apply_image(self,
                     data: List[np.ndarray],
                     transform_param: dict):
        return self.calculate_mask(data, transform_param)

    def calculate_mask(self, data: List[np.ndarray], transform_param: dict):
        if not transform_param['prob_flag']:
            return data
        beta_img = data[0]
        beta_img = cv2.cvtColor(beta_img, cv2.COLOR_BGR2GRAY)
        masks = []
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.dilate_kernel_size)
        for ind, img_ in enumerate(data):
            if transform_param["drop_out_flags"][ind]:
                mask = (np.ones(img_.shape) * 255).astype(np.uint8)
            else:
                img = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)

                diff = cv2.absdiff(img, beta_img)
                diff[diff <= self.threshold] = 0
                diff[diff > self.threshold] = 255
                mask = np.dstack([diff] * 3)

                beta_img = np.array(self.momentum * img + (1-self.momentum) * beta_img, dtype=np.uint8)

                mask = cv2.dilate(mask, kernel)
            masks.append(mask)

            """"
            alpha = 0.7
            blended_image = ((img_ * (mask/255).astype(np.uint8)) * alpha).astype(np.uint8) + ((img_ * ((255-mask)/255).astype(np.uint8))) #* (1-alpha)).astype(np.uint8)

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
            cv2.waitKey(int(1000/1))
            """

        masks[0] = masks[1] # copy mask2 to mask1 to prevent all zero mask
        masks = np.array(masks)
        alpha = transform_param['alpha']
        data = (data * masks) * alpha + data * (1-masks)

        return data

    def apply_boxes(self,
                    boxes: np.ndarray,
                    transform_param: dict):
        raise NotImplementedError

    def apply_flow(self,
                   flows: List[np.ndarray],
                   transform_param: dict):
        raise NotImplementedError


@TRANSFORMS.register_module()
class RandomContrast(BaseTransform):

    def __init__(self,
                 prob: float,
                 delta: float):
        self.contrast_prob = prob
        self.contrast_delta = delta

    def get_transform_param(self, *args, **kwargs) -> dict:
        flag = (0 < np.random.rand() < self.contrast_prob)
        delta = np.exp(np.random.uniform(-self.contrast_delta, self.contrast_delta))
        return dict(flag=flag, delta=delta)

    def _apply_image(self,
                     data: List[np.ndarray],
                     transform_param: dict):
        delta = transform_param['delta']
        if transform_param['flag']:
            data = [np.clip(img * delta, 0, 255) for img in data]
        return data


@TRANSFORMS.register_module()
class RandomHueSaturation(BaseTransform):

    def __init__(self,
                 prob: float,
                 hue_delta: float,
                 saturation_delta: float):
        self.prob = prob
        self.hue_delta = hue_delta
        self.saturation_delta = saturation_delta

    def get_transform_param(self, *args, **kwargs) -> dict:
        flag = (0 < np.random.rand() < self.prob)
        hue_delta = np.random.uniform(-self.hue_delta, self.hue_delta)
        saturation_delta = np.exp(np.random.uniform(-self.saturation_delta, self.saturation_delta))
        return dict(flag=flag,
                    hue_delta=hue_delta,
                    saturation_delta=saturation_delta)

    def _apply_image(self,
                     data: List[np.ndarray],
                     transform_param: dict):
        if transform_param['flag']:
            hue_delta = transform_param['hue_delta']
            saturation_delta = transform_param['saturation_delta']
            # convert to HSV color space
            data = [cv2.cvtColor(self.cvt_uint8(img), cv2.COLOR_BGR2HSV).astype(np.float32)
                    for img in data]
            for i in range(len(data)):
                data[i][:, :, 0] += hue_delta
                data[i][:, :, 1] *= saturation_delta
            data = [cv2.cvtColor(self.cvt_uint8(img, is_bgr=False), cv2.COLOR_HSV2BGR).astype(np.float32)
                    for img in data]
        return data

    @staticmethod
    def cvt_uint8(img, is_bgr=True):
        """ convert data type from numpy.float32 to numpy.uint8 """
        nimg = np.round(np.clip(img, 0, 255)).astype(np.uint8)
        if not is_bgr:
            nimg[:, :, 0] = np.clip(nimg[:, :, 0], 0, 179)
        return nimg


@TRANSFORMS.register_module()
class RandomSolarize(BaseTransform):
    def __init__(self,
                 prob: float,
                 threshold: float,
                 normalized: bool):
        self.solarize_prob = prob
        self.threshold = threshold
        self.max_value = 1 if normalized else 255

    def get_transform_param(self, *args, **kwargs) -> dict:
        flag = (0 < np.random.rand() < self.solarize_prob)
        return dict(flag=flag)

    def _apply_image(self,
                     data: List[np.ndarray],
                     transform_param: dict):
        if transform_param['flag']:
            data = [ img[img < self.threshold] for img in data]
        return data


@TRANSFORMS.register_module()
class RandomGaussianBlur(BaseTransform):
    def __init__(self,
                 prob: float,
                 sigma: list):
        self.blur_prob = prob
        self.blur_sigma = sigma

    def get_transform_param(self, *args, **kwargs) -> dict:
        flag = (0 < np.random.rand() < self.blur_prob)
        sigma = np.random.uniform(self.brightness_delta[0], self.blur_sigma[1])
        return dict(flag=flag, sigma=sigma)

    def _apply_image(self,
                     data: List[np.ndarray],
                     transform_param: dict):
        if transform_param['flag']:
            sigma = transform_param['sigma']
            ksize = int(data[0].shape[0] / 20) # as in SimCLR
            data = [cv2.GaussianBlur(img, (ksize, ksize), sigma, cv2.BORDER_DEFAULT) for img in data]
        return data


@TRANSFORMS.register_module()
class RandomGreyScale(BaseTransform):
    def __init__(self,
                 prob: float):
        self.greyscale_prob = prob

    def get_transform_param(self, *args, **kwargs) -> dict:
        flag = (0 < np.random.rand() < self.greyscale_prob)
        return dict(flag=flag)

    def _apply_image(self,
                     data: List[np.ndarray],
                     transform_param: dict):
        if transform_param['flag']:
            data = [np.dstack([cv2.cvtColor(RandomHueSaturation.cvt_uint8(img), cv2.COLOR_BGR2GRAY)] * 3) for img in data]
        return data


@TRANSFORMS.register_module()
class DynamicContrast(BaseTransform):

    def __init__(self,
                 prob: float,
                 delta: float,
                 num_key_frame_probs: List[float]):
        self.prob = prob
        self.delta = delta
        self.num_key_frame_probs = num_key_frame_probs

    def get_transform_param(self, data):
        flag = np.random.rand() < self.prob
        if not flag:
            return dict(flag=flag)
        key_frame_inds = sample_key_frames(len(data), self.num_key_frame_probs)
        num_key_frames = len(key_frame_inds)
        key_frame_deltas = np.exp(np.random.uniform(-self.delta, self.delta, size=(num_key_frames, )))
        deltas = extend_key_frame_to_all(key_frame_deltas, key_frame_inds, 'random')
        return dict(flag=flag, deltas=deltas)

    def _apply_image(self,
                     data: List[np.ndarray],
                     transform_param: dict):
        if transform_param['flag']:
            deltas = transform_param['deltas']
            data = [np.clip(img * deltas[i], a_min=0, a_max=255) for i, img in enumerate(data)]
        return data


@TRANSFORMS.register_module()
class DynamicBrightness(BaseTransform):

    def __init__(self,
                 prob: float,
                 delta: float,
                 num_key_frame_probs: List[float]):
        self.prob = prob
        self.delta = delta
        self.num_key_frame_probs = num_key_frame_probs

    def get_transform_param(self, data):
        flag = np.random.rand() < self.prob
        if not flag:
            return dict(flag=flag)
        key_frame_inds = sample_key_frames(len(data), self.num_key_frame_probs)
        num_key_frames = len(key_frame_inds)
        key_frame_deltas = np.random.uniform(-self.delta, self.delta, size=(num_key_frames, ))
        deltas = extend_key_frame_to_all(key_frame_deltas, key_frame_inds, 'random')
        return dict(flag=flag, deltas=deltas)

    def _apply_image(self,
                     data: List[np.ndarray],
                     transform_param: dict):
        if transform_param['flag']:
            deltas = transform_param['deltas']
            data = [np.clip(img + deltas[i], a_min=0, a_max=255) for i, img in enumerate(data)]
        return data
