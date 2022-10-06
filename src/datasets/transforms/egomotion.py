import cv2
import numpy as np
import random
from typing import Union, Tuple, List, Iterable

from .base_transform import BaseTransform
from ...builder import TRANSFORMS
from .dynamic_utils import (extend_key_frame_to_all, sample_key_frames)


@TRANSFORMS.register_module()
class EgoMotion(BaseTransform):
    def __init__(self, move_lims: list, move_probs: list, key_frame_probs: list, dir_flag=0.5):
        """ Synthetic EgoMotion transformation.
        Args:
            move_lims (list): first two movement limit in pixels, last value is scale factor
            to simulate motion in z direction
            move_probs (list): x y z direction movement probabilities
            dir_flag (float): whether movement direction will be changed or not
            key_frame_probs (list): probabilities of sampling how many key
                frames. The sum of this list should be 1.
        """
        self.move_lims = move_lims
        self.move_probs = move_probs
        self.dir_flag = dir_flag
        self.key_frame_probs = key_frame_probs

    def get_transform_param(self, data: List[np.ndarray], *args, **kwargs):
        flags = []
        for probs in self.move_probs:
            prob = (0 < np.random.rand() < probs)
            flags.append(prob)
        return dict(flags=flags)

    def _apply_image(self,
                     data: List[np.ndarray],
                     transform_param: dict):

        key_inds = sample_key_frames(len(data), self.key_frame_probs)
        moves = []
        for move_lim in self.move_lims:
            move = np.random.uniform(0, move_lim, size=(len(key_inds), 1))
            move = extend_key_frame_to_all(move, key_inds)
            moves.append(move)

        for img_ind, img in enumerate(data):
            height, width = img.shape[:2]
            flags = []
            for probs in self.move_probs:
                prob = (0 < np.random.rand() < probs)
                flags.append(prob)
            motion_x, motion_y, motion_z = 0, 0, 1
            direction = (0 < np.random.rand() < self.dir_flag)
            for ind, flag in enumerate(flags):
                if ind == 0: # x move
                    if flag:
                        motion_x = moves[0][img_ind] if direction else -moves[0][img_ind]
                        motion_x = int(motion_x)
                elif ind == 1:  # y move
                    if flag:
                        motion_y = moves[1][img_ind] if direction else -moves[1][img_ind]
                        motion_y = int(motion_y)
                elif ind == 2:  # z move
                    if flag:
                        motion_z = moves[2][img_ind] if direction else 1 / moves[2][img_ind] + np.finfo(np.float32).eps
                        if not direction: # if direction changed
                            motion_z = motion_z if motion_z > 1/self.move_lims[2] else 1/self.move_lims[2]
                        motion_z = float(motion_z)

            T = np.float32([[1, 0, motion_x], [0, 1, motion_y]])
            img_translated = cv2.warpAffine(img, T, (width, height))
            img_resized = cv2.resize(img_translated, (width, height), fx=motion_z, fy=motion_z,
                                     interpolation=cv2.INTER_AREA)
            """"
            cv2.imshow('FD', img_resized)
            cv2.waitKey(int(1000 / 1))
            # """
            data[img_ind] = img_resized
        return data

    def apply_boxes(self,
                    boxes: np.ndarray,
                    transform_param: dict):
        raise NotImplementedError

    def apply_flow(self,
                   flows: List[np.ndarray],
                   transform_param: dict):
        raise NotImplementedError
