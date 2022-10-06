import os
import cv2
import numpy as np
from typing import List
import time


class SSLK400Item(object):

    def __init__(self, video_info: dict, type_name: str, data_dir: str):
        self.data_dir = os.path.join(data_dir, type_name, video_info['name'])
        self.cap_id = None

    def __len__(self):
        if self.cap_id is None or not self.cap_id.isOpened():
            self._check_available(self.data_dir)
            self.cap_id = cv2.VideoCapture(self.data_dir)
        length = int(self.cap_id.get(cv2.CAP_PROP_FRAME_COUNT))
        self.cap_id.release()
        return length

    def close(self):
        if self.cap_id.isOpened() or self.cap_id is not None:
            self.cap_id.release()

    def get_frame(self, indices: List[int]) -> List[np.ndarray]:
        if isinstance(indices, int):
            indices = [indices]
        img_list = []
        if self.cap_id is None or not self.cap_id.isOpened():
            self._check_available(self.data_dir)
            self.cap_id = cv2.VideoCapture(self.data_dir)
        """"
        start_time = time.time()
        for idx in indices:
            img = self.load_image_jpeg(self.cap_id, idx)
            img_list.append(img)
        print("Inside Time: " + str(time.time() - start_time))
        #"""
        # start_time = time.time()
        start, end = indices[0], indices[-1]
        self.cap_id.set(cv2.CAP_PROP_POS_FRAMES, start)
        norm_indices = indices - start
        for idx, _ in enumerate(range(start, end+1)):
            ret, img = self.cap_id.read()
            if not ret:
                break #raise Exception("Frame could not read from video ")
            if idx in norm_indices:
                img_list.append(img)
                np.delete(norm_indices, 0)
        # print("Outside Time: " + str(time.time() - start_time))

        while len(img_list) != len(indices):
            img_list.append(img_list[-1])
            # raise Exception("len(img_list):" + str(len(img_list)) + " len(indices):" + str(len(indices)) +
            #                " start:" + str(start) + " end:" + str(end) + ' indices:', indices)
        return img_list

    @staticmethod
    def load_image_jpeg(cap, index, flag=cv2.IMREAD_COLOR):
        # set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, img = cap.read()
        if not ret:
            raise Exception("Frame could not read from video ")
        return img

    @staticmethod
    def _check_available(data_dir):
        if data_dir is None:
            raise ValueError("There is not file path defined in video annotations")
        if not os.path.isfile(data_dir):
            raise FileNotFoundError("Cannot find video file: {}".format(data_dir))


class SSLK400Backend(object):

    def __init__(self, frame_fmt: str = 'img_{:05d}.jpg', type_name ='train', data_dir: str = None):
        self.data_dir = data_dir
        self.type_name = type_name

    def open(self, video_info) -> SSLK400Item:
        return SSLK400Item(video_info, self.type_name, self.data_dir)
