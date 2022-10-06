import os
import cv2
import numpy as np
from typing import List


class MiniK400Item(object):

    def __init__(self, video_info: dict, type_name: str, data_dir: str, bbox_dir: str):
        self.data_dir = os.path.join(data_dir, type_name, video_info['name'])
        self.cap_id = None
        self.vid_name = video_info['name']
        self.bbox_dir = bbox_dir
        self.length = video_info['boxes']

    def __len__(self):
        if self.cap_id is None or not self.cap_id.isOpened():
            self._check_available(self.data_dir)
            self.cap_id = cv2.VideoCapture(self.data_dir)
        length = int(self.cap_id.get(cv2.CAP_PROP_FRAME_COUNT))
        self.cap_id.release()
        if length <= 0:
            raise Exception("Frame length zero exception: ", self.data_dir, 'CAP ID:', self.cap_id)
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
            if len(img_list) == 0:
                length = int( self.cap_id.get(cv2.CAP_PROP_FRAME_COUNT))
                raise Exception('length:' + str(length) + " len(indices):" + str(len(indices)) +
                                " start:" + str(start) + " end:" + str(end) + ' indices:', indices)
            img_list.append(img_list[-1])
        return img_list

    def get_bbox(self):
        return np.load(os.path.join(self.bbox_dir, self.vid_name.split('.')[0] + '.npy'), allow_pickle=True).item()

    @staticmethod
    def _check_available(data_dir):
        if data_dir is None:
            raise ValueError("There is not file path defined in video annotations")
        if not os.path.isfile(data_dir):
            raise FileNotFoundError("Cannot find video file: {}".format(data_dir))


class MiniK400Backend(object):

    def __init__(self, frame_fmt: str, type_name='train', data_dir: str = None, bbox_dir: str = None):
        self.data_dir = data_dir
        self.type_name = type_name
        self.bbox_dir = bbox_dir

    def open(self, video_info) -> MiniK400Item:
        return MiniK400Item(video_info, self.type_name, self.data_dir, self.bbox_dir)
