import os
import cv2
import numpy as np
from typing import List


class JpegBBItem(object):

    def __init__(self, video_info: dict, data_dir: str, frame_fmt: str):
        self.video_name = video_info['name'].split('.')[0]
        self.data_dir = os.path.join(data_dir, self.video_name.split("/")[1].split('.')[0])
        self.frame_fmt = frame_fmt
        self.img_frame_id = None

    def __len__(self):
        if self.img_frame_id is None:
            self._check_available(self.data_dir)
            self.img_frame_id = 1

        namelist = next(os.walk(self.data_dir), (None, None, []))[2]
        namelist = [name for name in namelist if name.endswith('.jpg')]
        return len(namelist)

    def close(self):
        self.img_frame_id = None

    def get_frame(self, indices: List[int]) -> List[np.ndarray]:
        if isinstance(indices, int):
            indices = [indices]
        img_list = []
        if self.img_frame_id is None:
            self._check_available(self.data_dir)
            self.img_frame_id = 1

        for idx in indices:
            file_name = self.frame_fmt.format(int(idx) + 1)
            img = self.load_image_jpeg(self.data_dir, file_name, cv2.IMREAD_COLOR)
            img_list.append(img)
        return img_list

    def get_bbox(self, bbox_data: dict, indices: List[int]) -> List[np.ndarray]:
        if isinstance(indices, int):
            indices = [indices]
        if self.img_frame_id is None:
            self._check_available(self.data_dir)
            self.img_frame_id = 1

        if self.video_name in bbox_data:
            annots = bbox_data[self.video_name]['annotations']
            for ann in annots:
                if indices.max() <= ann['ef'] and indices.min() >= ann['sf']:
                    return ann['boxes'][indices-ann['sf']]

        return list()

    @staticmethod
    def load_image_jpeg(data_dir, file_name, flag=cv2.IMREAD_COLOR):
        img_path = os.path.join(data_dir, file_name)
        img = cv2.imread(img_path, flag)
        return img

    @staticmethod
    def _check_available(data_dir):
        if data_dir is None:
            raise ValueError("There is not file path defined in video annotations")
        if not os.path.exists(data_dir):
            raise FileNotFoundError("Cannot find image folder: {}".format(data_dir))


class JpegBBBackend(object):

    def __init__(self,
                 frame_fmt: str = 'img_{:05d}.jpg',
                 data_dir: str = None):
        self.data_dir = data_dir
        self.frame_fmt = frame_fmt

    def open(self, video_info) -> JpegBBItem:
        return JpegBBItem(video_info, self.data_dir, self.frame_fmt)


class OnlyJpegBBItem(JpegBBItem):

    def __init__(self, video_info: dict, data_dir: str, frame_fmt: str):
        self.video_name = video_info[0]
        self.data_dir = os.path.join(data_dir, self.video_name.split("/")[1])
        self.frame_fmt = frame_fmt
        self.img_frame_id = None
        self.ef = video_info[1]['annotations'][0]['ef']
        self.sf = video_info[1]['annotations'][0]['sf']

    def __len__(self):
        if self.img_frame_id is None:
            self._check_available(self.data_dir)
            self.img_frame_id = 1

        return self.ef - self.sf

    @staticmethod
    def _check_available(data_dir):
        if data_dir is None:
            raise ValueError("There is not file path defined in video annotations")
        if not os.path.exists(data_dir):
            raise FileNotFoundError("Cannot find image folder: {}".format(data_dir))


class OnlyJpegBBBackend(object):

    def __init__(self,
                 frame_fmt: str = 'img_{:05d}.jpg',
                 data_dir: str = None):
        self.data_dir = data_dir
        self.frame_fmt = frame_fmt

    def open(self, video_info) -> OnlyJpegBBItem:
        return OnlyJpegBBItem(video_info, self.data_dir, self.frame_fmt)

