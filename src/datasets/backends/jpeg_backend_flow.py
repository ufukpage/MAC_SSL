import os
import cv2
from typing import List
from .jpeg_backend import JpegItem


class JpegFlowItem(JpegItem):

    def __init__(self, video_info: dict, flow_dir:str ,data_dir: str, frame_fmt: str):
        super(JpegFlowItem, self).__init__(video_info=video_info, data_dir=data_dir, frame_fmt=frame_fmt)

        self.flow_dir = flow_dir
        self.vid_name = self.data_dir.split('\\')[-1]

    def get_frame(self, indices: List[int]):
        if isinstance(indices, int):
            indices = [indices]
        img_list = []
        flow_list = []
        if self.img_frame_id is None:
            self._check_available(self.data_dir)
            self.img_frame_id = 1

        for idx in indices:
            file_name = self.frame_fmt.format(int(idx) + 1)
            img = self.load_image_jpeg(self.data_dir, file_name, cv2.IMREAD_COLOR)
            flow_u = self.load_image_flow(os.path.join(self.flow_dir, 'u', self.vid_name), file_name,
                                          cv2.IMREAD_GRAYSCALE)
            flow_v = self.load_image_flow(os.path.join(self.flow_dir, 'v', self.vid_name), file_name,
                                          cv2.IMREAD_GRAYSCALE)
            img_list.append(img)
            flow_list.append([flow_u, flow_v])

        return img_list, flow_list

    @staticmethod
    def load_image_flow(data_dir, file_name, flag=cv2.IMREAD_GRAYSCALE):
        img_path = os.path.join(data_dir, file_name)
        img = cv2.imread(img_path, flag)
        return img


class JpegFlowBackend(object):

    def __init__(self,
                 frame_fmt: str = 'img_{:05d}.jpg',
                 flow_dir: str = None,
                 data_dir: str = None):
        self.data_dir = data_dir
        self.frame_fmt = frame_fmt
        self.flow_dir = flow_dir

    def open(self, video_info) -> JpegFlowItem:
        return JpegFlowItem(video_info, self.flow_dir, self.data_dir, self.frame_fmt)
