import os
from os import listdir
from os.path import isfile, join
import cv2
import json


class SSLK400DataSource(object):

    def __init__(self, ann_file: str = None, data_dir: str = None):
        """ The video name & class label are stored in a json file. """
        self.data_dir = data_dir
        assert os.path.exists(self.data_dir), f'Cannot find folder {data_dir}'
        video_info_list = [dict(name=f) for f in listdir(data_dir) if isfile(join(data_dir, f))
                                and f.endswith('.mp4')]
        video_info_list = sorted(video_info_list, key=lambda d: d['name'])

        if ann_file is None:
            self.video_info_list = []
            error = list()
            for info in video_info_list:
                try:
                    cap_id = cv2.VideoCapture(join(data_dir, info['name']))
                    # if cap_id.isOpened():

                    # length = int(cap_id.get(cv2.CAP_PROP_FRAME_COUNT))

                    if cap_id.isOpened():
                        self.video_info_list.append(info)
                    else:
                        error.append(info['name'])
                    cap_id.release()
                except:
                    error.append(info['name'])
                    cap_id.release()
            print("SSLK400DataSource load ends")
        else:
            self.ann_file = ann_file
            assert self.ann_file.endswith('.json'), f'Support .json file only, but got {ann_file}'
            assert os.path.isfile(self.ann_file), f'Cannot find file {ann_file}'
            with open(self.ann_file, 'r') as f:
                self.video_info_list = json.load(f)

    def __len__(self):
        return len(self.video_info_list)

    def __getitem__(self, idx):
        return self.video_info_list[idx]
