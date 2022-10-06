import os
import json
import pickle
import numpy as np


class BBoxDataSource(object):

    def __init__(self, ann_file: str, bbox_file:str, data_dir: str = None):
        """ The video name & class label are stored in a json file. """
        self.data_dir = data_dir
        if data_dir is not None:
            ann_file = os.path.join(data_dir, ann_file)
        self.ann_file = ann_file
        assert self.ann_file.endswith('.json'), f'Support .json file only, but got {ann_file}'
        assert os.path.isfile(self.ann_file), f'Cannot find file {ann_file}'

        with open(self.ann_file, 'r') as f:
            self.video_info_list = json.load(f)

        with open(bbox_file, 'rb') as f:
            self.bbox_data = pickle.load(f)

    def __len__(self):
        return len(self.video_info_list)

    def __getitem__(self, idx):
        return self.video_info_list[idx]


class BBoxFolderDataSource(object):

    def __init__(self, ann_file: str, bbox_folder:str, data_dir: str = None):
        """ The video name & class label are stored in a json file. """
        self.data_dir = data_dir
        if data_dir is not None:
            ann_file = os.path.join(data_dir, ann_file)
        self.ann_file = ann_file
        assert self.ann_file.endswith('.json'), f'Support .json file only, but got {ann_file}'
        assert os.path.isfile(self.ann_file), f'Cannot find file {ann_file}'
        assert os.path.isdir(bbox_folder), f'Cannot find bbox folder {bbox_folder}'

        with open(self.ann_file, 'r') as f:
            self.video_info_list = json.load(f)

        self.bbox_data = {}
        path = os.walk(bbox_folder)
        for root, directories, files in path:
            for file in files:
                vid_name = file.split('.')[0]
                class_name = vid_name.split('_')[1]
                key = class_name + '/' + vid_name
                with open(os.path.join(root, file)) as f:
                    lines = f.readlines()
                annot_ind = 0
                annots = []
                has_started = False
                for ind, line in enumerate(lines):
                    parts = line.split()
                    if len(parts) > 1:
                        if not has_started:
                            annots.append({})

                            annots[annot_ind]['boxes'] = []
                            annots[annot_ind]['sf'] = ind
                            has_started = True
                        int_arr = [round(float(i)) for i in parts[1:5]]
                        int_arr[2] -= int_arr[0]
                        int_arr[3] -= int_arr[1]
                        annots[annot_ind]['boxes'].append(np.array(int_arr).astype(np.uint32))
                    elif has_started:
                        annots[annot_ind]['boxes'] = np.array(annots[annot_ind]['boxes'])
                        annots[annot_ind]['ef'] = ind - 1
                        has_started = False
                        annot_ind = annot_ind + 1
                if len(annots) > 0:
                    self.bbox_data[key] = {'annotations': annots}

    def __len__(self):
        return len(self.video_info_list)

    def __getitem__(self, idx):
        return self.video_info_list[idx]


class OnlyBBoxDataSource(object):

    def __init__(self, ann_file: str, bbox_file:str, data_dir: str = None):
        """ The video name & class label are stored in a json file. """
        self.data_dir = data_dir
        if data_dir is not None:
            ann_file = os.path.join(data_dir, ann_file)
        self.ann_file = ann_file
        assert self.ann_file.endswith('.json'), f'Support .json file only, but got {ann_file}'
        assert os.path.isfile(self.ann_file), f'Cannot find file {ann_file}'

        with open(bbox_file, 'rb') as f:
            self.bbox_data = pickle.load(f)

        self.bbox_list_data = [(k,v) for k,v in self.bbox_data.items()]

    def __len__(self):
        return len(self.bbox_list_data)

    def __getitem__(self, idx):
        return self.bbox_list_data[idx]

