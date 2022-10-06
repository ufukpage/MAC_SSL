from torch.utils.data import Dataset


from ....builder import DATASETS
from ....datasets.transforms import Compose
from ....datasets import builder
from numpy import random


@DATASETS.register_module()
class BeMoCoDataset(Dataset):

    def __init__(self,
                 data_dir: str,
                 data_source: dict,
                 backend: dict,
                 frame_sampler: dict,
                 transform_cfg: list,
                 test_mode: bool = False,
                 gamma: float = 0.3,
                 hard_negative=False):
        """ A dataset class to generate a pair of training examples
        for contrastive learning. Basically, the vanilla MoCo is traine on
        image dataset, like ImageNet-1M. To facilitate its pplication on video
        dataset, we random pick two video clip and discriminate whether
        these two clips are from same video or not.

        Args:
            data_source (dict): data source configuration dictionary
            data_dir (str): data root directory
            transform_cfg (list): data augmentation configuration list
            backend (dict): storage backend configuration
            test_mode (bool): placeholder, not available in MoCo training.
        """
        self.data_dir = data_dir
        self.data_source = builder.build_data_source(data_source, dict(data_dir=data_dir))
        self.backend = builder.build_backend(backend, dict(data_dir=data_dir))
        self.frame_sampler = builder.build_frame_sampler(frame_sampler)
        self.img_transform = Compose(transform_cfg)
        self.test_mode = test_mode
        self.gamma = gamma
        self.hard_negative = hard_negative

    def __len__(self):
        return len(self.data_source)

    def get_single_clip(self, storage_obj, frame_inds):
        """ Get single video clip according to the video_info query."""
        num_segs, clip_len = frame_inds.shape
        assert num_segs == 1
        img_list = storage_obj.get_frame(frame_inds.reshape(-1))
        img_tensor = self.img_transform.apply_image(img_list)
        img_tensor = img_tensor.permute(1, 0, 2, 3).contiguous()
        return img_tensor

    def __getitem__(self, idx):
        video_info = self.data_source[idx]
        storage_obj = self.backend.open(video_info)

        frame_inds = self.frame_sampler.sample(len(storage_obj))
        imgs_q = self.get_single_clip(storage_obj, frame_inds)
        imgs_k = self.get_single_clip(storage_obj, frame_inds)

        loss_prob = random.random() * self.gamma
        c, t, h, w = imgs_q.size()
        static_index = random.randint(t-1)
        static_vid = imgs_q[:, static_index, :, :]
        static_vid = static_vid.unsqueeze(1).repeat(1, t, 1, 1)
        imgs_q = imgs_q * (1 - loss_prob) + loss_prob * static_vid

        data = dict(
            imgs_q=imgs_q,
            imgs_k=imgs_k,
            imgs=imgs_q
        )

        # different clip, same video; for sth v1, it uses different video
        #  different clips of the same video contain different motion patterns but similar background.
        if self.hard_negative:
            thresh = 2
            frame_inds_n = self.frame_sampler.sample(len(storage_obj))
            while abs(frame_inds_n[0][0] - frame_inds[0][0]) < thresh:
                frame_inds_n = self.frame_sampler.sample(len(storage_obj))

            imgs_n = self.get_single_clip(storage_obj, frame_inds_n)
            data['imgs_n'] = imgs_n

        storage_obj.close()

        return data
