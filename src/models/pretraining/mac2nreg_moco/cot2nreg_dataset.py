import logging

from ....builder import DATASETS
from ..mac2reg_moco.mac2reg_moco_dataset import CoT2RegMocoDataset


@DATASETS.register_module()
class CoT2nRegMocoDataset(CoT2RegMocoDataset):

    def __init__(self, *args, **kwargs):
        try:
            super(CoT2nRegMocoDataset, self).__init__(*args, **kwargs)
        except ValueError:

            try:
                self.mix_alpha_trans_index = \
                    next(i for i, trans in enumerate(self.img_transform.transforms)
                         if trans.__class__.__name__ == 'GroupMask2nRegCalculation')

                logger = logging.getLogger()
                logger.info("Found 'GroupMask2nRegCalculation' in "
                             "CoT2nRegMocoDataset.")
            except Exception:
                logger = logging.getLogger()
                logger.error("Cannot find 'GroupMask2nRegCalculation' in "
                             "the image transformation configuration."
                             "It is necessary for CoT2nRegMocoDataset task.")
                raise ValueError


