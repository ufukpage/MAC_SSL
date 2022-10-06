"""
Self-Supervised Spatiotemporal Feature Learning via Video Rotation Prediction
Longlong Jing, Xiaodong Yang, Jingen Liu, Yingli Tian
"""
from .mac_moco_dataset import CoT2MocoDataset, CoT2MocoMiniK400BBoxDataset
from .mac_moco_model import CoT2Moco, CoTMocoSeparate
from .mac_moco_transforms import GroupMask2Calculation, GroupMask2NCalculation

__all__ = ['CoT2Moco', 'CoT2MocoDataset', 'CoTMocoSeparate', 'CoT2MocoMiniK400BBoxDataset', 'GroupMask2Calculation',
           'GroupMask2NCalculation']
