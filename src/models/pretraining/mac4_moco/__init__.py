"""
Self-Supervised Spatiotemporal Feature Learning via Video Rotation Prediction
Longlong Jing, Xiaodong Yang, Jingen Liu, Yingli Tian
"""
from .mac4_moco_dataset import CoT4MocoDataset
from .mac4_transforms import GroupMask4Calculation

__all__ = ['CoT4MocoDataset', 'GroupMask4Calculation']
