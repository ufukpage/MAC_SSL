"""
Self-Supervised Spatiotemporal Feature Learning via Video Rotation Prediction
Longlong Jing, Xiaodong Yang, Jingen Liu, Yingli Tian
"""
from .mac2nreg_model import CoT2nRegMoco
from .mac2nreg_transforms import GroupMask2nRegCalculation
from .mac2nreg_dataset import CoT2nRegMocoDataset

__all__ = ['CoT2nRegMoco', 'CoT2nRegMocoDataset', 'GroupMask2nRegCalculation']
