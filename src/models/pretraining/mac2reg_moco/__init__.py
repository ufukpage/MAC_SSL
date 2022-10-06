"""
Self-Supervised Spatiotemporal Feature Learning via Video Rotation Prediction
Longlong Jing, Xiaodong Yang, Jingen Liu, Yingli Tian
"""
from .mac2reg_moco_dataset import CoT2RegMocoDataset
from .mac2reg_moco_model import CoT2RegMoco

__all__ = ['CoT2RegMoco', 'CoT2RegMocoDataset']
