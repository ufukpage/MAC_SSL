_base_ = '../r3d_18_ucf101/pretraining.py'

work_dir = './output/mac2_moco_separate/r2plus1d_18_ucf101/pretraining/'

model = dict(
    type='CoTMocoSeparate',
    backbone=dict(
        type='R2Plus1D',
    ),
    cls_head=dict(
        with_avg_pool=False,
        temporal_feature_size=1,
        spatial_feature_size=1,
        dropout_ratio=0.5,
        in_channels=512,
        init_std=0.001,
        num_classes=4
    ),
    cls_head_2=dict(
        with_avg_pool=False,
        temporal_feature_size=1,
        spatial_feature_size=1,
        dropout_ratio=0.5,
        in_channels=512,
        init_std=0.001,
        num_classes=4
    ),
    nce_loss_weight=.0,
    cls_loss_weight=1.,
    symmetric_loss=False
)


