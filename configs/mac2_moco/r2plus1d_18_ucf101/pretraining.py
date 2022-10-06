_base_ = '../r3d_18_ucf101/pretraining.py'

work_dir = './output/mac2_moco/r2plus1d_18_ucf101/pretraining/'

model = dict(
    backbone=dict(
        type='R2Plus1D',
    ),
    nce_loss_weight=.0,
    cls_loss_weight=1.0,
    symmetric_loss=True
)


