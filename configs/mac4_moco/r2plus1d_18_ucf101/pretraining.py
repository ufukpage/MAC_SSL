_base_ = '../r3d_18_ucf101/pretraining.py'

work_dir = './output/mac4_moco/r2plus1d_18_ucf101/pretraining/'

model = dict(
    backbone=dict(
        type='R2Plus1D',
    ),
    nce_loss_weight=.5,
    cls_loss_weight=1.,
    symmetric_loss=False
)


