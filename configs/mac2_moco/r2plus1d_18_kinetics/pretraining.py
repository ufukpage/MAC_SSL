_base_ = '../r3d_18_kinetics/pretraining.py'

work_dir = './output/mac2_moco/r2plus1d_18_kinetics/pretraining/'

model = dict(
    backbone=dict(
        type='R2Plus1D',
    ),
    nce_loss_weight=.5,
    cls_loss_weight=.5,
    symmetric_loss=True
)


