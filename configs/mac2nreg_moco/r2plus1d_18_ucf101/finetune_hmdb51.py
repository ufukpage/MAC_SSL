_base_ = '../r3d_18_ucf101/finetune_hmdb51.py'

work_dir = './output/mac2nreg_moco/r2plus1d_18_hmdb51/finetune_hmdb51/'

model = dict(
    backbone=dict(
        type='R2Plus1D',
        pretrained='./output/mac2nreg_moco/r2plus1d_18_hmdb51/pretraining/epoch_300.pth',
    ),
    cls_head=dict(
        num_classes=51
    )
)
