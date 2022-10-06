_base_ = ['../../recognizers/_base_/model_r3d18.py', '../../recognizers/_base_/eval_runtime_ucf101.py']

work_dir = './output/mac2_moco/r2plus1d_18_ucf101/eval_ucf101/'

model = dict(
    backbone=dict(
        type='R2Plus1D',
        pretrained='./output/mac2_moco/r2plus1d_18_ucf101/finetune_ucf101/epoch_150.pth',
    ),
)
