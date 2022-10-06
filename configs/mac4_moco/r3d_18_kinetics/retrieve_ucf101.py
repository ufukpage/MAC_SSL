_base_ = '../r3d_18_kinetics/finetune_ucf101.py'

work_dir = './output/mac4_moco/r2plus1d_18_kinetics/finetune_kinetics/'

model = dict(
    backbone=dict(
        type='R2Plus1D',
        pretrained='./output/mac4_moco/r3d_18_kinetics/pretraining/epoch_300.pth',
    ),
    st_module=dict(
        spatial_type='avg',
        temporal_size=8,  # 16//8
        spatial_size=7)
)

data = dict(
    videos_per_gpu=1,  # total batch size is 8Gpus*4 == 32
    workers_per_gpu=4,
    train=dict(
        frame_sampler=dict(
            type='UniformFrameSampler',
            num_clips=10,
            clip_len=64,
            strides=2,
            temporal_jitter=False
        ),
        transform_cfg=[
            dict(type='GroupScale', scales=[(171, 128)]),
            dict(type='GroupCenterCrop', out_size=112),
            dict(
                type='GroupToTensor',
                switch_rgb_channels=True,
                div255=True,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )
        ]
    ),
    val=dict(
        frame_sampler=dict(
            type='UniformFrameSampler',
            num_clips=10,
            clip_len=64,
            strides=2,
            temporal_jitter=False
        ),
    ),
    test=dict(
        frame_sampler=dict(
            type='UniformFrameSampler',
            num_clips=10,
            clip_len=64,
            strides=2,
            temporal_jitter=False
        )
    ),
)
