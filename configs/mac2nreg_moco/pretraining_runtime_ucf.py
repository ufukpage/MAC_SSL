dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
syncbn = True
ssl_mode = True

evaluation = dict(interval=10)

data = dict(
    videos_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='CoT2nRegMocoDataset',
        data_source=dict(
            type='JsonClsDataSource',
            ann_file= "D:/UCF-101/split/train_split_1.json"
        ),
        backend=dict(
            type='JpegBackend',
            frame_fmt='frame{:06d}.jpg'
        ),
        frame_sampler=dict(
            type='RandomFrameSampler',
            num_clips=1,
            clip_len=16,
            strides= [1, 2, 3, 4],
            temporal_jitter=False
        ),
        transform_cfg=[
            dict(type='GroupScale', scales=[(149, 112), (171, 128), (192, 144)]),
            dict(type='GroupFlip', flip_prob=0.5),
            dict(type='GroupRandomCrop', out_size=112),
            dict(type='GroupMask2nRegCalculation', alphas=[0.165, 0.495, 0.83, 1.], threshold=30, momentum=0.5,
                 dist_type='normal', dilate_kernel_size=(2, 2), drop_prob=0),
            dict(type='RandomBrightness', prob=0.80, delta=32),
            dict(type='RandomContrast', prob=0.80, delta=0.20),
            dict(type='RandomHueSaturation', prob=0.80, hue_delta=12, saturation_delta=0.1),
            dict(type='RandomGreyScale', prob=0.20),
            dict(
                type='GroupToTensor',
                switch_rgb_channels=True,
                div255=True,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )
        ],
        test_mode=False
    ),
    val=dict(
        type='CoT2nRegMocoDataset',
        data_source=dict(
            type='JsonClsDataSource',
            ann_file= "D:/UCF-101/split/test_split_1.json"
        ),
        backend=dict(
            type='JpegBackend',
            frame_fmt='frame{:06d}.jpg'
        ),
        frame_sampler=dict(
            type='RandomFrameSampler',
            num_clips=1,
            clip_len=16,
            strides=[1, 2, 3, 4],
            temporal_jitter=False
        ),
        transform_cfg=[
            dict(type='GroupScale', scales=[(149, 112), (171, 128), (192, 144)]),
            dict(type='GroupFlip', flip_prob=0.5),
            dict(type='GroupRandomCrop', out_size=112),
            dict(type='GroupMask2nRegCalculation', alphas=[0.165, 0.495, 0.83, 1.], threshold=30, momentum=0.5,
                 dilate_kernel_size=(2, 2), drop_prob=0, dist_type='normal'),
            dict(
                type='GroupToTensor',
                switch_rgb_channels=True,
                div255=True,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )
        ],
        test_mode=False
    ),

)

# optimizer
total_epochs = 300
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    step=[100, 200]
)
checkpoint_config = dict(interval=10, max_keep_ckpts=10, create_symlink=False)
workflow = [('train', 1)]
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ]
)
