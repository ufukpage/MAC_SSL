_base_ = './pretraining_runtime_ucf.py'

data = dict(
    train=dict(
        type='CoT2MocoMiniK400BBoxDataset',
        data_source=dict(
            type='SSLK400DataSource',
            ann_file='/Downloads/K-400/K-400/demo_val_data_mini_k400.json',
            data_dir='/Downloads/K-400/K-400/val'
        ),
        backend=dict(
            type='MiniK400Backend',
            type_name='val',
            bbox_dir='D:/CMP784Projects/CtPW/val_mini_k400_bbox/'
        ),
    )
)

# optimizer
total_epochs = 90
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    step=[30, 60]
)
checkpoint_config = dict(interval=50, max_keep_ckpts=1, create_symlink=False)
workflow = [('train', 1)]
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ]
)
