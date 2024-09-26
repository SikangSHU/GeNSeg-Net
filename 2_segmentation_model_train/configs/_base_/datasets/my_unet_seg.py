# dataset settings
dataset_type = 'MUSDataset'
data_root = './dataset/nucleus_dataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        data_root=data_root,
        img_dir='img',
        ann_dir='annotation',
        type=dataset_type,
        pipeline=train_pipeline,
        split="dataset_split/train.txt"),
    val=dict(
        data_root=data_root,
        img_dir='img',
        ann_dir='annotation',
        type=dataset_type,
        pipeline=test_pipeline,
        split="dataset_split/val.txt"),
    test=dict(
        data_root=data_root,
        img_dir='img',
        ann_dir='ann_png1/',
        type=dataset_type,
        pipeline=test_pipeline,
        split="dataset_split/val.txt"))

