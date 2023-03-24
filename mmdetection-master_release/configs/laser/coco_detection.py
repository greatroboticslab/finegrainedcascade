# dataset settings
dataset_type = 'CocoDataset'
classes = ('day1 red laser', 'day2 red laser', 'day3 red laser', 'red laser day 4', 'red laser day 5')
#('day1 green laser', 'day2 green laser', 'day3 green laser', 'day4 green laser', 'day5 green laser')
data_root = '/data/laser_detection/red/labelme2coco/'#'/data/laser_detection/green/labelme2coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type, classes=classes,
        ann_file=data_root + './train.json',
        img_prefix=data_root + './',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type, classes=classes,
        ann_file=data_root + './val.json',
        img_prefix=data_root + './',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type, classes=classes,
        ann_file=data_root + './val.json',
        img_prefix=data_root + './',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric=['bbox', 'proposal'],
                  iou_thrs=[0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4,
                            0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75])
