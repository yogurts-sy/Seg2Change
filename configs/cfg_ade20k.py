_base_ = './base_config.py'

# model settings
model = dict(
    classname_path='./configs/cls_ade20k.txt',
    confidence_threshold=0.3,
)

# dataset settings
dataset_type = 'ADE20KDataset'
data_root = './data/ADE20K'

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='PackSegInputs')
]

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/validation',
            seg_map_path='annotations/validation'),
        pipeline=test_pipeline))