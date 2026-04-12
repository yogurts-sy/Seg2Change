_base_ = './base_config.py'

# model settings
model = dict(
    classname_path='./configs/cls_whu.txt',
    prob_thd=0.4,
    confidence_threshold=0.5,
)

# dataset settings
dataset_type = 'WHUDataset'
data_root = 'data/WHU-BD'

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
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
            img_path='val/image',
            seg_map_path='val/label_binary'),
        pipeline=test_pipeline))