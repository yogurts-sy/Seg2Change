_base_ = './base_config.py'

# model settings
model = dict(
    classname_path='./configs/cls_iSAID.txt',
    prob_thd=0.5,
    confidence_threshold=0.4,
)

# dataset settings
dataset_type = 'iSAIDDataset'
data_root = 'data/iSAID'

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
        reduce_zero_label=False,
        data_prefix=dict(
            img_path='img_dir/val',
            seg_map_path='ann_dir/val'),
        pipeline=test_pipeline))
