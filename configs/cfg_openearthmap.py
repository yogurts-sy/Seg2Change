_base_ = './base_config.py'

# model settings
model = dict(
    classname_path='./configs/cls_openearthmap.txt',
    prob_thd=0.1,
    confidence_threshold=0.1,
    slide_stride=512,
    slide_crop=512,
)

# dataset settings
dataset_type = 'OpenEarthMapDataset'
data_root = 'data/OpenEarthMap'

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
            img_path='val/images',
            seg_map_path='val/labels'),
        pipeline=test_pipeline))