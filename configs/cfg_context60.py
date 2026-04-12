_base_ = './base_config.py'

# model settings
model = dict(
    classname_path='./configs/cls_context60.txt',
    confidence_threshold=0.3,
    prob_thd=0.1,
)

# dataset settings
dataset_type = 'PascalContext60Dataset'
data_root = './data/VOC2010'

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
            img_path='JPEGImages', seg_map_path='SegmentationClassContext'),
        ann_file='ImageSets/SegmentationContext/val.txt',
        pipeline=test_pipeline))