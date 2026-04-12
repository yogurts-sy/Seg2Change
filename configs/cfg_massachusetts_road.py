_base_ = './base_config.py'

# model settings
model = dict(
    classname_path='./configs/cls_roadval.txt',
    prob_thd=0.3,
    confidence_threshold=0.5,
)

# dataset settings
dataset_type = 'RoadValDataset'
data_root = 'data/GlobalRoadSet_Val/Massachusetts_test_49'

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
            img_path='img',
            seg_map_path='label_cvt'),
        pipeline=test_pipeline))