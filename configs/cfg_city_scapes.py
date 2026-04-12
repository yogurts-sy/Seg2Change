_base_ = './base_config.py'

# model settings
model = dict(
    classname_path='./configs/cls_city_scapes.txt',
    confidence_threshold=0.3,
)

# dataset settings
dataset_type = 'CityscapesDataset'
data_root = './data/CityScapes'

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
            img_path='leftImg8bit/val', seg_map_path='gtFine/val'),
        pipeline=test_pipeline))