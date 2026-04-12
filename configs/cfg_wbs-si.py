_base_ = './base_config.py'

# model settings
model = dict(
    classname_path='./configs/cls_wbs-si.txt',
    prob_thd=0.3,
    confidence_threshold=0.3,
)

# dataset settings
dataset_type = 'WaterDataset'
data_root = ''

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='configs/wbs-si_val.txt',
        data_prefix=dict(
            img_path='data/water-body-segmentation-in-satellite-images/WaterBodiesDatasetPreprocessed/WaterBodiesDatasetPreprocessed/Images',
            seg_map_path='data/water-body-segmentation-in-satellite-images/WaterBodiesDatasetPreprocessed/WaterBodiesDatasetPreprocessed/Masks_binary'),
        pipeline=test_pipeline))