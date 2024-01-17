_base_ = [
    '../../_base_/default_runtime_on_local.py'
]

# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ViT_Zero_CLIP',
        pretrained="pretrained_models",
        input_resolution=224,
        patch_size=16,
        num_frames=16,
        width=768,
        layers=12,
        heads=12,
        # drop_path_rate=0.2,
        adapter_scale=0.5, # TODO 理论上没影响
        num_tadapter=2,
        stdha_cfg=dict(shift_div=12, divide_head=False)
        ),
    cls_head=dict(
        type='I3DHead',
        in_channels=768,
        num_classes=400,
        spatial_type='avg',
        dropout_ratio=0.5,
        average_clips='prob'),
        data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[122.769, 116.74, 104.04],
        std=[68.493, 66.63, 70.321], # 注意clip和imagenet的不一样
        # blending=dict(
        #     type='RandomBatchAugment',
        #     augments=[
        #         dict(type='MixupBlending', alpha=0.8, num_classes=400),
        #         dict(type='CutmixBlending', alpha=1, num_classes=400)
        #     ]),
        format_shape='NCTHW'))


# dataset settings
dataset_type = 'VideoDataset'
data_prefix = 'lxh:s3://lxhBucket/kinetics_400_320_30fps/'
data_root = 'lxh:s3://lxhBucket/kinetics_400_320_30fps/kinetics_400_320_30fps_train'
data_root_val = 'lxh:s3://lxhBucket/kinetics_400_320_30fps/kinetics_400_320_30fps_val'
ann_file_train = 'data/kinetics400/kinetics400_train_list_videos.txt'
ann_file_val = 'data/kinetics400/kinetics400_val_list_videos.txt'
ann_file_test = 'data/kinetics400/kinetics400_val_list_videos.txt'

file_client_args = dict(io_backend='petrel')

train_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='SampleFrames', clip_len=16, frame_interval=8, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='PytorchVideoWrapper',
        op='RandAugment',
        magnitude=7,
        num_layers=4),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    # dict(type='RandomErasing', erase_prob=0.25, mode='rand'),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=8,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=8,
        num_clips=3,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root_val),
        pipeline=val_pipeline,
        test_mode=True))
test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=dict(video=data_root_val),
        pipeline=test_pipeline,
        test_mode=True))

val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=40, val_begin=1, dynamic_intervals=[(1, 5), (25, 1)])
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# optimizer
optim_wrapper = dict(
    type='GradMonitorAmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=3e-4, betas=(0.9, 0.999), weight_decay=0.05),
    constructor='GradMonitorSwinOptimWrapperConstructor',
    paramwise_cfg=dict(class_embedding=dict(decay_mult=0.),
                        positional_embedding=dict(decay_mult=0.),
                        temporal_embedding=dict(decay_mult=0.),
                        absolute_pos_embed=dict(decay_mult=0.), # 这玩意不一定有，写着反正没损失
                        ln_1=dict(decay_mult=0.),
                        ln_2=dict(decay_mult=0.),
                        ln_pre=dict(decay_mult=0.),
                        ln_post=dict(decay_mult=0.),
                        scale=dict(decay_mult=0.), # TODO 理论上不需要
                                    ))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=True,
        begin=0,
        end=2.5,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=40,
        eta_min=0,
        by_epoch=True,
        begin=0,
        end=40)
]

find_unused_parameters = True
auto_scale_lr = dict(enable=False, base_batch_size=64)

