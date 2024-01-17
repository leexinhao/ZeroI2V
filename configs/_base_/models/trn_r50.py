# model settings
model = dict(
    type='Recognizer2D',
    backbone=dict(
        type='ResNet',
        pretrained='pretrained_models/resnet50-0676ba61.pth',
        depth=50,
        norm_eval=False,
        partial_bn=True),
    cls_head=dict(
        type='TRNHead',
        num_classes=400,
        in_channels=2048,
        num_segments=8,
        spatial_type='avg',
        relation_type='TRNMultiScale',
        hidden_dim=256,
        dropout_ratio=0.8,
        init_std=0.001,
        average_clips='prob'),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCHW'))
