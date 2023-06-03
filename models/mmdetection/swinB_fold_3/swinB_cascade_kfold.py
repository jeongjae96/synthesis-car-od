fold_num = 3
data_root = '/opt/ml/synthesis-car-od/data/'
dataset_type = 'CocoDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
size_min = 512
size_max = 1024
multi_scale = [(512, 512), (544, 544), (576, 576), (608, 608), (640, 640),
               (672, 672), (704, 704), (736, 736), (768, 768), (800, 800),
               (832, 832), (864, 864), (896, 896), (928, 928), (960, 960),
               (992, 992), (1024, 1024)]
multi_scale_light = (512, 512)
alb_transform = [
    dict(type='VerticalFlip', p=0.15),
    dict(type='HorizontalFlip', p=0.3),
    dict(
        type='OneOf',
        transforms=[
            dict(type='GaussNoise', p=1.0),
            dict(type='GaussianBlur', p=1.0),
            dict(type='Blur', p=1.0)
        ],
        p=0.1),
    dict(
        type='OneOf',
        transforms=[
            dict(type='RandomGamma', p=1.0),
            dict(type='HueSaturationValue', p=1.0),
            dict(type='ChannelDropout', p=1.0),
            dict(type='ChannelShuffle', p=1.0),
            dict(type='RGBShift', p=1.0)
        ],
        p=0.1),
    dict(
        type='OneOf',
        transforms=[
            dict(type='ShiftScaleRotate', p=1.0),
            dict(type='RandomRotate90', p=1.0)
        ],
        p=0.1)
]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(512, 512), (544, 544), (576, 576), (608, 608), (640, 640),
                   (672, 672), (704, 704), (736, 736), (768, 768), (800, 800),
                   (832, 832), (864, 864), (896, 896), (928, 928), (960, 960),
                   (992, 992), (1024, 1024)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(
        type='Albu',
        transforms=[
            dict(type='VerticalFlip', p=0.15),
            dict(type='HorizontalFlip', p=0.3),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='GaussNoise', p=1.0),
                    dict(type='GaussianBlur', p=1.0),
                    dict(type='Blur', p=1.0)
                ],
                p=0.1),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='RandomGamma', p=1.0),
                    dict(type='HueSaturationValue', p=1.0),
                    dict(type='ChannelDropout', p=1.0),
                    dict(type='ChannelShuffle', p=1.0),
                    dict(type='RGBShift', p=1.0)
                ],
                p=0.1),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='ShiftScaleRotate', p=1.0),
                    dict(type='RandomRotate90', p=1.0)
                ],
                p=0.1)
        ],
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap=dict(img='image', gt_bboxes='bboxes'),
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(
                type='Resize',
                img_scale=(512, 512),
                multiscale_mode='value',
                keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(512, 512), (544, 544), (576, 576), (608, 608), (640, 640),
                   (672, 672), (704, 704), (736, 736), (768, 768), (800, 800),
                   (832, 832), (864, 864), (896, 896), (928, 928), (960, 960),
                   (992, 992), (1024, 1024)],
        flip=False,
        transforms=[
            dict(
                type='Resize',
                img_scale=[(512, 512), (544, 544), (576, 576), (608, 608),
                           (640, 640), (672, 672), (704, 704), (736, 736),
                           (768, 768), (800, 800), (832, 832), (864, 864),
                           (896, 896), (928, 928), (960, 960), (992, 992),
                           (1024, 1024)],
                multiscale_mode='value',
                keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
classes = (
    'chevrolet_malibu_sedan_2012_2016', 'chevrolet_malibu_sedan_2017_2019',
    'chevrolet_spark_hatchback_2016_2021', 'chevrolet_trailblazer_suv_2021_',
    'chevrolet_trax_suv_2017_2019', 'genesis_g80_sedan_2016_2020',
    'genesis_g80_sedan_2021_', 'genesis_gv80_suv_2020_',
    'hyundai_avante_sedan_2011_2015', 'hyundai_avante_sedan_2020_',
    'hyundai_grandeur_sedan_2011_2016', 'hyundai_grandstarex_van_2018_2020',
    'hyundai_ioniq_hatchback_2016_2019', 'hyundai_sonata_sedan_2004_2009',
    'hyundai_sonata_sedan_2010_2014', 'hyundai_sonata_sedan_2019_2020',
    'kia_carnival_van_2015_2020', 'kia_carnival_van_2021_',
    'kia_k5_sedan_2010_2015', 'kia_k5_sedan_2020_', 'kia_k7_sedan_2016_2020',
    'kia_mohave_suv_2020_', 'kia_morning_hatchback_2004_2010',
    'kia_morning_hatchback_2011_2016', 'kia_ray_hatchback_2012_2017',
    'kia_sorrento_suv_2015_2019', 'kia_sorrento_suv_2020_',
    'kia_soul_suv_2014_2018', 'kia_sportage_suv_2016_2020',
    'kia_stonic_suv_2017_2019', 'renault_sm3_sedan_2015_2018',
    'renault_xm3_suv_2020_', 'ssangyong_korando_suv_2019_2020',
    'ssangyong_tivoli_suv_2016_2020')
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type='CocoDataset',
        classes=(
            'chevrolet_malibu_sedan_2012_2016',
            'chevrolet_malibu_sedan_2017_2019',
            'chevrolet_spark_hatchback_2016_2021',
            'chevrolet_trailblazer_suv_2021_', 'chevrolet_trax_suv_2017_2019',
            'genesis_g80_sedan_2016_2020', 'genesis_g80_sedan_2021_',
            'genesis_gv80_suv_2020_', 'hyundai_avante_sedan_2011_2015',
            'hyundai_avante_sedan_2020_', 'hyundai_grandeur_sedan_2011_2016',
            'hyundai_grandstarex_van_2018_2020',
            'hyundai_ioniq_hatchback_2016_2019',
            'hyundai_sonata_sedan_2004_2009', 'hyundai_sonata_sedan_2010_2014',
            'hyundai_sonata_sedan_2019_2020', 'kia_carnival_van_2015_2020',
            'kia_carnival_van_2021_', 'kia_k5_sedan_2010_2015',
            'kia_k5_sedan_2020_', 'kia_k7_sedan_2016_2020',
            'kia_mohave_suv_2020_', 'kia_morning_hatchback_2004_2010',
            'kia_morning_hatchback_2011_2016', 'kia_ray_hatchback_2012_2017',
            'kia_sorrento_suv_2015_2019', 'kia_sorrento_suv_2020_',
            'kia_soul_suv_2014_2018', 'kia_sportage_suv_2016_2020',
            'kia_stonic_suv_2017_2019', 'renault_sm3_sedan_2015_2018',
            'renault_xm3_suv_2020_', 'ssangyong_korando_suv_2019_2020',
            'ssangyong_tivoli_suv_2016_2020'),
        ann_file='/opt/ml/synthesis-car-od/data/coco_sgkf/train_3.json',
        img_prefix='/opt/ml/synthesis-car-od/data/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='Resize',
                img_scale=[(512, 512), (544, 544), (576, 576), (608, 608),
                           (640, 640), (672, 672), (704, 704), (736, 736),
                           (768, 768), (800, 800), (832, 832), (864, 864),
                           (896, 896), (928, 928), (960, 960), (992, 992),
                           (1024, 1024)],
                multiscale_mode='value',
                keep_ratio=True),
            dict(
                type='Albu',
                transforms=[
                    dict(type='VerticalFlip', p=0.15),
                    dict(type='HorizontalFlip', p=0.3),
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(type='GaussNoise', p=1.0),
                            dict(type='GaussianBlur', p=1.0),
                            dict(type='Blur', p=1.0)
                        ],
                        p=0.1),
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(type='RandomGamma', p=1.0),
                            dict(type='HueSaturationValue', p=1.0),
                            dict(type='ChannelDropout', p=1.0),
                            dict(type='ChannelShuffle', p=1.0),
                            dict(type='RGBShift', p=1.0)
                        ],
                        p=0.1),
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(type='ShiftScaleRotate', p=1.0),
                            dict(type='RandomRotate90', p=1.0)
                        ],
                        p=0.1)
                ],
                bbox_params=dict(
                    type='BboxParams',
                    format='pascal_voc',
                    label_fields=['gt_labels'],
                    min_visibility=0.0,
                    filter_lost_elements=True),
                keymap=dict(img='image', gt_bboxes='bboxes'),
                update_pad_shape=False,
                skip_img_without_anno=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ]),
    val=dict(
        type='CocoDataset',
        classes=(
            'chevrolet_malibu_sedan_2012_2016',
            'chevrolet_malibu_sedan_2017_2019',
            'chevrolet_spark_hatchback_2016_2021',
            'chevrolet_trailblazer_suv_2021_', 'chevrolet_trax_suv_2017_2019',
            'genesis_g80_sedan_2016_2020', 'genesis_g80_sedan_2021_',
            'genesis_gv80_suv_2020_', 'hyundai_avante_sedan_2011_2015',
            'hyundai_avante_sedan_2020_', 'hyundai_grandeur_sedan_2011_2016',
            'hyundai_grandstarex_van_2018_2020',
            'hyundai_ioniq_hatchback_2016_2019',
            'hyundai_sonata_sedan_2004_2009', 'hyundai_sonata_sedan_2010_2014',
            'hyundai_sonata_sedan_2019_2020', 'kia_carnival_van_2015_2020',
            'kia_carnival_van_2021_', 'kia_k5_sedan_2010_2015',
            'kia_k5_sedan_2020_', 'kia_k7_sedan_2016_2020',
            'kia_mohave_suv_2020_', 'kia_morning_hatchback_2004_2010',
            'kia_morning_hatchback_2011_2016', 'kia_ray_hatchback_2012_2017',
            'kia_sorrento_suv_2015_2019', 'kia_sorrento_suv_2020_',
            'kia_soul_suv_2014_2018', 'kia_sportage_suv_2016_2020',
            'kia_stonic_suv_2017_2019', 'renault_sm3_sedan_2015_2018',
            'renault_xm3_suv_2020_', 'ssangyong_korando_suv_2019_2020',
            'ssangyong_tivoli_suv_2016_2020'),
        ann_file='/opt/ml/synthesis-car-od/data/coco_sgkf/val_3.json',
        img_prefix='/opt/ml/synthesis-car-od/data/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 512),
                flip=False,
                transforms=[
                    dict(
                        type='Resize',
                        img_scale=(512, 512),
                        multiscale_mode='value',
                        keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CocoDataset',
        classes=(
            'chevrolet_malibu_sedan_2012_2016',
            'chevrolet_malibu_sedan_2017_2019',
            'chevrolet_spark_hatchback_2016_2021',
            'chevrolet_trailblazer_suv_2021_', 'chevrolet_trax_suv_2017_2019',
            'genesis_g80_sedan_2016_2020', 'genesis_g80_sedan_2021_',
            'genesis_gv80_suv_2020_', 'hyundai_avante_sedan_2011_2015',
            'hyundai_avante_sedan_2020_', 'hyundai_grandeur_sedan_2011_2016',
            'hyundai_grandstarex_van_2018_2020',
            'hyundai_ioniq_hatchback_2016_2019',
            'hyundai_sonata_sedan_2004_2009', 'hyundai_sonata_sedan_2010_2014',
            'hyundai_sonata_sedan_2019_2020', 'kia_carnival_van_2015_2020',
            'kia_carnival_van_2021_', 'kia_k5_sedan_2010_2015',
            'kia_k5_sedan_2020_', 'kia_k7_sedan_2016_2020',
            'kia_mohave_suv_2020_', 'kia_morning_hatchback_2004_2010',
            'kia_morning_hatchback_2011_2016', 'kia_ray_hatchback_2012_2017',
            'kia_sorrento_suv_2015_2019', 'kia_sorrento_suv_2020_',
            'kia_soul_suv_2014_2018', 'kia_sportage_suv_2016_2020',
            'kia_stonic_suv_2017_2019', 'renault_sm3_sedan_2015_2018',
            'renault_xm3_suv_2020_', 'ssangyong_korando_suv_2019_2020',
            'ssangyong_tivoli_suv_2016_2020'),
        ann_file='/opt/ml/synthesis-car-od/data/test.json',
        img_prefix='/opt/ml/synthesis-car-od/data/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=[(512, 512), (544, 544), (576, 576), (608, 608),
                           (640, 640), (672, 672), (704, 704), (736, 736),
                           (768, 768), (800, 800), (832, 832), (864, 864),
                           (896, 896), (928, 928), (960, 960), (992, 992),
                           (1024, 1024)],
                flip=False,
                transforms=[
                    dict(
                        type='Resize',
                        img_scale=[(512, 512), (544, 544), (576, 576),
                                   (608, 608), (640, 640), (672, 672),
                                   (704, 704), (736, 736), (768, 768),
                                   (800, 800), (832, 832), (864, 864),
                                   (896, 896), (928, 928), (960, 960),
                                   (992, 992), (1024, 1024)],
                        multiscale_mode='value',
                        keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
lr = 5e-05
optimizer = dict(type='AdamW', lr=5e-05, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=300,
    warmup_ratio=0.1,
    min_lr_ratio=7e-06)
total_epochs = 40
expr_name = 'swinB_fold_3'
dist_params = dict(backend='nccl')
checkpoint_config = dict(interval=9)
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='swinB-cascade-kfold',
                name='swinB_fold_3',
                entity='cv-09'))
    ])
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
evaluation = dict(save_best='bbox_mAP', metric=['bbox'])
runner = dict(type='EpochBasedRunner', max_epochs=40)
work_dir = './work_dirs/swinB_fold_3'
gpu_ids = [0]
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22kto1k.pth'
model = dict(
    type='CascadeRCNN',
    backbone=dict(
        type='SwinTransformer',
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22kto1k.pth'
        )),
    neck=dict(
        type='FPN',
        in_channels=[128, 256, 512, 1024],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0)),
    roi_head=dict(
        type='CascadeRoIHead',
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=34,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=34,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=34,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ]),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)
        ]),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.0,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)))
auto_resume = False
