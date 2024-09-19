
_base_ = [
    '../_base_/models/fcn_unet_s5-d16.py', '../_base_/datasets/my_unet_seg.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]


model = dict(
    decode_head=dict(num_classes=3, loss_decode=[dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0, class_weight=[0.9, 1.1, 1.2]),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0, class_weight=[0.9, 1.1, 1.2])]),
    auxiliary_head=dict(num_classes=3, loss_decode=[dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0, class_weight=[0.9, 1.1, 1.2]),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0, class_weight=[0.9, 1.1, 1.2])]),
    )





