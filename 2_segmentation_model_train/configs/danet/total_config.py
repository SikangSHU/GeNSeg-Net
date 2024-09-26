# _base_ = [
#     '../_base_/models/danet_r50-d8.py', '../_base_/datasets/my_road_detect.py',
#     '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
# ]
_base_ = [
    '../_base_/models/pspnet_r50-d8.py', '../_base_/datasets/my_path_detect.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]
model = dict(
	decode_head=dict(num_classes=3),auxiliary_head=dict(num_classes=3))

# runner = dict(type='EpochBasedRunner',
# 							max_epoch='200')
# checkpoint_config = dict(by_epoch=True,
# 													interval=2)  # save checkpoint per 20 epochs

