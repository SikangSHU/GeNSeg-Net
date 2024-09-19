# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=True)#
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=20000)
checkpoint_config = dict(by_epoch=False, interval=2000)

# runner = dict(type='EpochBasedRunner',max_epochs=200)
# checkpoint_config = dict(by_epoch=True,interval=20)  # save checkpoint per 20 epochs

evaluation = dict(interval=2000, metric='mIoU', pre_eval=True)
