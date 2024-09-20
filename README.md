# GeNSeg-Net
GeNSeg-Net: A General Segmentation Framework for Any Nucleus in Immunohistochemistry Images

Accepted by ACMMM 2024

Code is being continuously updated.

During the training of the second stage (respitory "segmentation_model_train"), you can start training by running the command: python train.py configs/unet/total_config.py --work-dir=log

If you encounter the error "TypeError: cannot unpack non-iterable int object: h_stride, w_stride = int(self.test_cfg.stride)", please refer to [this issue](https://github.com/open-mmlab/mmsegmentation/issues/843) on GitHub.

