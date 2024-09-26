# GeNSeg-Net
## GeNSeg-Net: A General Segmentation Framework for Any Nucleus in Immunohistochemistry Images (ACM MM 2024)

This repository includes three projects:

1. **1_enhancement_model_train** is for training the enhancement model (stage 1).

2. **2_segmentation_model_train** is for training the segmentation model (stage 2).

3. **GeNSegNet_test** is for testing the entire GeNSeg-Net.

The steps for using **1_enhancement_model_train** are as follows:

1. Place immunohistochemistry (IHC) images and annotated images (training data in the first stage) in `datasets/dataset_enhancement_model/A_img` and `datasets/dataset_enhancement_model/B_img_ann` respectively, following the examples in the folders. Then, run command `python datasets/combine_A_and_B.py` to generate the final form of training data.

2. Run command `python train.py --dataroot datasets/dataset_enhancement_model/AB --name enhancement_model --model enhancement` to start training. If you want to visualize the training process, you need to first run the command `python -m visdom.server`.

The steps for using **2_segmentation_model_train** are as follows:

1. Place original images and annotated images (training data in the second stage) in `dataset/nucleus_dataset/img` and `dataset/nucleus_dataset/annotation` respectively, following the examples in the folders. Then, run command `python dataset/dataset_split.py` to generate the final form of training data.

2. Run command `python train.py configs/unet/total_config.py --work-dir=log` to start training.


Code is being continuously updated.

During the training of the second stage (respitory "segmentation_model_train"), you can start training by running the command: python train.py configs/unet/total_config.py --work-dir=log

If you encounter the error "TypeError: cannot unpack non-iterable int object: h_stride, w_stride = int(self.test_cfg.stride)", please refer to [this issue](https://github.com/open-mmlab/mmsegmentation/issues/843) on GitHub.

