# GeNSeg-Net
## GeNSeg-Net: A General Segmentation Framework for Any Nucleus in Immunohistochemistry Images (ACM MM 2024)

### This repository includes three projects:

1. **1_enhancement_model_train** is for training the enhancement model (stage 1).

2. **2_segmentation_model_train** is for training the segmentation model (stage 2).

3. **GeNSegNet_test** is for testing the entire GeNSeg-Net.

### The steps for using **1_enhancement_model_train** are as follows:

1. Place immunohistochemistry (IHC) images and annotated images (training data in the first stage) in `datasets/dataset_enhancement_model/A_img` and `datasets/dataset_enhancement_model/B_img_ann` respectively, following the examples in the folders. Then, run command `python datasets/combine_A_and_B.py` to generate the final form of training data.

2. Run command `python train.py --dataroot datasets/dataset_enhancement_model/AB --name enhancement_model --model enhancement` to start training. If you want to visualize the training process, you need to first run the command `python -m visdom.server`.

### The steps for using **2_segmentation_model_train** are as follows:

1. This part of the code is inspired by mmsegmentation [2], and the related environment setup can refer to the method of mmsegmentation [2].

2. Place original images and annotated images (training data in the second stage) in `dataset/nucleus_dataset/img` and `dataset/nucleus_dataset/annotation` respectively, following the examples in the folders. Then, run command `python dataset/dataset_split.py` to generate the final form of training data.

3. Run command `python train.py configs/unet/total_config.py --work-dir=log` to start training.

4. Note: If you encounter the error "TypeError: cannot unpack non-iterable int object: h_stride, w_stride = int(self.test_cfg.stride)", please refer to [this issue](https://github.com/open-mmlab/mmsegmentation/issues/843) on GitHub.

### The steps for using **GeNSegNet_test** are as follows:

1. Place the `latest_net_G.pth` file obtained from the training in the `checkpoints/enhancement_model` folder of the first phase repository into the `flask_test/enhancement_model/checkpoints/test` folder of this repository. Place the `iter_xxx.pth` file obtained from the training in the `log` folder of the second phase repository into the `log` folder of this repository, and modify the filename in `flask_test/flask_test.py` accordingly.

2. Place brightfield or fluorescence immunohistochemistry images (test data) in `flask_test/1_ori_img`, following the examples in the folder.

3. Run command `python flask_test/flask_test.py` to open the server. Meanwhile, run command `python flask_test_for_flu.py` for fluorescence images, or run `python flask_test_for_bri.py` for brightfield images.


Code is being continuously updated.

