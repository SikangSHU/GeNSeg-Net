# GeNSeg-Net

## GeNSeg-Net: A General Segmentation Framework for Any Nucleus in Immunohistochemistry Images (ACM MM 2024)

### Repository Structure
This repository contains three key projects as follows.

1. **`1_enhancement_model_train`**: Training the enhancement model (Stage 1 of GeNSeg-Net).
2. **`2_segmentation_model_train`**: Training the segmentation model (Stage 2 of GeNSeg-Net).
3. **`GeNSegNet_test`**: Testing the entire GeNSeg-Net.

---

## Instructions

### 1. 1_enhancement_model_train

#### Steps:
1. Prepare the training data
   - Place immunohistochemistry (IHC) images in `datasets/dataset_enhancement_model/A_img`.
   - Place annotated images in `datasets/dataset_enhancement_model/B_img_ann`.
   - Run the command:
     ```bash
     python datasets/combine_A_and_B.py
     ```
     This generates the final form of training data.

2. Start training
   ```bash
   python train.py --dataroot datasets/dataset_enhancement_model/AB --name enhancement_model --model enhancement
   ```

3. (Optional) To visualize the training process, run the following command before starting.
   ```bash
   python -m visdom.server
   ```

---

### 2. 2_segmentation_model_train

#### Steps:
1. Environment setup
   - This code is inspired by [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) [1].
   - Follow the environment setup instructions from MMSegmentation.

2. Prepare the training data
   - Place original images in `dataset/nucleus_dataset/img`.
   - Place annotated images in `dataset/nucleus_dataset/annotation`.
   - Run the command:
     ```bash
     python dataset/dataset_split.py
     ```
     This generates the final form of training data.

3. Start training
   ```bash
   python train.py configs/unet/total_config.py --work-dir=log
   ```

4. Note
   - If you encounter the error:
     ```
     TypeError: cannot unpack non-iterable int object: h_stride, w_stride = int(self.test_cfg.stride)
     ```
     Refer to [this GitHub issue](https://github.com/open-mmlab/mmsegmentation/issues/843).

---

### 3. GeNSegNet_test

#### Steps:
1. Prepare the trained models
   - Place `latest_net_G.pth` from `checkpoints/enhancement_model` (output of `1_enhancement_model_train`) into `flask_test/enhancement_model/checkpoints/test`.
   - Place `iter_xxx.pth` from `log` (output of `2_segmentation_model_train`) into the `log` folder.
   - Update the filename in `flask_test/flask_test.py` to match the model filename.

2. Prepare test data
   - Place brightfield or fluorescence IHC images (test data) in `flask_test/1_ori_img`, following the examples in the folder.

3. Start the server
   - Navigate to the `flask_test` directory:
     ```bash
     cd flask_test
     ```  
   - Open the server and test:
     ```bash
     python flask_test.py
     ```
     - For fluorescence images:
       ```bash
       python flask_test_for_flu.py
       ```
     - For brightfield images:
       ```bash
       python flask_test_for_bri.py
       ```

---

## Citation

```bibtex
[1] @misc{mmseg2020,
    title={{MMSegmentation}: OpenMMLab Semantic Segmentation Toolbox and Benchmark},
    author={MMSegmentation Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmsegmentation}},
    year={2020}
}

