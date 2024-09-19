import mmcv
import os.path as osp
data_root = "nucleus_dataset"
ann_dir = "annotation"
split_dir = 'dataset_split'
mmcv.mkdir_or_exist(osp.join(data_root, split_dir))
filename_list = [osp.splitext(filename)[0] for filename in mmcv.scandir(
    osp.join(data_root, ann_dir), suffix='.png')]
with open(osp.join(data_root, split_dir, 'train.txt'), 'w') as f:
    # select 4/5 (or other ratios) as training data
    train_length = int(len(filename_list)*4/5)
    f.writelines(line + '\n' for line in filename_list[:train_length])
with open(osp.join(data_root, split_dir, 'val.txt'), 'w') as f:
    # select 1/5 (or other ratios) as validation data
    f.writelines(line + '\n' for line in filename_list[train_length:])
