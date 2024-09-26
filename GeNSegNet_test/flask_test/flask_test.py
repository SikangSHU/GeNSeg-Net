import cv2
import numpy as np
from PIL import Image
import multiprocessing as mp
from flask import Flask, jsonify, request
from flask_cors import CORS
import sys
import os
import shutil
from torch.utils.data import Dataset
from werkzeug.utils import secure_filename
from mmseg.apis import inference_segmentor, init_segmentor
from enhancement_model.options.test_options import TestOptions
from enhancement_model.data import create_dataset
from enhancement_model.models import create_model
from enhancement_model.util.visualizer import save_images

from concurrent.futures import ThreadPoolExecutor


try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')

app = Flask(__name__)
CORS(app, resources=r'/*')


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


class get_image(Dataset):
    def __init__(self, path):
        self.path = path
        self.name = os.listdir(path)

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        pusedo_pic_name = self.name[index]  # xx.png
        pusedo_path = os.path.join(self.path, pusedo_pic_name)

        img0 = cv2.imread(pusedo_path)

        return img0, pusedo_pic_name


""" Function used to convert RGB to HSI. """
def rgb2hsi(image):

    b, g, r = cv2.split(image.astype(np.float32))     # Read channels.
    eps = 1e-6                                        # Avoid dividing zero.

    min_rgb = cv2.min(r, cv2.min(b, g))
    img_s = 1 - 3 * min_rgb / (r + g + b + eps)       # Component S.

    temp_s = img_s - np.min(img_s)
    img_s = temp_s / np.max(temp_s)

    return img_s


""" If the file not exists, then creates. If exists, then clears. """
def RemoveDir(filepath):

    if not os.path.exists(filepath):
        os.mkdir(filepath)
    else:
        shutil.rmtree(filepath)
        os.mkdir(filepath)


def replace_pixels(result_img, ori_img):

    result = Image.open(result_img)
    ori = Image.open(ori_img)

    width, height = result.size

    ori_pixels = ori.load()

    for x in range(width):
        for y in range(2):
            result.putpixel((x, y), ori_pixels[x, y])
    for x in range(width):
        for y in range(height - 2, height):
            result.putpixel((x, y), ori_pixels[x, y])
    for x in range(2):
        for y in range(height):
            result.putpixel((x, y), ori_pixels[x, y])
    for x in range(width - 2, width):
        for y in range(height):
            result.putpixel((x, y), ori_pixels[x, y])

    return result


def delete_invalid_images(folder_path):
    # Iterate through all files in the folder.
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            # Check if the file size is less than 5 KB, and if the file type is an image file.
            if os.path.getsize(file_path) < 5 * 1024 and is_image_file(file):
                print(f"Deleting {file_path}")
                os.remove(file_path)


""" Check if the file extension is a common image format. """
def is_image_file(file_path):

    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    return any(file_path.lower().endswith(ext) for ext in image_extensions)


def process_bri_image(filename, path_in, path_out):

    I = cv2.imread(path_in + "/" + filename)
    s = rgb2hsi(I)
    s = np.clip(s, 0, 1)
    s = (s * 255.0).astype(np.uint8)
    cv2.imwrite(path_out + "/" + filename, s)


def Part_2_2_b(filename, basename):

    img_ori = cv2.imread(r'./1_ori_img/{}.png'.format(basename))

    unet_model_result = cv2.imread(r'./4_unet_model_result/{}'.format(filename), 0)
    _, unet_model_result_bg = cv2.threshold(unet_model_result, 0, 255, cv2.THRESH_BINARY)
    _, unet_model_result_fg = cv2.threshold(unet_model_result, 128, 255, cv2.THRESH_BINARY)
    unet_model_result_fg = unet_model_result_bg - unet_model_result_fg
    fg_ori = cv2.medianBlur(unet_model_result_fg, 3)
    bg_ori = cv2.medianBlur(unet_model_result_bg, 3)

    unknown_ori = bg_ori - fg_ori
    _, markers_ori = cv2.connectedComponents(fg_ori)
    markers_ori = markers_ori + 1
    markers_ori[unknown_ori == 255] = 0

    markers_cv_ori = cv2.watershed(img_ori, markers_ori)
    markers_cv_ori[markers_cv_ori == -1] = 0
    markers_cv_ori[markers_cv_ori == 1] = 0


    img_1 = cv2.imread(r'./1_ori_img/{}.png'.format(basename))
    img_2 = cv2.imread(r'./1_ori_img/{}.png'.format(basename))
    zero_512_c3 = np.zeros((512, 512, 3), dtype=np.uint8)
    zero_512_c1 = np.zeros((512, 512), dtype=np.uint8)

    inst_map_1 = np.zeros(markers_cv_ori.shape, dtype="uint8")
    inst_map_1[markers_cv_ori == 0] = 0
    inst_map_1[markers_cv_ori > 0] = 255
    kernel_bg_1 = np.ones((2, 2), np.uint8)
    bg_1 = cv2.dilate(inst_map_1, kernel_bg_1, iterations=1)
    fg_1 = np.zeros(markers_cv_ori.shape, dtype="uint8")
    kernel_fg_1 = np.ones((3, 3), np.uint8)
    for label in np.unique(markers_cv_ori):
        if label > 0:
            mask_tem_1 = np.zeros(fg_1.shape, dtype="uint8")
            mask_tem_1[markers_cv_ori == label] = 255
            mask_tem_1 = cv2.erode(mask_tem_1, kernel_fg_1, iterations=1)
            fg_1[mask_tem_1 == 255] = 255

    unknown_1 = bg_1 - fg_1
    _, markers_1 = cv2.connectedComponents(fg_1)
    markers_1 = markers_1 + 1
    markers_1[unknown_1 == 255] = 0

    markers_1_copy = markers_1.copy()
    markers_1_copy[markers_1 == 0] = 1

    bg_copy = zero_512_c3.copy()
    bg_copy[:, :, 0] = bg_1
    bg_copy[:, :, 1] = bg_1
    bg_copy[:, :, 2] = bg_1
    markers_cv_1 = cv2.watershed(bg_copy, markers_1)
    img_1[markers_cv_1 == -1] = [0, 255, 0]
    cv2.imwrite('./watershed/{}.png'.format(basename), img_1)
    result_img_1 = r'./watershed/{}.png'.format(basename)
    ori_img_1 = r'./1_ori_img/{}.png'.format(basename)
    result_img_1 = replace_pixels(result_img_1, ori_img_1)
    result_img_1.save(r'./seg_result/{}.png'.format(basename))


    markers_cv_1_copy = zero_512_c1.copy()
    markers_cv_1_copy[markers_cv_1 > 1] = 255
    markers_cv_1_copy[markers_cv_1 == 1] = 0
    markers_cv_1_copy[markers_cv_1 == -1] = 0

    kernel_dilate_2 = np.ones((3, 3), np.uint8)
    cell_after_watershed_dilate = cv2.dilate(markers_cv_1_copy, kernel_dilate_2, iterations=1)

    cell_after_watershed_dilate_copy = zero_512_c3.copy()
    cell_after_watershed_dilate_copy[:, :, 0] = cell_after_watershed_dilate
    cell_after_watershed_dilate_copy[:, :, 1] = cell_after_watershed_dilate
    cell_after_watershed_dilate_copy[:, :, 2] = cell_after_watershed_dilate

    unknown_2 = cell_after_watershed_dilate - fg_1
    markers_1_copy[unknown_2 == 255] = 0

    markers_cv_2 = cv2.watershed(cell_after_watershed_dilate_copy, markers_1_copy)
    img_2[markers_cv_2 == -1] = [0, 255, 0]
    cv2.imwrite('./watershed_dilate/{}.png'.format(basename), img_2)
    result_img_2 = r'./watershed_dilate/{}.png'.format(basename)
    ori_img_2 = r'./1_ori_img/{}.png'.format(basename)
    result_img_2 = replace_pixels(result_img_2, ori_img_2)
    result_img_2.save(r'./seg_result_dilate/{}.png'.format(basename))


def Part_2_2(model_2_2):

    image_2_2_a = get_image(r'./3_enhancement_model_result')
    print('Part 2-2 Segmentation Model Test. Total Number of Images: ', image_2_2_a.__len__())

    for i in range(image_2_2_a.__len__()):
        filename = image_2_2_a[i][1]
        img = './3_enhancement_model_result/{}'.format(filename)
        result = inference_segmentor(model_2_2, img)
        cv2.imwrite(r'./4_unet_model_result/{}'.format(filename), (result[0] / 2 * 255))


    image_2_2_b = get_image(r'./4_unet_model_result')
    filename_list = []
    basename_list = []
    for i in range(image_2_2_b.__len__()):
        filename = image_2_2_b[i][1]
        filename = r'{}'.format(filename)
        basename = (r'{}'.format(filename)).replace('.png', '')
        filename_list.append(filename)
        basename_list.append(basename)

    zip_args = list(zip(filename_list, basename_list))
    pool = mp.Pool(4)
    pool.starmap(Part_2_2_b, zip_args)

    print('Part 2-2 Segmentation Model Test Over.')


""" Entrance of Flask """
@app.route('/GeNSegNettest', methods=['POST'])
def response_to_frontEnd():
    if request.method == 'POST':

        type_param = request.form.get('type')

        RemoveDir('2_ori_img_to_gray')
        RemoveDir('3_enhancement_model_result')
        RemoveDir('4_unet_model_result')
        RemoveDir('watershed')
        RemoveDir('watershed_dilate')
        RemoveDir('seg_result')
        RemoveDir('seg_result_dilate')

        file_db_paths = []
        uploaded_files = request.files.getlist('file')
        for file in uploaded_files:
            filename = secure_filename(file.filename)
            upload_path = './1_ori_img'
            upload_path = os.path.join(upload_path, filename)
            file_db_paths.append(upload_path)
            file.save(upload_path)


        # Part 1: Grayscale Conversion.
        image_1 = get_image(r'./1_ori_img')
        print('Part1 Grayscale Conversion Begins. Total Number of Images: ', image_1.__len__())

        # Conversion for Fluorescence Images.
        if type_param == "flu":
            print("Grayscale Conversion for Fluorescence Images.")
            path_in = './1_ori_img'
            path_out = './2_ori_img_to_gray'
            file_list_in = os.listdir(path_in)
            for filename in file_list_in:
                I = cv2.imread(path_in + "/" + filename, 0)
                num_50h = (I > 50).sum()
                if num_50h > 800:
                    cv2.imwrite(path_out + "/" + filename, I)

        # Conversion for Brightfield Images.
        elif type_param == "bri":
            print("Grayscale Conversion for Brightfield Images.")
            path_in = './1_ori_img'
            path_out = './2_ori_img_to_gray'
            file_list_in = os.listdir(path_in)

            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(process_bri_image, filename, path_in, path_out) for filename in file_list_in]
                for future in futures:
                    future.result()

        else:
            print("Invalid Format.")

        delete_invalid_images(path_out)


        # Part 2: GeNSegNet Test.
        image_2 = get_image(r'./2_ori_img_to_gray')
        print('Part 2 GeNSegNet Test Begins. Total Number of Images: ', image_2.__len__())

        # Part 2-1: Enhancement Model Test
        image_2_1 = get_image(r'./1_ori_img')
        print('Part 2-1 Enhancement Model Test Begins. Total Number of Images: ', image_2_1.__len__())

        opt = TestOptions().parse()    # get test options
        # hard-code some parameters for test
        opt.num_threads = 0            # test code only supports num_threads = 0
        opt.batch_size = 1             # test code only supports batch_size = 1
        opt.serial_batches = True      # disable data shuffling; comment this line if results on randomly chosen images are needed.
        opt.no_flip = True             # no flip; comment this line if results on flipped images are needed.
        opt.display_id = -1            # no visdom display; the test code saves the results to a HTML file.
        dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
        model = create_model(opt)      # create a model given opt.model and other options
        model.setup(opt)               # regular setup: load and print networks; create schedulers

        results_dir = opt.results_dir
        for i, data in enumerate(dataset):
            if i >= opt.num_test:      # only apply our model to opt.num_test images.
                break
            model.set_input(data)      # unpack data from data loader
            model.test()

            visuals = model.get_current_visuals()       # get image results
            img_path = model.get_image_paths()          # get image paths
            if i % 5 == 0:                              # save images to an HTML file
                print('Processing (%04d)-th Image... %s' % (i, img_path))
            save_images(results_dir, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize,
                        use_wandb=opt.use_wandb)

        print('Part 2-1 Enhancement Model Test Over.')


        # Part 2-2: Segmentation Model Test
        config_file = r"../configs/unet/total_config.py"
        checkpoint_file_1 = r"../log/iter_12000_pi_un_wa_0911.pth"     # TODO: check
        model_2_2 = init_segmentor(config_file, checkpoint_file_1, device='cuda:0')

        Part_2_2(model_2_2)

        return jsonify({"message": "GeNSegNet_test Over."}), 200


if __name__ == '__main__':
    app.run(port=9000)