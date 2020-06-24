# coding: utf-8

# # Mask R-CNN - Train on Shapes Dataset
# 
# 
# This notebook shows how to train Mask R-CNN on your own dataset. To keep things simple we use a synthetic dataset of shapes (squares, triangles, and circles) which enables fast training. You'd still need a GPU, though, because the network backbone is a Resnet101, which would be too slow to train on a CPU. On a GPU, you can start to get okay-ish results in a few minutes, and good results in less than an hour.
# 
# The code of the *Shapes* dataset is included below. It generates images on the fly, so it doesn't require downloading any data. And it can generate images of any size, so we pick a small image size to train faster. 

#%%
import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import skimage
import pydicom
#%%
# Root directory of the project
MRCNN_DIR = os.path.abspath("../../")
ROOT_DIR = os.path.abspath("../../../")
# Import Mask RCNN
sys.path.append(MRCNN_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
import os.path as osp
from mrcnn.model import log
# get_ipython().magic('matplotlib inline')

# %%
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2"
config = tf.ConfigProto()  
config.gpu_options.allow_growth=True
config.allow_soft_placement=True
sess = tf.Session(config=config)
KTF.set_session(sess)
# Directory to save logs and trained model
MODEL_DIR = os.path.join(MRCNN_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(MRCNN_DIR, "mask_rcnn_coco.h5")
IMAGENET_MODEL_PATH = os.path.join(MRCNN_DIR, "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# %%
import pandas as pd
df = pd.read_excel(io = osp.join(ROOT_DIR,"dataset","pathology.xls"), header = 0)
# %%
# ## Configurations
class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "liver"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 2
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 5 + 1  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    IMAGE_RESIZE_MODE = "none"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (32, 64, 128, 256)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 128
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256
    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 150

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 15


    BACKBONE = "resnet101"

    USE_MINI_MASK = False

    LEARNING_RATE = 1e-5

    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.,
        "mrcnn_mask_dice_loss": 1.,
        "mrcnn_mask_scoring_loss": 1.,
    }

config = ShapesConfig()
config.display()

# ## Notebook Preferences
# %%
def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax

# %%
class DrugDataset(utils.Dataset):
    # label.png中，像素值为0的是背景，1为第一类目标，2维第二类……
    # 获得mask中目标的种类数量（不含背景）
    def get_obj_index(self, image):
        n = np.max(image)
        return n

    def from_txt_get_class(self, image_id):
        info = self.image_info[image_id]
        with open(info['txt_path'])as f:
            labels = []
            for line in f:
                labels.append(line.strip('\n'))
            del labels[0]
        return labels

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        path = self.image_info[image_id]['path']
        if path.endswith(".png" or ".jpg"):
            image = skimage.io.imread(path)
        elif path.endswith(".dcm"):
            ds = pydicom.read_file(path)
            image = ds.pixel_array
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

    # 重新写draw_mask
    def draw_mask(self, count, mask, image, image_id):
        info = self.image_info[image_id]
        for index in range(count):
            for i in range(info['height']):
                for j in range(info['width']):
                    # at_pixel = image.getpixel((i, j))
                    at_pixel = image[i][j]
                    if at_pixel == index + 1:
                        mask[i, j, index] = 255
        return mask

    # 重新写load_shapes，里面包含自己的自己的类别（我的是yellow,white,back三类）
    #这里的class是类别顺序要按着自己网络输出的顺序来添加。
    # 并在self.image_info信息中添加了path、mask_path 、yaml_path
    def load_shapes(self, count, height, width, data_list):
        # self.add_class("liver", 1, "liver-delay")
        # self.add_class("liver", 2, "liver-portal")
        self.add_class("liver", 1, "liver_0")
        self.add_class("liver", 2, "liver_1")
        self.add_class("liver", 3, "liver_2")
        self.add_class("liver", 4, "liver_3")
        self.add_class("liver", 5, "liver_4")
        for i in range(count):
            # img_path = os.path.join(data_list[i], "img.png")
            json_dir_path = data_list[i]
            dcm_parent_path = osp.dirname(json_dir_path).replace("dataset", "dataset_dicom")
            dcm_right_name = osp.basename(json_dir_path).replace("_json", ".dcm").split("-")[2]
            assert osp.isdir(dcm_parent_path)
            dcm_list = os.listdir(dcm_parent_path)
            for x in dcm_list:
                if x.find(dcm_right_name) != -1:
                    dcm_name = x
                    break
            dcm_path = osp.join(dcm_parent_path, dcm_name)
            mask_path = os.path.join(data_list[i], "label.png")
            txt_path = os.path.join(data_list[i], "label_names.txt")
            self.add_image("liver", image_id=i, 
                            path=dcm_path,
                            width=width, height=height, 
                            mask_path=mask_path, txt_path=txt_path)

    # 重写load_mask
    def load_mask(self, image_id):
        info = self.image_info[image_id]
        labels = []
        labels = self.from_txt_get_class(image_id)
        count = len(labels)

        patient_id = osp.basename(osp.dirname(osp.dirname(info["path"]))) # "aXX"
        patient_id_num = int(patient_id[1:]) # XX
        # img = Image.open(info['mask_path'])
        img = cv2.imread(info['mask_path'], cv2.IMREAD_GRAYSCALE)
        num_obj = self.get_obj_index(img)

        mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)

        mask = self.draw_mask(count, mask, img, image_id)
        # occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        # for i in range(count - 1, -1, -1):
        #     mask[:, :, i] = mask[:, :, i] * occlusion
        #     occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        
        labels_from = []
        for i in range(len(labels)):
            label = ""
            if re.findall("liver|[12345]", labels[i]) != []:
                label+="liver_"
                label+=str(df.values[patient_id_num-1,1])
            labels_from.append(label)
        class_ids = np.array([self.class_names.index(s) for s in labels_from])
        # class_ids = np.array([self.class_names.index(s) for s in labels])
        return mask.astype(np.bool), class_ids.astype(np.int32)


# %%
# ## Dataset
# 
# Create a synthetic dataset
# 
# Extend the Dataset class and add a method to load the shapes dataset, `load_shapes()`, and override the following methods:
# 
# * load_image()
# * load_mask()
# * image_reference()

# %%
# Training dataset
data_folder = os.path.join(ROOT_DIR, "dataset")
dcm_folder = os.path.join(ROOT_DIR, "dataset_dicom")
data_list = []
for root, dirs, files in os.walk(data_folder, topdown=True):
    for name in dirs:
        if name.endswith("_json"):
            data_list.append(osp.join(root, name))
# %%
from sklearn.model_selection import train_test_split
data_list_train, data_list_test = train_test_split(data_list, test_size=0.3, random_state=42)
count = len(data_list)
count_train = len(data_list_train)
count_test = len(data_list_test)
# 修改为自己的网络输入大小
width = 512
height = 512
# %%
dataset_train = DrugDataset()
dataset_train.load_shapes(count_train, height, width, data_list_train)
dataset_train.prepare()

# Validation dataset
dataset_val = DrugDataset()
dataset_val.load_shapes(count_test, height, width, data_list_test)
dataset_val.prepare()
# %%
# Load and display random samples
# image_ids = np.random.choice(dataset_train.image_ids, 4)
# for image_id in image_ids:
#     image = dataset_train.load_image(image_id)
#     mask, class_ids = dataset_train.load_mask(image_id)
#     visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)
#  %%
# Create model in training mode
print('create model')
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

# Which weights to start with?
init_with = "imagenet"  # imagenet, coco, or last

if init_with == "imagenet":
    # model.load_weights(model.get_imagenet_weights(), by_name=True)
    model.load_weights(IMAGENET_MODEL_PATH, by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last(), by_name=True)
elif init_with == "none":
    pass

# ## Training
# 
# Train in two stages:
# 1. Only the heads. Here we're freezing all the backbone layers and training only the randomly initialized layers (i.e. the ones that we didn't use pre-trained weights from MS COCO). To train only the head layers, pass `layers='heads'` to the `train()` function.
# 
# 2. Fine-tune all layers. For this simple example it's not necessary, but we're including it to show the process. Simply pass `layers="all` to train all layers.

# %%:
# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
# model.train(dataset_train, dataset_val,
#             learning_rate=config.LEARNING_RATE,
#             epochs=100,
#             layers='heads')

# %%:
# Fine tune all layers
# Passing layers="all" trains all layers. You can also 
# pass a regular expression to select which layers to
# train by name pattern.
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=200,
            layers="all")

# %%
# Save weights
# Typically not needed because callbacks save after every epoch
# Uncomment to save manually
model_path = os.path.join(MODEL_DIR, "mask_rcnn_liver.h5")
model.keras_model.save_weights(model_path)


# ## Detection

# In[11]:

#
# class InferenceConfig(ShapesConfig):
#     GPU_COUNT = 1
#     IMAGES_PER_GPU = 1
#
#
# inference_config = InferenceConfig()
#
# # Recreate the model in inference mode
# model = modellib.MaskRCNN(mode="inference",
#                           config=inference_config,
#                           model_dir=MODEL_DIR)
#
# # Get path to saved weights
# # Either set a specific path or find last trained weights
# # model_path = os.path.join(MRCNN_DIR, ".h5 file name here")
# model_path = model.find_last()
#
# # Load trained weights
# print("Loading weights from ", model_path)
# model.load_weights(model_path, by_name=True)
#
# # In[ ]:
#
#
# # Test on a random image
# image_id = random.choice(dataset_val.image_ids)
# original_image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset_val, inference_config,
#                                                                                    image_id, use_mini_mask=False)
#
# log("original_image", original_image)
# log("image_meta", image_meta)
# log("gt_class_id", gt_class_id)
# log("gt_bbox", gt_bbox)
# log("gt_mask", gt_mask)
#
# visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
#                             dataset_train.class_names, figsize=(8, 8))
# results = model.detect([original_image], verbose=1)
#
# r = results[0]
# visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
#                             dataset_val.class_names, r['scores'], ax=get_ax())
#
# # ## Evaluation
