
# %%
import os
import os.path as osp
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
MRCNN_DIR = os.path.abspath("../../")
ROOT_DIR = osp.abspath("../../../")
sys.path.append(MRCNN_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
import cv2
from mrcnn.model import log
import pydicom
# %%
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
config = tf.ConfigProto()  
config.gpu_options.allow_growth=True
config.allow_soft_placement=True
sess = tf.Session(config=config)
KTF.set_session(sess)
# %%
# 修改为自己的识别类别
# class_names = ['BG', 'liver-delay', 'liver-portal']
# class_names = ['liver-delay', 'liver-portal']
class_names = ['BG', 'liver_0','liver_1','liver_2','liver_3','liver_4']
width=512
height=512

class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "liver"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 5 + 1  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    IMAGE_RESIZE_MODE = "none"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Use smaller anchors because our image and objects are small
    # RPN_ANCHOR_SCALES = (32, 64, 128, 256)  # anchor side in pixels

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
        "mrcnn_mask_dice_loss": 1.
    }

    # ROIs kept after non-maximum suppression (training and inference)
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 100

class InferenceConfig(ShapesConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # NUM_CLASSES = 1 + 2 # background + 3 class

# %%
config = InferenceConfig()
config.display()

# %%
# Directory to save logs and trained model
MODEL_DIR = os.path.join(MRCNN_DIR, "logs")

# Directory of images to run detection on
# dataset_root_path = os.path.join(MRCNN_DIR, "..", "dataset")
# data_folder = dataset_root_path
# data_list_temp = os.listdir(dataset_root_path)
# img_list = []
data_folder = os.path.join(ROOT_DIR, "dataset")
dcm_folder = os.path.join(ROOT_DIR, "dataset_dicom")
data_list = []
for root, dirs, files in os.walk(data_folder, topdown=True):
    for name in dirs:
        if name.endswith("_json"):
            data_list.append(osp.join(root, name))
dcm_list = []

# for data in data_list_temp:
#     if os.path.isdir(os.path.join(data_folder, data)):
#         img_list.append(os.path.join(data_folder, data, "img.png"))
# %%
# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", config=config , model_dir=MODEL_DIR)

# %%
# Local path to trained weights file
MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_liver.h5")
model.load_weights(MODEL_PATH, by_name=True)
# %%
def load_image(path):
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
    if ds:
        return ds, image
    else:
        return image

def find_dcm(data_path):
    dcm_parent_path = osp.dirname(data_path).replace("dataset", "dataset_dicom")
    dcm_right_name = osp.basename(data_path).replace("_json", ".dcm").split("-")[2]
    assert osp.isdir(dcm_parent_path)
    dcm_list_temp = os.listdir(dcm_parent_path)
    for x in dcm_list_temp:
        if x.find(dcm_right_name) != -1:
            dcm_name = x
            break
    dcm_path = osp.join(dcm_parent_path, dcm_name)
    return dcm_path

def mask_ascend_dim(mask):
    dim = np.max(mask)
    shape = (width, height, dim)
    new_mask = np.zeros(shape, dtype = int)
    for k in range(1, dim+1):
        for i in range(height):
            for j in range(width):
                if mask[i][j]==k:
                    new_mask[i][j][k-1]=1
    return new_mask.astype(np.bool)

def load_mask(data_path):
    mask_path = os.path.join(data_path, "label.png")
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = mask_ascend_dim(mask)
    return mask

def caculate_dice(y_true, y_pred, smooth = 0.000001):
    y_true_flatten = y_true.flatten().astype(np.bool)
    y_pred_flatten = y_pred.flatten().astype(np.bool)
    return (2. * np.sum(y_true_flatten * y_pred_flatten)) / (np.sum(y_true_flatten) + np.sum(y_pred_flatten) + smooth)

def pixel2area(ds):
    return ds.PixelSpacing[0] * ds.PixelSpacing[1]
# %%
# data_list_test = np.random.choice(data_list, 10)

# data_list_test = data_list

from sklearn.model_selection import train_test_split
data_list_train, data_list_test = train_test_split(data_list, test_size=0.3, random_state=42)

n = len(data_list_test)
test_res = [] #[(id,dice,hit),...]
err_list = []
# %%
import pandas as pd
df = pd.read_excel(io = osp.join(ROOT_DIR,"dataset","pathology.xls"), header = 0)
# %%
# a = []
for i, data_path in enumerate(data_list_test):
    dcm_path = find_dcm(data_path)
    patient_id = osp.basename(osp.dirname(osp.dirname(data_path)))
    patient_id_num = int(patient_id[1:])
    stage_id = osp.basename(osp.dirname(data_path))
    img_id = osp.basename(data_path).split("-")[-1].split('_')[0]
    main_id = patient_id + '_' + stage_id +'_' + img_id
    if main_id == "a92_p_00022":
        continue
    # if main_id not in ["a97_p_00021", "a17_d_00018", "a02_p_00016"]:
    #     continue
    # if main_id != "a92_p_00022":
    #     continue
    ds, img = load_image(dcm_path)
    
    img=cv2.resize(img,(width,height))

    # Run detection
    results = model.detect([img], verbose=0)

    # Visualize results
    r = results[0]
    mask_true = load_mask(data_path)
    mask_pred = r['masks']
    # a.append(mask_pred)

    class_name_true = "liver_"+str(df.values[patient_id_num-1,1])
    
    # 针对该项目，每个样本有且只有一个目标(a92_p_00022有两个)
    # 没有检测出目标
    if mask_pred.shape[2] == 0:
        print("error found nothing:{}:{}/{}".format(main_id, i+1, n))
        err_list.append((main_id, 'found nothing'))
        continue
    # 检测出了多个目标，取出对应一层的mask
    elif mask_true.shape[2] <= mask_pred.shape[2]:
        class_ids_pred = list(r['class_ids'])
        class_id_true = class_names.index(class_name_true)
        class_scores = list(r['class_scores']) 
        masks_scores = list(r['masks_scores'])
        scores = list(r['scores'])

        try:
            matched_channel = class_ids_pred.index(class_id_true)
        except ValueError:
            print("error found wrong:{}:{}/{}".format(main_id, i+1, n))
            err_list.append((main_id, 'found wrong'))
            continue
        mask_pred = mask_pred[:,:,matched_channel]
        class_score = class_scores[matched_channel]
        class_name_pred = class_name_true
        masks_score = np.squeeze(masks_scores[matched_channel])
        score = np.squeeze(scores[matched_channel])

    # pixel_num_true = np.sum(mask_true)
    # pixel_num_pred = np.sum(mask_pred)
    # pixel_coef = pixel2area(ds)
    # area_true = pixel_num_true * pixel_coef
    # area_pred = pixel_num_pred * pixel_coef

    

    try:
        dice = caculate_dice(mask_true, mask_pred)
        print("{}: {}/{}, dice: {}, class_score: {}/{}, mask_score: {}, scores: {}".format(main_id, i+1, n, dice, class_name_pred, class_score, masks_score, score))
        test_res.append((main_id, dice, class_score, masks_score, score))
    except ValueError:
        print("ERROR",main_id)
        # exit()
    # visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'],
    #                             figsize=(8,8), show_bbox=True, save=False)
# %%
test_res.sort(key = lambda x:x[1])
ids = [x[0] for x in test_res]
dices = [x[1] for x in test_res]
class_scores = [x[2] for x in test_res]
masks_scores = [x[3] for x in test_res]
scores = [x[4] for x in test_res]
for x in test_res:
    print("{}: {:.4f}, {:.4f}, {}, {}".format(x[0], x[1], x[2], x[3], x[4]))
if err_list!=[]:
    print("error count: {}".format(len(err_list)))
    print("error samples: {}".format(err_list))
dice_max = max(test_res, key = lambda x:x[1])
dice_min = min(test_res, key = lambda x:x[1])
dice_mean = np.sum(dices)/n
class_score_mean = np.sum(class_scores)/n
mask_score_mean = np.sum(masks_scores)/n
score_mean = np.sum(scores)/n
print("dice_max: {}, dice_min: {}, dice_mean: {}, class_score_mean: {}, mask_score_mean: {}, score_mean: {}".format(dice_max, dice_min, dice_mean, class_score_mean, mask_score_mean, score_mean))
# %%
with open("./test_res.log","w+") as f:
    for x in err_list:
        f.write("err:"+str(x)+"\n")
    f.write("dice_mean:"+str(dice_mean)+"\n")
    f.write("class_score_mean:"+str(class_score_mean)+"\n")
    f.write("mask_score_mean:"+str(mask_score_mean)+"\n")
    f.write("score_mean:"+str(score_mean)+"\n")
    for x in test_res:
        f.write(str(x[0])+":"+str(x[1])+","+str(x[2])+","+str(x[3])+","+str(x[4])+"\n")
# %%
plt.figure()
plt.plot(range(len(dices)), dices)
plt.plot(range(len(scores)), scores)
plt.grid(True)
plt.savefig(fname = "result.png", format = "png" ,dpi=500, bbox_inches = 'tight')
# %%