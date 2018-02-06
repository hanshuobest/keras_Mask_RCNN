import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from config import Config
import utils
import model as modellib
import visualize
from model import log
import yaml
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

ROOT_DIR = os.getcwd()
# 保存产生的日志和已训练的权重文件
MODEL_DIR = os.path.join(ROOT_DIR , "logs")
# 权重文件
COCO_MODEL_PATH = os.path.join(ROOT_DIR , "mask_rcnn_coco.h5")


def get_ax(rows = 1 , cols = 1 , size = 8):
	_, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
	return ax

class ShapesConfig(Config):
	NAME = "shapes"
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1
	NUM_CLASSES = 1 + 3
	
	IMAGE_MIN_DIM = 800
	IMAGE_MAX_DIM = 1280
	
	RPN_ANCHOR_SCALES = (8 * 6 , 16 * 6 , 32 * 6 , 64 * 6 , 128 * 6)
	TRAIN_ROIS_PER_IMAGE = 32
	STEPS_PER_EPOCH = 100
	
	VALIDATION_STEPS = 5

config = ShapesConfig()
config.display()


# 自定义数据集类
class DrugDataset(utils.Dataset):
	def get_obj_index(self , i,age):
		'''
		得到该图中有多少个物体
		:param i:
		:param age:
		:return:
		'''
		n = np.max(image)
		return n
	
	def from_yaml_get_classes(self , image_id):
		'''
		解析labelme中得到yaml文件，从而得到mask每一层对应的实例标签
		:param image_id:
		:return:
		'''
		info = self.image_info[image_id]
		with open(info['yaml_path']) as f:
			temp = yaml.load(f.read())
			labels = temp['label_names']
			del labels[0]
		return labels
	
	def draw_mask(self , num_obj , mask , image):
		info = self.image_info[image_id]
		for index in range(num_obj):
			for i in range(info['width']):
				for j in range(info['height']):
					at_pixel = image.getpixel((i, j))
					if at_pixel == index + 1:
						mask[j, i, index] = 1
		return mask
		
	def load_shapes(self, count, height, width, img_floder, mask_floder, imglist, dataset_root_path):
		'''
		重新写load_shapes里面包含自己的类别
		:param count:
		:param height:
		:param width:
		:param img_floder:
		:param mask_floder:
		:param imglist:
		:param dataset_root_path:
		:return:
		'''
		# Add classes
		self.add_class("shapes", 1, "box")
		self.add_class("shapes", 2, "column")
		self.add_class("shapes", 3, "package")
		self.add_class("shapes", 4, "fruit")
		for i in range(count):
			filestr = imglist[i].split(".")[0]
			filestr = filestr.split("_")[1]
			mask_path = mask_floder + "/" + filestr + ".png"
			yaml_path = dataset_root_path + "total/rgb_" + filestr + "_json/info.yaml"
			self.add_image("shapes", image_id=i, path=img_floder + "/" + imglist[i],
			               width=width, height=height, mask_path=mask_path, yaml_path=yaml_path)
	
	def load_mask(self, image_id):
		'''
		重写
		:param image_id:
		:return:
		'''
		global iter_num
		info = self.image_info[image_id]
		count = 1  # number of object
		img = Image.open(info['mask_path'])
		num_obj = self.get_obj_index(img)
		mask = np.zeros([info['height'], info['width'], num_obj], dtype=np.uint8)
		mask = self.draw_mask(num_obj, mask, img)
		occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
		for i in range(count - 2, -1, -1):
			mask[:, :, i] = mask[:, :, i] * occlusion
			occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
		labels = []
		labels = self.from_yaml_get_class(image_id)
		labels_form = []
		for i in range(len(labels)):
			if labels[i].find("box") != -1:
				# print "box"
				labels_form.append("box")
			elif labels[i].find("column") != -1:
				# print "column"
				labels_form.append("column")
			elif labels[i].find("package") != -1:
				# print "package"
				labels_form.append("package")
			elif labels[i].find("fruit") != -1:
				# print "fruit"
				labels_form.append("fruit")
		class_ids = np.array([self.class_names.index(s) for s in labels_form])
		return mask, class_ids.astype(np.int32)
	
#基础设置
dataset_root_path="/home/yangjunfeng/workspace_lj/fg_dateset/"
img_floder = dataset_root_path+"rgb"
mask_floder = dataset_root_path+"mask"
#yaml_floder = dataset_root_path
imglist = os.listdir(img_floder)
count = len(imglist)
width = 1280
height = 800

#train与val数据集准备
dataset_train = DrugDataset()
dataset_train.load_shapes(count, height, width, img_floder, mask_floder, imglist,dataset_root_path)
dataset_train.prepare()

dataset_val = DrugDataset()
dataset_val.load_shapes(count, 800, 1280, img_floder, mask_floder, imglist,dataset_root_path)
dataset_val.prepare()


# Load and display random samples
image_ids = np.random.choice(dataset_train.image_ids, 4)
for image_id in image_ids:
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

print(MODEL_DIR)
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)
# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # 加载coco权重，但是跳过由于不同类别的层
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last()[1], by_name=True)

model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=1,
            layers='heads')

# 微调所有层
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE / 10,
            epochs=2,
            layers="all")

class InferenceConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = model.find_last()[1]

# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

# Test on a random image
image_id = random.choice(dataset_val.image_ids)
original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset_val, inference_config,
                           image_id, use_mini_mask=False)

log("original_image", original_image)
log("image_meta", image_meta)
log("gt_class_id", gt_bbox)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)

visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
                            dataset_train.class_names, figsize=(8, 8))

results = model.detect([original_image], verbose=1)

r = results[0]
visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                            dataset_val.class_names, r['scores'], ax=get_ax())

# Compute VOC-Style mAP @ IoU=0.5
# Running on 10 images. Increase for better accuracy.
image_ids = np.random.choice(dataset_val.image_ids, 10)
APs = []
for image_id in image_ids:
	# Load image and ground truth data
	image, image_meta, gt_class_id, gt_bbox, gt_mask = \
		modellib.load_image_gt(dataset_val, inference_config,
		                       image_id, use_mini_mask=False)
	molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
	# Run object detection
	results = model.detect([image], verbose=0)
	r = results[0]
	# Compute AP
	AP, precisions, recalls, overlaps = \
		utils.compute_ap(gt_bbox, gt_class_id,
		                 r["rois"], r["class_ids"], r["scores"])
	APs.append(AP)

print("mAP: ", np.mean(APs))
    
