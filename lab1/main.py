import cv2 as cv
import numpy
import os
import imutils

print(cv.__version__)

dataset_path = "/dataset"

# Список классов, которые будем тренировать. (формируется по списку директорий)
training_classes = os.listdir(dataset_path)

image_paths = []
image_classes = []
class_id = 0
for training_class in training_classes:
	dir = os.path.join(dataset_path, training_class)
	class_path = imutils.imlist(dir)
	image_paths += class_path
	image_classes += [class_id]*len(class_path)
	class_id += 1

# Используются SIFT дескрипторы
fea_det = cv2.FeatureDetector_create("SIFT")
des_ext = cv2.DescriptorExtractor_create("SIFT")

# Список всех дескрипторов
des_list = []