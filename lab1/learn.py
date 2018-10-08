import numpy as np
import cv2
import os
import imutils
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler

dataset_path = "./dataset/train"

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
sift  = cv2.xfeatures2d.SIFT_create()

# Список всех дескрипторов
des_list = []

for image_path in image_paths:
    im = cv2.imread(image_path)
    kpts, des = sift.detectAndCompute(im, None)
    des_list.append((image_path, des))

# Все дескрипторы складываются вертикально в numpy массив
descriptors = des_list[0][1]
for image_path, descriptor in des_list[1:]:
    descriptors = np.vstack((descriptors, descriptor))

# Используется метод k-средних
k = 100
voc, variance = kmeans(descriptors, k, 1)

# Вычисляется гистограмма
im_features = np.zeros((len(image_paths), k), "float32")
for i in range(len(image_paths)):
    words, distance = vq(des_list[i][1],voc)
    for w in words:
        im_features[i][w] += 1

# TF-IDF vectorization
nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

# Масштабирование слов
stdSlr = StandardScaler().fit(im_features)
im_features = stdSlr.transform(im_features)

# Linear SVM 
clf = LinearSVC()
clf.fit(im_features, np.array(image_classes))

# Сохранение SVM
joblib.dump((clf, training_classes, stdSlr, k, voc), "bow.pkl", compress=3)
