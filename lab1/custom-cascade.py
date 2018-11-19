import argparse as ap
import cv2
import imutils 
import numpy as np
import os
import sys
from collections import Counter
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import *

clf, classes_names, stdSlr, k, voc = joblib.load("bow.pkl")

input_type = sys.argv[1]
input_path = sys.argv[2]
print(len(sys.argv))
if len(sys.argv) >= 4:
	if sys.argv[3] == '-save':
		input_save = True
		if len(sys.argv) >= 5:
			if sys.argv[4] == '-mark':
				input_mark = True
			else:
				input_mark = False
		else:
			input_mark = False
	else:
		input_save = False
else:
	input_save = False



captions = {
	'car': 'Автомобилей',
	'bus': 'Автобусов',
	'truck': 'Грузовиков'
}

CASCADE_FILE = './data/cascade.xml'

def processFile(path):
	image = cv2.imread(path)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	cascade = cv2.CascadeClassifier(CASCADE_FILE)

	# Фиксирование объектов и обёртывание их в прямоугольники
	rectangles = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=19, minSize=(30, 30))
	#rectangles = cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=10, minSize=(20, 20))

	# Используются sift дескрипторы
	sift = cv2.xfeatures2d.SIFT_create()
	# Массив всех дескрипторов
	des_list = []
	kpts_list = []

	# Для каждого прямоугольника находим ключевые точки
	# Вместе с дескриптором складываем в массив
	for (i, (x, y, w, h)) in enumerate(rectangles):
		#Выделяем прямоугольный регион
		print(x)
		print(x+w)
		print(y)
		print(y+h)
		roi = gray[x:x+w,y:y+h]
		kpts, des = sift.detectAndCompute(roi, None)
		kpts_list.append(kpts)
		des_list.append((roi, des))

	# Положить все дескрипторы вертикально в numpy массив
	descriptors = des_list[0][1]
	for roi, descriptor in des_list[0:]:
	    descriptors = np.vstack((descriptors, descriptor)) 

	test_features = np.zeros((len(rectangles), k), "float32")
	for i in range(len(rectangles)):
	    words, distance = vq(des_list[i][1],voc)
	    for w in words:
	        test_features[i][w] += 1

	# TF-IDF vectorization
	nbr_occurences = np.sum( (test_features > 0) * 1, axis = 0)
	idf = np.array(np.log((1.0*len(rectangles)+1) / (1.0*nbr_occurences + 1)), 'float32')

	# Масштабирование слов
	test_features = stdSlr.transform(test_features)

	# Прогнозирование
	predictions =  [classes_names[i] for i in clf.predict(test_features)]

	# Визуализация результатов
	cv2.namedWindow("Image", cv2.WINDOW_NORMAL)

	for (x, y, w, h), prediction, kp in zip(rectangles, predictions, kpts_list):
		cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 0, 255), 2)
		pt = (x, y + 20)
		cv2.putText(gray, prediction, pt ,cv2.FONT_HERSHEY_DUPLEX, 1, [255, 0, 0], 2)

	counter = Counter(predictions)
	occurrences = dict(counter);

	for key in occurrences:
	    print("{0} найдено {1}".format(captions[key], occurrences[key]))

	if input_save:
		if input_mark:
			start, ext = os.path.splitext(path)
			path = start + '_'
			if 'car' in occurrences:
				path += 'a' + str(occurrences['car'])
			if 'bus' in occurrences:
				path += 'b' + str(occurrences['bus'])
			if 'track' in occurrences:
				path += 't' + str(occurrences['track'])
			path += ext
		cv2.imwrite(path, gray)
	else:
		cv2.imshow("Image", gray)
		cv2.waitKey(0)

if input_type == '-file':
	processFile(input_path)
elif input_type == '-dir':
	names = os.listdir(input_path)
	for name in names:
		print(name)
		fullname = os.path.join(input_path, name)
		if os.path.isfile(fullname):
			processFile(fullname)


#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#roi = gray[x:w,y:h];
#cv2.putText(image, CASCADE_ITEM + " #{}".format(i + 1), (x, y - 10),
#	cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
#image = cv2.drawKeypoints(gray,kp,image,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Ссылки:
# http://docs.opencv.org/2.4/doc/tutorials/objdetect/cascade_classifier/cascade_classifier.html
# http://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Image_Object_Detection_Face_Detection_Haar_Cascade_Classifiers.php
# https://stackoverflow.com/questions/30857908/face-detection-using-cascade-classifier-in-opencv-python