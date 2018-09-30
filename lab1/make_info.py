import os
import imutils
from PIL import Image

dataset_path = "./dataset/good/"
image_list = imutils.imlist(dataset_path)

f = open('./cars.info', 'w')

for image in image_list:
	im = Image.open(image)
	(width, height) = im.size
	f.write(image + ' 1 0 0 ' + str(width) + ' ' + str(height) + '\n')