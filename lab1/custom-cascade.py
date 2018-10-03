import cv2

IMAGE_FILE1 = './test/test-1.pgm'
IMAGE_FILE = './dataset/test/n04487081/image_17.jpg'
IMAGE_FILE2 = './dataset/test/n04487081/image_0.jpg'
IMAGE_FILE3 = './dataset/test/n04487081/image_0.jpg'
IMAGE_FILE4 = './dataset/test/n04487081/image_0.jpg'
IMAGE_FILE5 = './dataset/test/n04487081/image_0.jpg'
CASCADE_FILE = './data/cascade.xml'
CASCADE_ITEM = 'Car'

image = cv2.imread(IMAGE_FILE)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cascade = cv2.CascadeClassifier(CASCADE_FILE)
# Фиксирование объектов и обёртывание их в прямоугольники
rectangles = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(50, 20))
for (i, (x, y, w, h)) in enumerate(rectangles):
	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
	cv2.putText(image, CASCADE_ITEM + " #{}".format(i + 1), (x, y - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

cv2.imshow(CASCADE_ITEM + "s", image)
cv2.waitKey(0)

# Ссылки:
# http://docs.opencv.org/2.4/doc/tutorials/objdetect/cascade_classifier/cascade_classifier.html
# http://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Image_Object_Detection_Face_Detection_Haar_Cascade_Classifiers.php
# https://stackoverflow.com/questions/30857908/face-detection-using-cascade-classifier-in-opencv-python
