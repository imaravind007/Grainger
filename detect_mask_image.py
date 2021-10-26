from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import os

argp = argparse.ArgumentParser()
argp.add_argument("-i", "--image", required=True,
	help="path to input image")
argp.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
argp.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")
argp.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
values = vars(argp.parse_args())


ip = os.path.sep.join([values["face"], "deploy.prototxt"])
wp = os.path.sep.join([values["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
nt = cv2.dnn.readNet(ip, wp)
model = load_model(values["model"])
img = cv2.imread(values["image"])
orig = img.copy()
(h, w) = img.shape[:2]
binary_data = cv2.dnn.blobFromImage(img, 1.0, (300, 300),
	(104.0, 177.0, 123.0))
nt.setInput(binary_data)
detections = nt.forward()

for i in range(0, detections.shape[2]):
	if detections[0, 0, i, 2] > values["confidence"]:
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(s_X, s_Y, e_X, e_Y) = box.astype("int")
		(s_X, s_Y) = (max(0, s_X), max(0, s_Y))
		(e_X, e_Y) = (min(w - 1, e_X), min(h - 1, e_Y))
		face_image = img[s_Y:e_Y, s_X:e_X]
		face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
		face_image = cv2.resize(face_image, (224, 224))
		face_image = img_to_array(face_image)
		face_image = preprocess_input(face_image)
		face_image = np.expand_dims(face_image, axis=0)
		(m, wm) = model.predict(face_image)[0]
		lb = "No Mask...please be safe" if m < wm else "Mask...Wow responsible"
		clr = (0, 255, 0) if lb == "Mask...Wow responsible" else (0, 0, 255)
		lb = "{}: {:.2f}%".format(lb, max(m, wm) * 100)
		cv2.putText(img, lb, (s_X, s_Y - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, clr, 2)
		cv2.rectangle(img, (s_X, s_Y), (e_X, e_Y), clr, 2)

cv2.imshow("Output", img)
cv2.waitKey(0)