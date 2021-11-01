from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os

class mask:
	def __init__(self):
		parser = argparse.ArgumentParser()
		parser.add_argument("-f", "--face", type=str, default="face_detector",
							help="path to face detector model directory")
		parser.add_argument("-m", "--model", type=str, default="mask_detector.model",
							help="path to trained face mask detector model")
		parser.add_argument("-c", "--confidence", type=float, default=0.5,
							help="minimum probability to filter weak detections")
		self.args = vars(parser.parse_args())
		prototexttPath = os.path.sep.join([self.args["face"], "deploy.prototxt"])
		wp = os.path.sep.join([self.args["face"], "res10_300x300_ssd_iter_140000.caffemodel"])
		self.faceNet = cv2.dnn.readNet(prototexttPath, wp)
		self.maskNet = load_model(self.args["model"])
		print("[INFO] starting video stream...")
		self.vs = VideoStream(src=0).start()
		self.sleep = 20
	def mask_detect(self,frameval, faceval, maskval):
		(height, width) = frameval.shape[:2]
		binary_data = cv2.dnn.blobFromImage(frameval, 1.0, (300, 300), (104.0, 177.0, 123.0))
		faceval.setInput(binary_data)
		detections = faceval.forward()
		f = []
		locations = []
		predict = []
		for i in range(0, detections.shape[2]):

			if detections[0, 0, i, 2] > self.args["confidence"]:
				box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
				(s_X, s_Y, e_X, e_Y) = box.astype("int")

				(s_X, s_Y) = (max(0, s_X), max(0, s_Y))
				(e_X, e_Y) = (min(width - 1, e_X), min(height - 1, e_Y))

				face_image = frameval[s_Y:e_Y, s_X:e_X]
				face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
				face_image = cv2.resize(face_image, (224, 224))
				face_image = img_to_array(face_image)
				face_image = preprocess_input(face_image)

				f.append(face_image)
				locations.append((s_X, s_Y, e_X, e_Y))

		if len(f) > 0:
			f = np.array(f, dtype="float32")
			predict = maskval.predict(f, batch_size=32)

		return (locations, predict)
	def detect(self):
		wearingMask=False
		while True:
			self.sleep-=1
			frameval = self.vs.read()
			frameval = imutils.resize(frameval, width=400)
			(locations, predict) = self.mask_detect(frameval, self.faceNet, self.maskNet)

			for (box, predict) in zip(locations, predict):
				(s_X, s_Y, e_X, e_Y) = box
				(m, wm) = predict
				lb = "Mask" if m > wm else "No Mask"
				wearingMask = True if m > wm else False
				clr = (0, 255, 0) if lb == "Mask" else (0, 0, 255)
				lb = "{}: {:.2f}%".format(lb, max(m, wm) * 100)
				cv2.putText(frameval, lb, (s_X, s_Y - 10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.45, clr, 2)
				cv2.rectangle(frameval, (s_X, s_Y), (e_X, e_Y), clr, 2)



			cv2.imshow("Frame", frameval)
			key = cv2.waitKey(1) & 0xFF
			if self.sleep == 0:
				break
			if key == ord("q"):
				break


		cv2.destroyAllWindows()
		self.vs.stop()
		print(wearingMask)
		return wearingMask


mask().detect()