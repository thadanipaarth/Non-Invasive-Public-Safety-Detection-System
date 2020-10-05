from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import os
from gender_detection_module import detect_gender
from mask_detection_module import detect_mask
def mask_image():
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", required=True)
	args = vars(ap.parse_args())
	
	################## Face detector model #################
	prototxtPath = "../Models/Face-Detection-Model/deploy.prototxt"
	weightsPath = "../Models/Face-Detection-Model/res10_300x300_ssd_iter_140000.caffemodel"
	net = cv2.dnn.readNet(prototxtPath, weightsPath)

	image = cv2.imread(args["image"])
	orig = image.copy()
	(h, w) = image.shape[:2]
	blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),(104.0, 177.0, 123.0))
	net.setInput(blob)
	detections = net.forward()

	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence > 0.4:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))
			face = image[startY:endY, startX:endX]
			label_mask,confidence_mask=detect_mask(face)
			label_gender,confidence_gender=detect_gender(face)
			color = (0, 255, 0) if label_mask == "Mask" else (0, 0, 255)
			text_mask = "{}: {:.2f}%".format(label_mask, confidence_mask * 100)
			text_gender="{}: {:.2f}%".format(label_gender, confidence_gender * 100)
			cv2.putText(image, text_mask, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
			cv2.putText(image,text_gender,(startX,startY-(startY-endY)+10),cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
			cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

	cv2.imwrite('Output.png', image) 
	
if __name__ == "__main__":
	mask_image()
