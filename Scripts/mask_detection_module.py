from keras.preprocessing.image import img_to_array
from keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import cv2
import os

############## Mask detection model path #################
model_path='../Models/Mask-Detection/Mask_Detection.model'
mask_detection=load_model(model_path)

def detect_mask(face_crop):

	face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
	face_crop = cv2.resize(face_crop, (224, 224))
	face_crop = img_to_array(face_crop)
	face_crop = preprocess_input(face_crop)
	face_crop = np.expand_dims(face_crop, axis=0)

	(mask, withoutMask) = mask_detection.predict(face_crop)[0]
	label = "Mask" if mask > withoutMask else "No Mask"

	return label,(mask)