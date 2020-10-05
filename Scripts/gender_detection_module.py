from keras.preprocessing.image import img_to_array
from keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import cv2
import os

########## Enter the path of the model ##############
model_path ='../Models/Gender-Detection-Model/Gender_Detection.model'
gender_detection= load_model(model_path)

def detect_gender(face_crop):
	face_crop = cv2.resize(face_crop, (96,96))
	face_crop = face_crop.astype("float")/255.0
	face_crop = img_to_array(face_crop)
	face_crop = np.expand_dims(face_crop, axis=0)
	conf = gender_detection.predict(face_crop)[0]
	label = "Male" if conf[0] > conf[1] else "Female"
	return label,max(conf[0],conf[1])