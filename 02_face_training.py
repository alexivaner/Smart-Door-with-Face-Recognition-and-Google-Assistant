''''
Training Multiple Faces stored on a DataBase:
    ==> Each face should have a unique numeric integer ID as 1, 2, 3, etc                       
    ==> LBPH computed model will be saved on trainer/ directory. (if it does not exist, pls create one)
    ==> for using PIL, install pillow library with "pip install pillow"

Based on original code by Anirban Kar: https://github.com/thecodacus/Face-Recognition    

Developed by Marcelo Rovai - MJRoBot.org @ 21Feb18   

'''

import cv2
import numpy as np
from PIL import Image
import os

import logging
import platform
import subprocess
import sys

import datetime
import paho.mqtt.publish as publish
import paho.mqtt.client as mqtt
import traceback
import my_config

# Path for face image database
path = '/home/pi/AIY-voice-kit-python/src/examples/face_recognition/FacialRecognition/dataset'

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("/home/pi/AIY-voice-kit-python/src/examples/face_recognition/FacialRecognition/haarcascade_frontalface_default.xml");

def mosquitto(topic, message):
    """Send MQTT commands"""

    try:
        publish.single(topic, payload=message,
                    hostname=my_config.mqtt_host,
                    port=my_config.mqtt_port,
                    auth={'username':my_config.mqtt_username,
                    'password':my_config.mqtt_password})
        logging.info("MQTT Command received: " + topic + " " +
                    message)
    except Exception as e:
            logging.error("MQTT error: " + traceback.format_exc())

# Initialize number of try

# function to get the images and label data
def getImagesAndLabels(path):

    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []

    for imagePath in imagePaths:

        PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
        img_numpy = np.array(PIL_img,'uint8')

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)

    return faceSamples,ids

start = datetime.datetime.now()
print("script execution stared at:", start)
print ("Training faces. Wait ...")
faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

# Save the model into trainer/trainer.yml
recognizer.write('/home/pi/AIY-voice-kit-python/src/examples/face_recognition/FacialRecognition/trainer/trainer.yml') # recognizer.save() worked on Mac, but not on Pi

# Print the numer of faces trained and end program
mosquitto('dev/FinishedTraining',"1")
end = datetime.datetime.now()
total_time = end-start
print("\n {0} faces trained. Time:".format(len(np.unique(ids))), total_time)
