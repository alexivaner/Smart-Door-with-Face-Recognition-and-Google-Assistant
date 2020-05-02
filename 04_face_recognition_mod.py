''''
Real Time Face Recogition
    ==> Each face stored on dataset/ dir, should have a unique numeric integer ID as 1, 2, 3, etc                       
    ==> LBPH computed model (trained faces) should be on trainer/ dir
Based on original code by Anirban Kar: https://github.com/thecodacus/Face-Recognition    

Developed by Marcelo Rovai - MJRoBot.org @ 21Feb18  

'''

import cv2
import numpy as np
import os

import logging
import platform
import subprocess
import sys

import paho.mqtt.publish as publish
import traceback
import my_config
from gtts import gTTS 

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('/home/pi/AIY-voice-kit-python/src/examples/face_recognition/FacialRecognition/trainer/trainer.yml')
cascadePath = "/home/pi/AIY-voice-kit-python/src/examples/face_recognition/FacialRecognition/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
language = 'en-US'


#iniciate id counter
id = 0

# names related to ids: example ==> Marcelo: id=1,  etc
names = ['None', 'Ivan', 'Angela', 'Dannaezar', 'Z', 'W'] 

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height

# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

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
count = 0


for count in range(5):

    ret, img =cam.read()
    img = cv2.flip(img, -1) # Flip vertically

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )

    for(x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        # Check if confidence is less them 100 ==> "0" is perfect match 
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
            mosquitto('dev/test','0')
            mosquitto('dev/id',str(id))
            mosquitto('dev/confidence',str(confidence))
            print(id,confidence)
            myobj = gTTS(text='welcome'+id, lang=language, slow=False) 
            myobj.save("/home/pi/AIY-voice-kit-python/src/examples/voice/welcome.mp3") 
            mosquitto('dev/facedetected','1')
            count=5
            exit()
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
            mosquitto('dev/id',str(id))
            mosquitto('dev/confidence',str(confidence))
            print(id,confidence)
            count += 1

cam.release()
cv2.destroyAllWindows()





