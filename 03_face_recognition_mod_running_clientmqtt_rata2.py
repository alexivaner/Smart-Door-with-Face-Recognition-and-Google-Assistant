''''
Real Time Face Recogition
    ==> Each face stored on dataset/ dir, should have a unique numeric integer ID as 1, 2, 3, etc                       
    ==> LBPH computed model (trained faces) should be on trainer/ dir
Based on original code by Anirban Kar: https://github.com/thecodacus/Face-Recognition    

Developed by Marcelo Rovai - MJRoBot.org @ 21Feb18  

'''
import time
#time.sleep(30)
import cv2
import numpy as np
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
from firebase import firebase


firebase = firebase.FirebaseApplication('https://aiy-voice-kit-217919.firebaseio.com/MainDoor')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('/home/pi/AIY-voice-kit-python/src/examples/face_recognition/FacialRecognition/trainer/trainer.yml')
cascadePath = "/home/pi/AIY-voice-kit-python/src/examples/face_recognition/FacialRecognition/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
language = 'en-US'

HOST = 'localhost'
PORT = 1883

#iniciate id counter
id = 0
wajah = []
doorauthorize=[]
indexofconfidence=[]
a = []
# names related to ids: example ==> Marcelo: id=1,  etc
names = firebase.get('/MainDoor/FaceData',None)
face_detector = cv2.CascadeClassifier('/home/pi/AIY-voice-kit-python/src/examples/face_recognition/FacialRecognition/haarcascade_frontalface_default.xml')
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


def on_connect(client, userdata, flags, rc):
    print("Connected to {0} with result code {1}".format(HOST, rc))
    client.subscribe("dev/anyHuman")
    client.subscribe("dev/NameCapture")
    client.subscribe("dev/NameData")
    client.subscribe("dev/FinishedTraining")

def on_message_nama(client, userdata, msg): 
    global names
    print("Name refreshed")
    names = firebase.get('/MainDoor/FaceData',None)
    print(names)

def on_message_training(client, userdata, msg): 
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('/home/pi/AIY-voice-kit-python/src/examples/face_recognition/FacialRecognition/trainer/trainer.yml')
    cascadePath = "/home/pi/AIY-voice-kit-python/src/examples/face_recognition/FacialRecognition/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath);
    print("Training Initiation Complete")
  

def on_message_anyHuman(client, userdata, msg):
    global names
    count=0
    wajah.clear()
    doorauthorize.clear()
    indexofconfidence.clear()
    a.clear()
    start = datetime.datetime.now()
    print("script execution stared at:", start)
    while msg.payload.decode() == '1':
            ret, img =cam.read()
            #img = cv2.flip(img, -1) # Flip vertically
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale( 
                gray,
                scaleFactor = 1.2,
                minNeighbors = 5,
                minSize = (int(minW), int(minH)),
            )
        
            for(x,y,w,h) in faces:

                #cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
                id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
                count+=1
                # Check if confidence is less them 100 ==> "0" is perfect match 
                if (confidence < 100):
                    if count == 1:
                        cv2.imwrite("/home/pi/AIY-voice-kit-python/src/examples/voice/guest.jpg", gray[y:y+h,x:x+w])
                    if count>5:
                        wajah.append(id)
                        doorauthorize.append (round(100 - confidence))
                        newconfidence = "  {0}%".format(round(100 - confidence))
                        nama = names[id]
                        print(nama,newconfidence)
                    else:
                        print("Initialize..",count)
                else:
                    nama = "unknown"
                    newconfidence = "  {0}%".format(round(100 - confidence))
                    print(nama,newconfidence)
            if count >= 9: # Take 9 face sample and stop video
                newid = np.bincount(wajah).argmax() #mencari wajah terbanyak yang muncul
                newnames = names[newid] #menampilkan nama wajah yang id nya paling banyak muncul
                indexconfidence=[i for i, n in enumerate(wajah) if n == newid]
                #mendapatkan index wajah dari newid(yang palinf banyak muncul)
                
                for x in indexconfidence:
                    a.append(doorauthorize[x]) 
                    #membuat array baru yang diambil hanya dari wajah yang paling banyak muncul
                
                average = sum(a)/len(a) #menghitung rerata confidence
                print("The face is belonged to:", newnames)
                print("Confidence average is:",average)

                if (average > 30):
                    mosquitto('dev/test','0')
                    mosquitto('dev/id',str(newnames))
                    mosquitto('dev/confidence',str(average))
                else:
                    mosquitto('dev/id','unknown')
                end = datetime.datetime.now()
                print("Script execution ended at:", end)
                total_time = end-start
                print("Script totally ran for :", total_time)
                print("Doneeeeee")
                break
               
    #if msg.payload.decode() == '0' :
        #mosquitto('dev/information',"No Human") 
        
        

def on_message_capture(client, userdata, msg):
    face_id = msg.payload.decode()
    count = 0
    mosquitto('dev/information',"Capturing..") 

    for count in range(36):

        ret, img = cam.read()
        #img = cv2.flip(img, -1) # flip video image vertically
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
            count += 1
            if(count>5):
               # Save the captured image into the datasets folder
               cv2.imwrite("/home/pi/AIY-voice-kit-python/src/examples/face_recognition/FacialRecognition/dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

    # Do a bit of cleanup
    mosquitto('dev/information',"Finished..") 



client = mqtt.Client()
client.on_connect = on_connect
client.message_callback_add("dev/anyHuman", on_message_anyHuman)
client.message_callback_add("dev/NameCapture", on_message_capture)
client.message_callback_add("dev/NameData", on_message_nama)
client.message_callback_add("dev/FinishedTraining", on_message_training)
client.username_pw_set(username="smartdoorlock",password="141097")
client.connect(HOST, PORT, 60)
client.loop_forever()







