''''
Capture multiple Faces from multiple users to be stored on a DataBase (dataset directory)
    ==> Faces will be stored on a directory: dataset/ (if does not exist, pls create one)
    ==> Each face will have a unique numeric integer ID as 1, 2, 3, etc                       

Based on original code by Anirban Kar: https://github.com/thecodacus/Face-Recognition    

Developed by Marcelo Rovai - MJRoBot.org @ 21Feb18
Modified by Ivan Surya H - PCU @ 19 Feb 2019    

'''

import cv2
import os
from sys import argv

cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height



face_detector = cv2.CascadeClassifier('/home/pi/AIY-voice-kit-python/src/examples/face_recognition/FacialRecognition/haarcascade_frontalface_default.xml')

# For each person, enter one numeric face id
script, face_id = argv

# Initialize individual sampling face count
count = 0


for count in range(36):

    ret, img = cam.read()
    #img = cv2.flip(img, -1) # flip video image vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    count2= 0
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1
        if count <=36 and count>5:
            # Save the captured image into the datasets folder
            count2 += 1
            cv2.imwrite("/home/pi/AIY-voice-kit-python/src/examples/face_recognition/FacialRecognition/dataset/User." + str(face_id) + '.' + str(count2) + ".jpg", gray[y:y+h,x:x+w])

# Do a bit of cleanup
print("Capture Finished")
cam.release()
cv2.destroyAllWindows()



