
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

from google.assistant.library.event import EventType

from aiy.assistant import auth_helpers
from aiy.assistant.library import Assistant
from aiy.board import Board, Led
from aiy.voice import tts

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX
def power_off_pi():
	tts.say('Good bye!')
	subprocess.call('sudo shutdown now', shell=True)


def reboot_pi():
	tts.say('See you in a bit!')
	subprocess.call('sudo reboot', shell=True)


def say_ip():
	ip_address = subprocess.check_output("hostname -I | cut -d' ' -f1", shell=True)
	tts.say('My IP address is %s' % ip_address.decode('utf-8'))
	
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
         
def door_lock():
	mosquitto('dev/test', '1')
	tts.say("The door already locked", lang='en-GB', volume=25, pitch=70)


def door_unlock():
	mosquitto('dev/test', '0')
	tts.say("Welcome Ivan,The door already unlocked", lang='en-GB', volume=25, pitch=70)	 

def process_event(assistant, led, event):
	logging.info(event)
	if event.type == EventType.ON_START_FINISHED:
		led.state = Led.BEACON_DARK  # Ready.
		print('Say "OK, Google" then speak, or press Ctrl+C to quit...')
	elif event.type == EventType.ON_CONVERSATION_TURN_STARTED:
		led.state = Led.ON  # Listening.
	elif event.type == EventType.ON_RECOGNIZING_SPEECH_FINISHED and event.args:
		print('You said:', event.args['text'])
		text = event.args['text'].lower()
		if text == 'power off':
			assistant.stop_conversation()
			power_off_pi()
		elif text == 'reboot':
			assistant.stop_conversation()
			reboot_pi()
		elif text == 'ip address':
			assistant.stop_conversation()
			say_ip()
		elif text == 'lock the door':
			assistant.stop_conversation()
			door_lock()
		elif text == 'unlock the door':
			assistant.stop_conversation()
			door_unlock()
	elif event.type == EventType.ON_END_OF_UTTERANCE:
		led.state = Led.PULSE_QUICK  # Thinking.
	elif (event.type == EventType.ON_CONVERSATION_TURN_FINISHED
			or event.type == EventType.ON_CONVERSATION_TURN_TIMEOUT
			or event.type == EventType.ON_NO_RESPONSE):
		led.state = Led.BEACON_DARK  # Ready.
	elif event.type == EventType.ON_ASSISTANT_ERROR and event.args and event.args['is_fatal']:
		sys.exit(1)


def main():
	logging.basicConfig(level=logging.INFO)

	credentials = auth_helpers.get_assistant_credentials()
	with Board() as board, Assistant(credentials) as assistant:
		for event in assistant.start():
			process_event(assistant, board.led, event)


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

	while True:

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
			
			cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

			id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

			# Check if confidence is less them 100 ==> "0" is perfect match 
			if (confidence < 100):
				id = names[id]
				mosquitto('dev/test', '0')
				confidence = "  {0}%".format(round(100 - confidence))
			else:
				id = "unknown"
				mosquitto('dev/test', '1') #doorlock
				confidence = "  {0}%".format(round(100 - confidence))
        
			cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
			mosquitto('dev/id',str(id))
			cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
    
		cv2.imshow('camera',img) 

		k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
		if k == 27:
			break

	# Do a bit of cleanup
	print("\n [INFO] Exiting Program and cleanup stuff")
	cam.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
    main()