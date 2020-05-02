import paho.mqtt.client as mqtt
import os
import tempfile
import time
import traceback
from gtts import gTTS 


HOST = 'localhost'
PORT = 1883



def on_connect(client, userdata, flags, rc):
    print("Connected to {0} with result code {1}".format(HOST, rc))
    client.subscribe("dev/id")

def on_message_anyid(client, userdata, msg):
     if msg.payload.decode() == 'unknown':
        print("Hotword unknown!")
        os.system("mpg321 -g 40 /home/pi/AIY-voice-kit-python/src/examples/voice/unknown.mp3 /dev/null")
     else :
        id = msg.payload.decode()
        myobj = gTTS(text='welcome'+id, lang=language, slow=False) 
        myobj.save("/home/pi/AIY-voice-kit-python/src/examples/voice/welcome.mp3")
        print(id)
        os.system("mpg321 -g 40 /home/pi/AIY-voice-kit-python/src/examples/voice/welcome.mp3 /dev/null")

client = mqtt.Client()
client.on_connect = on_connect
client.message_callback_add("dev/id", on_message_anyid)
client.username_pw_set(username="smartdoorlock",password="141097")
client.connect(HOST, PORT, 60)
client.loop_forever()