import time
from collections import Counter
import sys


config_length = 1200                   #Defining led array
config_length_hex = (config_length*6)
config_brightness=-1
config_error=[]
config_error_time=(int(time.time())+10)


def console(frame):
    #Function keeps track of occurring errors, displaying them every 10 seconds..
        global config_error,config_error_time
        config_error.append(frame)
        if config_error_time<int(time.time()):
                counts = Counter(config_error)
                print(counts)
                config_error=[]
                config_error_time=(int(time.time())+10)

try:
        import serial
        port = "/dev/ttyACM0"                #Defining our serial port
        usart = serial.Serial(port,921600)
except:
        console('Serial module fail')


def push(frame='',repeat=False):            #Main function, send frame to the Teensy
    global config_length_hex,config_brightness
    while len(frame)<config_length_hex:     #We do need a full frame
        if(repeat and len(frame)>0):        #Repeat...
            frame += frame
        else:                   #or fill with zeros
            frame += '000000000000000000'
    if len(frame) % 2:                  #Hex frames shouldn't be odd-length
        frame += '0'

    frame = brightness(frame,20)
    frame_hex = '01'                #needed to announce start of frame

    #print frame_hex
    frame_hex += frame[:config_length_hex]
    frame_bytes = frame_hex.decode("hex")
    console("Frame pushed")

    try:
            usart.write(frame_bytes)
            #sys.stdout.write(frame_bytes)
            #filename = '/var/www/html/lumos/frame.txt'
            #with open(filename, "w") as f:
            #            f.write(frame_hex)

    except:
            console('Serial connection fail')


def clear():                        #Send an empty frame
    push('000000',1)


def brightness(frame,offset=100):
    offset = offset/100.00
    pixels=[]
    pixels=[int(frame[x:x+2],16) for x in range(0,len(frame),2)]
    #print(offset)
    pixels=[(n*offset) for n in pixels]
    #print(pixels)
    frame=''
    for i in pixels:
        i = int(i)
        if i<0:
            i=0
        if i>255:
            i=255
        i=hex(i)[2:]
        if len(i)<2:
            i = '0'+i
        frame=frame+i
    return frame
