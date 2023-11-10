#Short script to test communication with Arduino via serial

import serial
import time
import keyboard


arduino = serial.Serial(port="/dev/ttyUSB0", baudrate=9600, timeout=.1)

a=400
b=400
c=400
d=400
e=400
speed = 5



def changePos(a,b,c,d,e):
    command = str(a) + ',' +str(b) + ',' +str(c) + ',' + str(d) + ',' + str(e) +'\n'
    arduino.write(bytes(command, 'utf-8'))

    data = arduino.readline()

    return data

while True:
    print(changePos(a,b,c,d,e))
    time.sleep(0.5)
    if keyboard.is_pressed('a'):
        a += speed
    if keyboard.is_pressed('z'):
        a -= speed
    if keyboard.is_pressed('s'):
        b += speed
    if keyboard.is_pressed('x'):
        b -= speed
    if keyboard.is_pressed('d'):
        c += speed
    if keyboard.is_pressed('c'):
        c -= speed
    if keyboard.is_pressed('f'):
        d += speed
    if keyboard.is_pressed('v'):
        d -= speed
    if keyboard.is_pressed('g'):
        e += speed
    if keyboard.is_pressed('b'):
        e -= speed
            
    


