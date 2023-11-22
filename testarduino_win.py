#Short script to test communication with Arduino via serial

import serial
import time
import keyboard
import pygame



arduino = serial.Serial(port="COM4", baudrate=9600, timeout=.1)

a=400
b=360
c=400
d=400
e=400
speed = 5



def changePos(a,b,c,d,e):
    command = str(a) + ',' +str(b) + ',' +str(c) + ',' + str(d) + ',' + str(e) +'\n'
    arduino.write(bytes(command, 'utf-8'))

    data = arduino.readline()

    return data
pygame.init()


clock = pygame.time.Clock()
screen = pygame.display.set_mode((600, 400))
pygame.display.set_caption("Physics")
white = (255, 255, 255)

while True:
    changePos(a,b,c,d,e)
    time.sleep(0.1)
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.KEYDOWN:

            if event.key == pygame.K_DOWN:
                a += speed
                print(f"A:{a}")
            if event.key == pygame.K_UP:
                a -= speed
                print(f"A:{a}")
            if event.key == pygame.K_RIGHT:
                b += speed
                print(f"B:{b}")
            if event.key == pygame.K_LEFT:
                b -= speed
                print(f"B:{b}")
            if event.key == pygame.K_RCTRL:
                a=400
                b=360
                c=400
                d=400
                e=400            
    screen.fill(white)
    pygame.display.flip()


