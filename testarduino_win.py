# Short script to test communication with Arduino via serial

import serial
import time
import keyboard
import pygame


arduino = serial.Serial(port="COM7", baudrate=9600, timeout=0.1)

a = 308
b = 412
c = 400
d = 215
e = 400
speed = 2


def changePos(a, b, c, d, e):
    command = str(a) + "," + str(b) + "," + str(c) + "," + str(d) + "," + str(e) + "\n"
    arduino.write(bytes(command, "utf-8"))

    data = arduino.readline()

    return data


pygame.init()


clock = pygame.time.Clock()
screen = pygame.display.set_mode((600, 400))
pygame.display.set_caption("Physics")
white = (255, 255, 255)

while True:
    changePos(a, b, c, d, e)
    time.sleep(0.1)
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_DOWN:
                a += speed
                print(f"a:{a}")
            if event.key == pygame.K_UP:
                a -= speed
                print(f"a:{a}")
            if event.key == pygame.K_RIGHT:
                b += speed
                print(f"b:{b}")
            if event.key == pygame.K_LEFT:
                b -= speed
                print(f"b:{b}")
            if event.key == pygame.K_a:
                d = 485
                changePos(a, b, c, d, e)
                time.sleep(1.3)
                d = 215
                changePos(a, b, c, d, e)
            if event.key == pygame.K_z:
                e -= speed
            if event.key == pygame.K_RCTRL:
                a = 400
                b = 360
                c = 400
                d = 400
                e = 400
    screen.fill(white)
    pygame.display.flip()

