
import cv2
import numpy as np
import time
import serial
import time
# import keyboard
# Import os for file path management
# using python 3.11.6 because i didnt manage to install torch and stable-baselines3 with newer version
# pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
# pip install stable-baselines3[extra] protobuf==3.20.*
import os
from gym.spaces import Box, Discrete
from gym import Env


SERVO_1_MIN = 200
SERVO_1_NOLL=400
SERVO_1_MAX = 600
SERVO_2_MIN= 200
SERVO_2_NOLL=400
SERVO_2_MAX=600
SERVO_3_NOLL=175
SERVO_3_MIN = 175
SERVO_3_MAX = 495
SERVO_4_NOLL=150
SERVO_5_NOLL=400
#SERIAL_PORT = "/dev/ttyUSB0" # När jag kör från Linux
SERIAL_PORT = "COM5" # När jag kör från PCn

 
def ero_dia(img):
    kernel = np.ones((5, 5), np.uint8) 
  
    # The first parameter is the original image, 
    # kernel is the matrix with which image is 
    # convolved and third parameter is the number 
    # of iterations, which will determine how much 
    # you want to erode/dilate a given image. 
    img_dilation = cv2.dilate(img, kernel, iterations=2) 
    img_erosion = cv2.erode(img_dilation, kernel, iterations=2) 
    
    return img_erosion


class LabyrintGame(Env):
    def __init__(self):
        super().__init__()
        self.observation_space = Box(low=0, high=255, shape=(1,83,100), dtype=np.uint8)
        self.action_space = Discrete(4)
        self.servo1 = SERVO_1_NOLL
        self.servo2 = SERVO_2_NOLL
        self.speed = 15
        self.undetected_frame = 0
        self.ball_cord = [0,0,0]
        self.visited = np.zeros((200,200,1), np.uint8)
        

    def step(self, action):
        action_map = {
            0:'up',
            1:'down',
            2:'left',
            3:'right',
            4:'no_action'
        }
        if action !=4:
            #pydirectinput.press(action_map[action])
            self.tilt_board(action_map[action])
            pass
        done = self.get_done() 
        observation,_,new_pos = self.get_observation()
        reward = -1 
        if new_pos:
            reward += 10
        info = {}
        if done:
            reward = -20
        return observation, reward, done, info
    

    def tilt_board(self, action):
        if action == 'up' and self.servo1 > SERVO_1_MIN :
            self.servo1 -= self.speed
        if action == 'down' and self.servo1 < SERVO_1_MAX:
            self.servo1 += self.speed
        if action == 'left' and self.servo2 > SERVO_2_MIN:
            self.servo2 -= self.speed
        if action == 'right' and self.servo2 < SERVO_2_MAX:
            self.servo2 += self.speed
        self.changePos(self.servo1,self.servo2,SERVO_3_NOLL,SERVO_4_NOLL,SERVO_5_NOLL)


    def changePos(self,a,b,c,d,e):
        command = str(a) + ',' +str(b) + ',' +str(c) + ',' + str(d) + ',' + str(e) +'\n'
        arduino.write(bytes(command, 'utf-8'))
        _ = arduino.readline()
        return True


    def reset(self):
        time.sleep(1)
        new_game = False
        self.servo1 = SERVO_1_NOLL -50
        self.servo2 = SERVO_2_NOLL-80
        self.changePos(self.servo1,self.servo2,SERVO_3_MIN,SERVO_4_NOLL,SERVO_5_NOLL)
        self.visited = np.zeros((200,200,1), np.uint8)
        print('Resetting')
        time.sleep(2)
        self.changePos(self.servo1+10,self.servo2,SERVO_3_MIN,SERVO_4_NOLL+100,SERVO_5_NOLL)
        time.sleep(1)
        self.changePos(self.servo1+10,self.servo2,SERVO_3_MAX,SERVO_4_NOLL,SERVO_5_NOLL)
        print("lägger tillbaka bollen")
        time.sleep(1)
        self.changePos(self.servo1+10,self.servo2,SERVO_3_MIN,SERVO_4_NOLL,SERVO_5_NOLL)



        while not new_game:
            print("väntar på att bollen rullar ner")
            self.get_observation()
            #print(self.ball_cord)
            #if self.ball_cord[0] > 130 and self.ball_cord[0]< 185 and self.ball_cord[1] > 95 and self.ball_cord[1]< 130:
            if not self.get_done():
                print("hittat boll")
                new_game = True
        print("OMSTART")
        return True
    

    def render(self):
        cv2.imshow('Game', self.current_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.close()


    def get_observation(self):
        ret, raw = cam.read()
        new_pos = False
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
        #Try to isolate the red marble
        img_hsv=cv2.cvtColor(raw, cv2.COLOR_BGR2HSV)
        hMin = 0
        sMin = 175
        vMin = 20
        hMax = 10
        sMax = 255
        vMax = 255
        hMin2 = 170
        sMin2 = 175
        vMin2 = 20
        hMax2 = 180
        sMax2 = 255
        vMax2 = 255
        lower = np.array([hMin, sMin, vMin])
        upper = np.array([hMax, sMax, vMax])
        mask1 = cv2.inRange(img_hsv,lower,upper)
        lower2 = np.array([hMin2, sMin2, vMin2])
        upper2 = np.array([hMax2, sMax2, vMax2])
        mask2 = cv2.inRange(img_hsv,lower2,upper2)
        mask = mask1 + mask2
        red_ball_raw = cv2.bitwise_and(img_hsv,img_hsv, mask=mask)
        resized = cv2.resize(raw[0:700, 300:1000], (200,200))
        red_ball = ero_dia(mask)
        red_ball = cv2.resize(red_ball[0:700, 300:1000], (200,200))
        ball_position= cv2.HoughCircles(red_ball, cv2.HOUGH_GRADIENT, 1.2,100, param1=100, param2=10, minRadius=3,maxRadius=500)
        if ball_position is None:
            self.undetected_frame += 1

        if ball_position is not None:
            ball_position = np.uint16(np.around(ball_position))
            #print(len(ball_position))
            for i in ball_position[0,:]:
                #print(f'Oupptäckt i {self.undetected_frame} frames')
                self.undetected_frame = 0
                self.ball_cord = i
                try:
                    cv2.circle(resized,(i[0],i[1]),3,(0,0,255),4)
                    if self.visited[i[1],i[0]] == 0:
                        new_pos = True
                        #cv2.circle(self.visited, (i[0],i[1]), 1, 1, -1)
                        self.visited[i[1],i[0]]= 255
                        cv2.imshow('visited', self.visited)
                        #print(f'Bollens positions: x{i[0]} y{i[1]} Servo position: servo1:{self.servo1} servo2:{self.servo2}' )
                        #print("Ny pixel")
                except:
                    print("kan inte rita cirkel")
        #channel = np.reshape(red_ball, (3,200,200))
        # Testar att byta ut rezised mot self.visited
        return self.visited, red_ball, new_pos
    
    def get_done(self):
        done = False
        if self.undetected_frame > 30:
            done = True
        return done
    
    def close(self):
        cv2.destroyAllWindows()


cam = cv2.VideoCapture(0)
arduino = serial.Serial(port=SERIAL_PORT, baudrate=9600, timeout=.1)
env = LabyrintGame()
env.reset()



# while True:
#     obs, red_ball,new_pos= env.get_observation()
#     if env.get_done():
#         print("spel slut...")
#         time.sleep(3)
#         env.reset()
#     if new_pos:
#         print("ny position")
#     #if frame is read correctly ret is True
#     #show pictures for testing
#     cv2.imshow('obs', obs)
#     cv2.imshow('red', red_ball)

#     if cv2.waitKey(1) == ord('q'):
#         break
    

for episode in range(10): 
    obs = env.reset()
    done = False  
    total_reward   = 0
    while not done  : 
        obs, reward,  done, info =  env.step(env.action_space.sample())
        total_reward  += reward 
    print('Total Reward for episode {} is {}'.format(episode, total_reward))   