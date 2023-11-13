
import cv2
import numpy as np




import time
import serial
import time
import keyboard
# Import os for file path management
import os


SERVO_1_MIN = 200
SERVO_1_NOLL=400
SERVO_1_MAX = 600
SERVO_2_MIN= 200
SERVO_2_NOLL=400
SERVO_2_MAX=600
SERVO_3_NOLL=400
SERVO_4_NOLL=400
SERVO_5_NOLL=400
 
def ero_dia(img):
    kernel = np.ones((5, 5), np.uint8) 
  
    # The first parameter is the original image, 
    # kernel is the matrix with which image is 
    # convolved and third parameter is the number 
    # of iterations, which will determine how much 
    # you want to erode/dilate a given image. 
    img_erosion = cv2.erode(img, kernel, iterations=1) 
    img_dilation = cv2.dilate(img_erosion, kernel, iterations=1) 
    
    return img_dilation


class LabyrintGame():
    def __init__(self):
        super().__init__()
        
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
        self.servo1 = SERVO_1_NOLL
        self.servo2 = SERVO_2_NOLL
        self.changePos(self.servo1,self.servo2,SERVO_3_NOLL,SERVO_4_NOLL,SERVO_5_NOLL)
        self.visited = np.zeros((200,200,1), np.uint8)
        print('Resetting')


        while not new_game:
            self.get_observation()
            #print(self.ball_cord)
            if self.ball_cord[0] > 130 and self.ball_cord[0]< 185 and self.ball_cord[1] > 95 and self.ball_cord[1]< 130:
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
        ball_position= cv2.HoughCircles(red_ball, cv2.HOUGH_GRADIENT, 1.2,100, param1=100, param2=10, minRadius=2,maxRadius=500)
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
                    cv2.circle(resized,(i[0],i[1]),1,(0,0,255),4)
                    if self.visited[i[0],i[1]] == 0:
                        new_pos = True
                        #cv2.circle(self.visited, (i[0],i[1]), 1, 1, -1)
                        self.visited[i[0],i[1]]= 255
                        cv2.imshow('visited', self.visited)
                        #print(f'Bollens positions: x{i[0]} y{i[1]} Servo position: servo1:{self.servo1} servo2:{self.servo2}' )
                        #print("Ny pixel")
                except:
                    print("kan inte rita cirkel")
        #channel = np.reshape(red_ball, (3,200,200))
        return resized, red_ball, new_pos
    
    def get_done(self):
        done = False
        if self.undetected_frame > 30:
            done = True
        return done
    
    def close(self):
        cv2.destroyAllWindows()


cam = cv2.VideoCapture(0)
arduino = serial.Serial(port="/dev/ttyUSB0", baudrate=9600, timeout=.1)
env = LabyrintGame()
env.reset()


while True:
    obs, red_ball,new_pos= env.get_observation()
    if env.get_done():
        print("spel slut...")
        time.sleep(3)
        env.reset()
    if new_pos:
        print("ny position")
    #if frame is read correctly ret is True
    #show pictures for testing
    cv2.imshow('obs', obs)
    cv2.imshow('red', red_ball)

    if cv2.waitKey(1) == ord('q'):
        break
    

