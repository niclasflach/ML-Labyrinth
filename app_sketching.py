import math
import os
import time

import cv2
import gymnasium as gym
import numpy as np
import serial
from gymnasium import Env
from gymnasium.spaces import Box, Dict, Discrete
from stable_baselines3 import PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

# import keyboard
# Import os for file path management
# using python 3.11.6 because i didnt manage to install torch and stable-baselines3 with newer version
# pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
# pip install stable-baselines3[extra] protobuf==3.20.*


# Servo positions
SERVO_1_MIN = 120
SERVO_1_NOLL = 220
SERVO_1_MAX = 320
SERVO_2_MIN = 125
SERVO_2_NOLL = 225
SERVO_2_MAX = 325
SERVO_3_NOLL = 175
SERVO_3_MIN = 175
SERVO_3_MAX = 495
SERVO_4_NOLL = 215
SERVO_5_NOLL = 500
# SERIAL_PORT = "/dev/ttyUSB0" # När jag kör från Linux
SERIAL_PORT = "COM7"  # När jag kör från PCn
WAYPOINT_PRECISION = 5


def ero_dia(img):
    kernel = np.ones((5, 5), np.uint8)

    img_dilation = cv2.dilate(img, kernel, iterations=4)
    img_erosion = cv2.erode(img_dilation, kernel, iterations=4)

    return img_erosion


def callback(i, reward):
    # if i % 100 == 0:
    #    print("Iteration:", i, "Reward:", reward)
    pass


class LabyrintGame(Env):
    def __init__(self):
        super().__init__()

        self.observation_space = Dict(
            {
                "picture": Box(low=0, high=255, shape=(200, 200, 3), dtype=np.uint8),
                "position_x": Box(low=0, high=255, shape=(1,), dtype=np.uint8),
                "position_y": Box(low=0, high=255, shape=(1,), dtype=np.uint8),
                "visited": Box(low=0, high=255, shape=(200, 200, 1), dtype=np.uint8),
                "servoPosition1": Box(low=0, high=32, shape=(1,), dtype=np.uint8),
                "servoPosition2": Box(low=0, high=32, shape=(1,), dtype=np.uint8),
                "distanceToPoi": Box(low=0, high=232, shape=(1,), dtype=np.uint8),
                "nextPoi": Box(low=0, high=232, shape=(2,), dtype=np.uint8),
            }
        )

        self.pois = []
        self.current_pos_x = 0
        self.current_pos_y = 0
        self.action_space = Discrete(5)
        self.servo1 = SERVO_1_NOLL
        self.servo2 = SERVO_2_NOLL
        self.speed = 25
        self.undetected_frame = 0
        self.ball_cord = [0, 0]
        self.visited = np.zeros((200, 200, 1), np.uint8)

    def step(self, action):
        action_map = {0: "up", 1: "down", 2: "left", 3: "right", 4: "no_action"}
        if action != 4:
            self.tilt_board(action_map[action])
        done = False
        done = self.get_done()
        observation, test, new_pos, waypoint_reached = self.get_observation()
        reward = 0
        if not new_pos:
            reward -= 1
        if waypoint_reached:
            reward += 100
        info = {}
        if done:
            reward = -200
        truncated = False
        return observation, reward, done, truncated, info

    def tilt_board(self, action):
        if action == "up" and self.servo1 > SERVO_1_MIN:
            self.servo1 -= self.speed
        if action == "down" and self.servo1 < SERVO_1_MAX:
            self.servo1 += self.speed
        if action == "left" and self.servo2 > SERVO_2_MIN:
            self.servo2 -= self.speed
        if action == "right" and self.servo2 < SERVO_2_MAX:
            self.servo2 += self.speed
        self.changePos(
            self.servo1, self.servo2, SERVO_3_NOLL, SERVO_4_NOLL, SERVO_5_NOLL
        )
        return

    def changePos(self, a, b, c, d, e):
        command = (
            str(a) + "," + str(b) + "," + str(c) + "," + str(d) + "," + str(e) + "\n"
        )
        arduino.write(bytes(command, "utf-8"))
        _ = arduino.readline()
        return True

    def reset(self, seed=None):
        self.pois = []
        # Ladda in lista med Waypoints från fil
        with open("poi.txt") as f:
            for line in f:
                line = line.strip()
                tmp = line.split(",")
                try:
                    self.pois.append((int(tmp[0]), int(tmp[1])))
                    # result.append((eval(tmp[0]), eval(tmp[1])))
                except:
                    pass
        # plocka upp första i listan och sätta som point of interest
        self.poi = self.pois.pop(0)
        print(f"Point to reach{self.poi}")
        time.sleep(1)
        new_game = False
        self.servo1 = SERVO_1_NOLL
        self.servo2 = SERVO_2_NOLL
        self.changePos(
            self.servo1, self.servo2, SERVO_3_MIN, SERVO_4_NOLL, SERVO_5_NOLL
        )

        self.visited = np.zeros((200, 200, 1), np.uint8)
        print("Resetting")
        time.sleep(3)
        self.changePos(self.servo1 + 10, self.servo2, SERVO_3_MAX, 485, SERVO_5_NOLL)
        time.sleep(1.5)
        self.changePos(self.servo1 + 10, self.servo2, SERVO_3_MAX, 215, SERVO_5_NOLL)
        observation, _, _, _ = self.get_observation()
        self.undetected_frame = 0
        info = {}

        while not new_game:
            # print("väntar på att bollen rullar ner")
            _, _, _, _ = self.get_observation()
            if not self.get_done():
                # print("hittat boll")
                new_game = True
        return observation, info

    def render(self):
        """
        function to render output, however this is not used att the moment
        """
        _, camera_picture, _ = self.get_observation()
        cv2.imshow("Game", camera_picture)
        return True

    def get_observation(self):
        """
        function to get observation.
        return the state of the game in some sense.
        This function changes quite a lot at this moment.
        """
        ret, raw = cam.read()  # try to grab a picture from the camera
        new_pos = False
        waypoint_reached = False
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
        # Try to isolate the green
        hsv = cv2.cvtColor(raw, cv2.COLOR_BGR2HSV)  # convert image to hsv
        # masking for green color
        lower_red = np.array([56, 133, 24])
        upper_red = np.array([84, 255, 121])
        mask = cv2.inRange(hsv, lower_red, upper_red)

        resized = cv2.resize(raw[0:700, 300:1000], (200, 200))
        red_ball = ero_dia(
            mask
        )  # Erode and dialate the result to remove small dots etc
        red_ball = cv2.resize(red_ball[0:700, 300:1000], (200, 200))
        # try to detect ball in the resulting picture
        ball_position = cv2.HoughCircles(
            red_ball,
            cv2.HOUGH_GRADIENT,
            1.2,
            100,
            param1=100,
            param2=9,
            minRadius=2,
            maxRadius=100,
        )
        # what if ball is not detected
        if ball_position is None:
            self.undetected_frame += 1

        # and if the ball is detected
        if ball_position is not None:
            ball_position = np.uint16(np.around(ball_position))
            # print(len(ball_position))
            for i in ball_position[0, :]:
                # print(f'Oupptäckt i {self.undetected_frame} frames')
                self.current_pos_x = i[1]
                self.current_pos_y = i[0]
                self.undetected_frame = 0
                self.ball_cord = (i[0], i[1])
                try:
                    cv2.circle(resized, (i[0], i[1]), 3, (0, 0, 255), 4)
                    if self.visited[i[1], i[0]] == 0:
                        new_pos = True
                        # cv2.circle(self.visited, (i[0],i[1]), 1, 1, -1)
                        self.visited[i[1], i[0]] = 255
                        # print(f"Bollens positions: y{i[0]} y{i[1]}  ")
                        # print("Ny pixel")
                    if math.dist(self.ball_cord, self.poi) < WAYPOINT_PRECISION:
                        waypoint_reached = True
                        self.poi = self.pois.pop(0)
                        print(f"New point to reach:{self.poi}")

                except:
                    pass
                    # print(f"Failing coordinates:{i[0]} {i[1]}")
                    # print("kan inte rita cirkel")
            # print(f"distance is: {math.dist(self.ball_cord, self.poi)}")
        # channel = np.reshape(red_ball, (3,200,200))
        # print(self.ball_cord)
        # print(self.poi)
        resized = np.array(resized, dtype=np.uint8)
        observation = {
            "picture": resized,
            "position_x": self.current_pos_x,
            "position_y": self.current_pos_y,
            "visited": self.visited,
            "servoPosition1": self.servo1,
            "servoPosition2": self.servo2,
            "distanceToPoi": math.dist(self.ball_cord, self.poi),
            "nextPoi": self.poi,
        }
        return observation, red_ball, new_pos, waypoint_reached

    def get_done(self):
        done = False
        if self.undetected_frame > 30:
            done = True
        return done

    def close(self):
        cv2.destroyAllWindows()


cam = cv2.VideoCapture(0)
arduino = serial.Serial(port=SERIAL_PORT, baudrate=9600, timeout=0.1)
env = LabyrintGame()
CHECKPOINT_DIR = "./train/"
LOG_DIR = "./logs/"
env.reset()

debuggin = False

if debuggin:
    while True:
        obs, red_ball, new_pos = env.get_observation()
        if env.get_done():
            print("spel slut...")
            time.sleep(3)
            env.reset()
        if new_pos:
            print("ny position")
        # if frame is read correctly ret is True
        # show pictures for testing
        cv2.imshow("obs", obs["visited"])
        cv2.imshow("red", red_ball)
        if cv2.waitKey(1) == ord("q"):
            break

else:
    model_path = "./model_med_poi.zip"

    if os.path.exists(model_path):
        model = PPO.load(model_path, env)
    else:
        model = PPO(
            "MultiInputPolicy",
            env,
            batch_size=10000,
            tensorboard_log=LOG_DIR,
            verbose=1,
            n_steps=10000,
            learning_rate=0.009,
        )
    for i in range(300):
        model.learn(total_timesteps=20000, progress_bar=True)
        print("saving model....")
        model.save(model_path)
# , buffer_size=10000
