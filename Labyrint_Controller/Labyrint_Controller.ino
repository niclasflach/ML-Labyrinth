// Simple code that takes 5 integers from the serial port and
// drives 5 servos accordingly.
// Servos are connected to and Adafruit PWM Servo Driver Shield on 
// a Arduino UNO


#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>


Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();
/
#define SERVOMIN  150 // This is the 'minimum' pulse length count (out of 4096)
#define SERVOMAX  600 // This is the 'maximum' pulse length count (out of 4096)
#define USMIN  600 // This is the rounded 'minimum' microsecond length based on the minimum pulse of 150
#define USMAX  2400 // This is the rounded 'maximum' microsecond length based on the maximum pulse of 600
#define SERVO_FREQ 50 // Analog servos run at ~50 Hz updates

// our servo # counter
uint8_t servonum = 0;
String command;

//Initial position of the servos
int servo_a = 300;
int servo_b = 300;
int servo_c = 300;
int servo_d = 300;
int servo_e = 300; 


void setup() {
  Serial.begin(9600);
  Serial.println("Labyrint Controller");

  pwm.begin();
  pwm.setOscillatorFrequency(27000000);
  pwm.setPWMFreq(SERVO_FREQ);  // Analog servos run at ~50 Hz updates

  delay(10);
}



void loop() {
   
  // Drive each servo one at a time using setPWM()
  //Serial.println(servonum);
  if(Serial.available()){
    //command = Serial.readStringUntil('\n');
    servo_a = Serial.parseInt();
    servo_b = Serial.parseInt();
    servo_c = Serial.parseInt();
    servo_d = Serial.parseInt();
    servo_e = Serial.parseInt();
    char r = Serial.read();
    if(r == '\n'){}
    // Serial.println(servo_a);
    // Serial.println(servo_b);
    // Serial.println(servo_c);
    // Serial.println(servo_d);
    // Serial.println(servo_e);
  }
  /
  pwm.setPWM(0,0, servo_a);
  pwm.setPWM(1,0, servo_b);
  pwm.setPWM(2,0, servo_c);
  pwm.setPWM(3,0, servo_d);
  pwm.setPWM(4,0, servo_e);
  


}
