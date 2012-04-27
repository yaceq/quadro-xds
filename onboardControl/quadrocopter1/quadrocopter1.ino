#include <stdio.h>
#include <Wire.h>
#include <Servo.h>
#include <SoftwareSerial.h>

#include "mma7660.h"
#include "itg3200.h"
#include "rcbt.h"

Servo srv1;
Servo srv2;
Servo srv3;
Servo srv4;

void setup() 
{
  Wire.begin();
  Serial.begin(57600);

  srv1.attach(5);   srv1.write(10);
  srv2.attach(6);   srv2.write(10);
  srv3.attach(10);  srv3.write(10);
  srv4.attach(11);  srv4.write(10);
  
//  rcbt::init();
  itg3200::init();
  itg3200::calibrate(100, 20);
  mma7660::init();
  mma7660::calibrate(100, 20);
}


int loop_count = 0;

float roll = 0;
float pitch;
float yaw = 0;
float throttle = 0;


void apply_control( float throttle, float roll, float pitch, float yaw, float factor )
{
  float throttle1 = constrain( throttle + factor * ( - roll - pitch + yaw ), 0, 1 );
  float throttle2 = constrain( throttle + factor * ( - roll + pitch - yaw ), 0, 1 );
  float throttle3 = constrain( throttle + factor * ( + roll + pitch + yaw ), 0, 1 );
  float throttle4 = constrain( throttle + factor * ( + roll - pitch - yaw ), 0, 1 );
  
  srv1.write( sqrt(throttle1) * 180 );
  srv2.write( sqrt(throttle2) * 180 );
  srv3.write( sqrt(throttle3) * 180 );
  srv4.write( sqrt(throttle4) * 180 );
}


void loop()
{
  float a,b,c,x,y,z,g;
  itg3200::get_data(a,b,c);
  mma7660::get_data(x,y,z);
  
  int stb_roll  = ( -y/100 - a/100 );
  int stb_pitch = (  x/100 - b/100 );
  int stb_yaw   = c/100;
  
  apply_control( throttle, roll + stb_roll, pitch + stb_pitch, yaw + stb_yaw, 0.1f );
  
/*  Serial.print(a, 6);    Serial.print(" ");
  Serial.print(b, 6);    Serial.print(" ");
  Serial.print(c, 6);    Serial.print(" ");
  Serial.print(x, 6);    Serial.print(" ");
  Serial.print(y, 6);    Serial.print(" ");
  Serial.print(z, 6);    Serial.print(" ");
  Serial.println("");*/
  

  char *cmd = cmd_recv::recv_cmd();
  if (cmd) {
    int t=0,r=0,p=0,y=0;
    if (cmd[0]=='X') {
      sscanf(cmd, "X %x %x %x %x", &t, &r, &p, &y);
      throttle = t/127.0f;
      roll     = r/127.0f;
      pitch    = p/127.0f;
      yaw      = y/127.0f;
    }
  }
  
  loop_count++;
  if (loop_count>30) {
    loop_count=0;
    char str[64];
    sprintf(str, "X %x %x %x", itg3200::gx, itg3200::gy, itg3200::gz);
    Serial.println(str);
  }
}
