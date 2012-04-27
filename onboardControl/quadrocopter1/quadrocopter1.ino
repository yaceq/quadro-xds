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
  itg3200::calibrate(100, 30);
  mma7660::init();
  mma7660::calibrate(100, 30);
}


int loop_count = 0;

float roll = 0;
float pitch;
float yaw = 0;
float throttle = 0;


void apply_control( float throttle, float roll, float pitch, float yaw, float factor )
{
  float throttle1 = constrain( throttle + throttle * factor * ( - roll - pitch + yaw ), 0, 1 );
  float throttle2 = constrain( throttle + throttle * factor * ( - roll + pitch - yaw ), 0, 1 );
  float throttle3 = constrain( throttle + throttle * factor * ( + roll + pitch + yaw ), 0, 1 );
  float throttle4 = constrain( throttle + throttle * factor * ( + roll - pitch - yaw ), 0, 1 );
  
  int pwm1 = (throttle1) * 170+10;
  int pwm2 = (throttle2) * 170+10;
  int pwm3 = (throttle3) * 170+10;
  int pwm4 = (throttle4) * 170+10;
  if (throttle<0.05) {
    srv1.write( 10 );    srv2.write( 10 );
    srv3.write( 10 );    srv4.write( 10 );
  } else {
    srv1.write( pwm1 );  srv2.write( pwm2 );
    srv3.write( pwm3 );  srv4.write( pwm4 );
  }
  char str[64];
  //sprintf(str, "%x %x %x %x", pwm1, pwm1, pwm3, pwm4);
  //Serial.println( str );
}


void loop()
{
  float a,b,c,x,y,z,g;
  itg3200::get_data(a,b,c);
  mma7660::get_data(x,y,z);
  
  int stb_roll  = ( -y/3 - a/40 );
  int stb_pitch = (  x/3 - b/40 );
  int stb_yaw   = c/40;
  
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
      t = (t << 8) / 0xFF;
      r = (r << 8) / 0xFF;
      p = (p << 8) / 0xFF;
      y = (y << 8) / 0xFF;
      throttle = t/127.0f;
      roll     = r/127.0f;
      pitch    = p/127.0f;
      yaw      = y/127.0f;
    }
  }
  
  loop_count++;
  if (loop_count>10) {
    loop_count=0;
    char str[64];
    sprintf(str, "X %d %d %d %d %d %d", itg3200::gx, itg3200::gy, itg3200::gz, mma7660::ax, mma7660::ay, mma7660::az);
    Serial.println(str);
  }
}
