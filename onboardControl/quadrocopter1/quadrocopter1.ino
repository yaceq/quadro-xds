#include <Wire.h>
#include <Servo.h>

#include "mma7660.h"
#include "itg3200.h"

Servo srv1;
Servo srv2;
Servo srv3;
Servo srv4;

void setup() 
{
  Wire.begin();
  Serial.begin(9600);
  Serial.println("Quadrocoper");
  
  mma7660_init();
  itg3200_init();
  
  srv1.attach(5);   srv1.write(10);
  srv2.attach(6);   srv2.write(10);
  srv3.attach(10);  srv3.write(10);
  srv4.attach(11);  srv4.write(10);
  
  while(0) {
    char x,y,z;
    float a,b,c;
    mma7660_get(&x, &y, &z);
    itg3200_getdeg(a, b, c);
    Serial.print(x, DEC);    Serial.print(" ");
    Serial.print(y, DEC);    Serial.print(" ");
    Serial.print(z, DEC);    Serial.print(" ");
    Serial.print(a, DEC);    Serial.print(" ");
    Serial.print(b, DEC);    Serial.print(" ");
    Serial.print(c, DEC);    Serial.print(" ");
    Serial.println("");
    delay(20);
  }
  
  Serial.println("DONE");
}


int throttle1 = 10;
int throttle2 = 10;
int throttle3 = 10;
int throttle4 = 10;

int throttle0 = 0;

float abs_roll = 0;
float abs_pitch = 0;
float abs_yaw = 0;

int stab_r = 1;
int stab_g = 1;

void stabilize(float dt)
{
  float x,y,z;
  float a,b,c;
  mma7660_getf(x, y, z);
  itg3200_getdeg(a, b, c);
  
  abs_roll  += a*dt;
  abs_pitch += b*dt;
  abs_yaw   += c*dt;
  
/*  Serial.print(x, DEC);  Serial.print(" ");
  Serial.print(y, DEC);  Serial.print(" ");
  Serial.print(z, DEC);  Serial.print(" ");
  Serial.print(abs_roll,  DEC);  Serial.print(" ");
  Serial.print(abs_pitch, DEC);  Serial.print(" ");
  Serial.print(abs_yaw,   DEC);  Serial.print(" ");*/
  Serial.println("");//*/
  
/*  if (abs(x)<1) { x = 0; }
  if (abs(y)<1) { y = 0; }*/
  
  
  int roll  = ( -y/stab_g - a/stab_r ) * throttle0/4;
  int pitch = (  x/stab_g - b/stab_r ) * throttle0/4;
  int yaw   =        c/stab_r  * throttle0/4;

  /*Serial.print(roll,  DEC);  Serial.print(" ");
  Serial.print(pitch, DEC);  Serial.print(" ");
  Serial.print(yaw,   DEC);  Serial.print(" ");
  Serial.println();//*/
  
  control(roll, pitch, yaw);
}


void control(int roll, int pitch, int yaw)
{
    throttle1 -= roll;    throttle2 -= roll;
    throttle3 += roll;    throttle4 += roll;
    
    throttle1 -= pitch;    throttle2 += pitch;
    throttle3 += pitch;    throttle4 -= pitch;
    
    throttle1 += yaw;    throttle2 -= yaw;
    throttle3 += yaw;    throttle4 -= yaw;
}

void loop()
{
  if (Serial.available()>=6) {
    throttle0 = map( Serial.read(), 0, 200, 10, 180 );
    throttle1 = throttle0;
    throttle2 = throttle0;
    throttle3 = throttle0;
    throttle4 = throttle0;
    
    int roll  = map( Serial.read(), 0,200, -throttle0/4, throttle0/4);
    int pitch = map( Serial.read(), 0,200, -throttle0/4, throttle0/4);
    int yaw   = map( Serial.read(), 0,200, -throttle0/4, throttle0/4);
    
    stab_r = Serial.read();
    stab_g = Serial.read();
    
    control(roll, pitch, yaw);

    stabilize(0.02);
    
   /* Serial.write(throttle1);
    Serial.write(throttle2);
    Serial.write(throttle3);
    Serial.write(throttle4);*/
  }
  
  throttle1 = constrain( throttle1, 10, 180 );
  throttle2 = constrain( throttle2, 10, 180 );
  throttle3 = constrain( throttle3, 10, 180 );
  throttle4 = constrain( throttle4, 10, 180 );
  
/*  Serial.print(throttle1, DEC);  Serial.print(" ");
  Serial.print(throttle2, DEC);  Serial.print(" ");
  Serial.print(throttle3, DEC);  Serial.print(" ");
  Serial.print(throttle4, DEC);  Serial.print(" ");
  Serial.println();*/
    
  srv1.write( throttle1 );
  srv2.write( throttle2 );
  srv3.write( throttle3 );
  srv4.write( throttle4 );
  
  delay(10);
}
