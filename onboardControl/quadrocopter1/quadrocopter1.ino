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

void loop()
{
  if (Serial.available()) {
    throttle1 = map( Serial.read(), 0, 200, 10, 180 );
    throttle2 = throttle1;
    throttle3 = throttle1;
    throttle4 = throttle1;
    
    int roll  = map( Serial.read(), 0,200, -100,100);
    int pitch = map( Serial.read(), 0,200, -100,100);
    int yaw   = map( Serial.read(), 0,200, -100,100);
    
    throttle1 -= roll;
    throttle2 -= roll;
    throttle3 += roll;
    throttle4 += roll;
    
    throttle1 -= pitch;
    throttle2 += pitch;
    throttle3 += pitch;
    throttle4 -= pitch;
    
    throttle1 += yaw;
    throttle2 -= yaw;
    throttle3 += yaw;
    throttle4 -= yaw;
    
    Serial.write('q');
  }
  
  
  
  /*srv1.write( throttle1 );
  srv2.write( throttle2 );
  srv3.write( throttle3 );
  srv4.write( throttle4 );*/
  
  delay(10);
}
