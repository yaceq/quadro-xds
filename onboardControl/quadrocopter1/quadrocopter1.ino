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
  itg3200::init();
  itg3200::calibrate(100, 20);
  mma7660::init();
  mma7660::calibrate(100, 20);
}

void loop()
{
  float a,b,c,x,y,z;
  itg3200::get_data(a,b,c);
  mma7660::get_data(x,y,z);
  
  Serial.print(a, 6);    Serial.print(" ");
  Serial.print(b, 6);    Serial.print(" ");
  Serial.print(c, 6);    Serial.print(" ");
  Serial.print(x, 6);    Serial.print(" ");
  Serial.print(y, 6);    Serial.print(" ");
  Serial.print(z, 6);    Serial.print(" ");
  Serial.println("");
  
  delay(50);
}
