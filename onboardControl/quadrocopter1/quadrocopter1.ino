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
  
//  cmd_recv_bt::init();
  itg3200::init();
  itg3200::calibrate(50, 30);
  mma7660::init();
  mma7660::calibrate(50, 30);
}


void loop()
{
  float a,b,c,x,y,z,g;
  itg3200::get_data(a,b,c);
  mma7660::get_data(x,y,z);

  char *cmd = cmd_recv::recv_cmd();
  if (cmd) {
    int t=0,r=0,p=0,y=0;
    if (cmd[0]=='X') {
      sscanf(cmd, "X %x %x %x %x", &t, &r, &p, &y);
      char answ[32];
      sprintf(answ, "X %04X %04X %04X", itg3200::gx, itg3200::gy, itg3200::gz ); 
      Serial.println(answ);
    }
  }


}
