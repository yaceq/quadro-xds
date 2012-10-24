#include <Servo.h>

Servo srv1;
Servo srv2;
Servo srv3;
Servo srv4;

#define RxD 4
#define TxD 7

void setup()
{
  Serial.begin(9600);
  Serial.println("ESC setting up...");
  srv1.attach(5);   srv1.write(10);
  srv2.attach(6);   srv2.write(10);
  srv3.attach(10);  srv3.write(10);
  srv4.attach(11);  srv4.write(10);
  Serial.println("Connect the power.");
}


void loop()
{
  if (Serial.available()>=4) {
    byte r1 = Serial.read();
    byte r2 = Serial.read();
    byte r3 = Serial.read();
    byte r4 = Serial.read();
    
    srv1.write(r1);
    srv2.write(r2);
    srv3.write(r3);
    srv4.write(r4);
  } 
}

