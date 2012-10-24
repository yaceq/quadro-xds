#include <Servo.h>

Servo srv1;
Servo srv2;
Servo srv3;
Servo srv4;

void countdown(int time)
{
   for (int i=time; i>0; i-=1000) {
     Serial.print("countdown: ");
     Serial.println(i/1000);
     delay(1000);
   }
}


void setup()
{
  Serial.begin(9600);
  Serial.println("setup()");
  srv1.attach(5);   srv1.write(10);
  srv2.attach(6);   srv2.write(10);
  srv3.attach(10);  srv3.write(10);
  srv4.attach(11);  srv4.write(10);
  Serial.println("Connect the power...");
  do {
    delay(100);
  } while (!Serial.available());
  Serial.println("Done.");
}


void loop()
{
  for (int i=0; i<180; i+=1) {
    delay(30);
    srv2.write(i);
  }//*/
}
//*/
