#include <Servo.h>
#include <SoftwareSerial.h>   //Software Serial Port

#define RxD 4
#define TxD 7
#define USE_BT
 
SoftwareSerial blueToothSerial(RxD,TxD);
Servo srv1;
Servo srv2;
Servo srv3;
Servo srv4;


void setup()
{
  Serial.begin(9600);
  Serial.println("ESC setting up...");
  srv1.attach(5);   srv1.write(10);
  srv2.attach(6);   srv2.write(10);
  srv3.attach(10);  srv3.write(10);
  srv4.attach(11);  srv4.write(10);
  Serial.println("Done. Connect the power.");
  
  setupBlueToothConnection();
}


void setupBlueToothConnection()
{
#ifndef USE_BT
  return;
#else
  Serial.println("BT setting up...");
  
  pinMode(RxD, INPUT);
  pinMode(TxD, OUTPUT);
  blueToothSerial.begin(38400); //Set BluetoothBee BaudRate to default baud rate 38400
  blueToothSerial.print("\r\n+STWMOD=0\r\n"); //set the bluetooth work in slave mode
  blueToothSerial.print("\r\n+STNA=SeeedBTSlave\r\n"); //set the bluetooth name as "SeeedBTSlave"

  blueToothSerial.print("\r\n+STOAUT=1\r\n"); // Permit Paired device to connect me
  blueToothSerial.print("\r\n+STAUTO=0\r\n"); // Auto-connection should be forbidden here
  delay(2000); // This delay is required.

  blueToothSerial.print("\r\n+INQ=1\r\n"); //make the slave bluetooth inquirable 
  Serial.println("The slave bluetooth is inquirable!");
  delay(2000); // This delay is required.

  blueToothSerial.flush();
  Serial.println("Done.");
#endif  
}


String recvStr = String("");


byte hex2byte ( byte a, byte b ) {
  byte h=0, l = 0;
  if (a>=0x30 && a<=0x39) h = a - 0x30;
  if (a>=0x41 && a<=0x46) h = a - 0x41 + 0x0A;
  if (b>=0x30 && b<=0x39) l = b - 0x30;
  if (b>=0x41 && b<=0x46) l = b - 0x41 + 0x0A;
  return ((h<<4) | l);
}

void HandleCommand()
{
  Serial.print("CMD");
  Serial.println(recvStr);
  if (recvStr.length()==9 && recvStr[0]=='X') {
    byte r1 = hex2byte( recvStr[1], recvStr[2] );
    byte r2 = hex2byte( recvStr[3], recvStr[4] );
    byte r3 = hex2byte( recvStr[5], recvStr[6] );
    byte r4 = hex2byte( recvStr[7], recvStr[8] );
    Serial.print("R:");
    Serial.print(r1, HEX);
    Serial.print(r2, HEX);
    Serial.print(r3, HEX);
    Serial.print(r4, HEX);
    Serial.println();
    srv1.write(r1);
    srv2.write(r2);
    srv3.write(r3);
    srv4.write(r4);
  }
}


void loop()
{
  if (blueToothSerial.available()) {
    byte b = blueToothSerial.read();
    if (b=='\n') {
      HandleCommand();
      recvStr = String("");
    } else {
      recvStr += String((char)b);
    }
  }
}
