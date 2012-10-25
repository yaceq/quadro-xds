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

void HandleCommand()
{
  if (recvStr[0]=='X') {
    srv1.write(recvStr[1]);
    srv2.write(recvStr[2]);
    srv3.write(recvStr[3]);
    srv4.write(recvStr[4]);
  }
}


void loop()
{
  while (1) {
    byte b = blueToothSerial.read();
    if (b=='\n') {
      HandleCommand();
      recvStr = String("");
    }
  }
  
  char recvChar;
  while(1){
    if(blueToothSerial.available()){//check if there's any data sent from the remote bluetooth shield
      recvChar = blueToothSerial.read();
      Serial.print(recvChar);
    }
    if(Serial.available()){//check if there's any data sent from the local serial terminal, you can add the other applications here
      recvChar  = Serial.read();
      blueToothSerial.print(recvChar);
    }
  }
  return;  
#ifdef USE_BT
#else
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
#endif  
}

