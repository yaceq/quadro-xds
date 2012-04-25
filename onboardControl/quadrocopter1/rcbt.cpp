
#include <Arduino.h>

// NewSoftSerial Library, get here :
// http://arduiniana.org/NewSoftSerial/NewSoftSerial10c.zip
#include <NewSoftSerial.h>

#define MAX_PACKET_SIZE 16
#define RxD 6
#define TxD 7

byte in_data[MAX_PACKET_SIZE];
byte out_data[MAX_PACKET_SIZE];


void rcbt::init()
{
  pinMode(RxD, INPUT);
  pinMode(TxD, OUTPUT);

  blueToothSerial.begin(38400); //  set BluetoothBee BaudRate to default baud rate 38400
  blueToothSerial.print("\r\n+STWMOD=0\r\n"); // set the bluetooth work in slave mode
  blueToothSerial.print("\r\n+STNA=Quadrotor HG\r\n"); // set the bluetooth name as "SeeedBTSlave"
  blueToothSerial.print("\r\n+STOAUT=1\r\n"); // permit Paired device to connect me
  blueToothSerial.print("\r\n+STAUTO=0\r\n"); // auto-connection should be forbidden here
  delay(2000); // This delay is required.
  blueToothSerial.print("\r\n+INQ=1\r\n"); // make the slave bluetooth inquirable 
  Serial.println("The slave bluetooth is inquirable!");
  delay(2000); // This delay is required.
  blueToothSerial.flush();
}


void rcbt::send_data( byte *data, int sz )
{
  
}


void rcbt::recv_data( byte *data, int sz )
{
}





