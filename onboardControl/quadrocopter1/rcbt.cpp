#include <Arduino.h>
#include <SoftwareSerial.h>
#include "rcbt.h"

#define RxD 6
#define TxD 7

SoftwareSerial bt_serial(RxD,TxD);

int  rcbt::count = 0;
char rcbt::in_cmd[MAX_COMMAND_SIZE];


void rcbt::init()
{
  pinMode(RxD, INPUT);
  pinMode(TxD, OUTPUT);

  Serial.println("opening remote control...");

  bt_serial.begin(38400); //  set BluetoothBee BaudRate to default baud rate 38400
  bt_serial.print("\r\n+STWMOD=0\r\n"); // set the bluetooth work in slave mode
  bt_serial.print("\r\n+STNA=Quadrotor HG\r\n"); // set the bluetooth name as "SeeedBTSlave"
  bt_serial.print("\r\n+STOAUT=1\r\n"); // permit Paired device to connect me
  bt_serial.print("\r\n+STAUTO=0\r\n"); // auto-connection should be forbidden here
  bt_serial.print("\r\n+STPIN=0000\r\n "); // no PIN-code 
  delay(2000); // This delay is required.
  
  bt_serial.print("\r\n+INQ=1\r\n"); // make the slave bluetooth inquirable 
  Serial.println("the slave bluetooth is inquirable.");
  
  delay(2000); // This delay is required.
  bt_serial.flush();

  Serial.println("done.");
}


SoftwareSerial *rcbt::serial()
{
  return &bt_serial;
}


void rcbt::send_cmd( char *string )
{
  bt_serial.println(string);
}


void rcbt::btstate()
{
  
}


char *rcbt::recv_cmd()
{
  while (bt_serial.available()) {
    
    // get character :
    char b = (char)bt_serial.read();

    // skip '\r' character :
    // if (b=='\r') { continue; }

    // '\n' is a command end :
    if (b=='\n') {
      in_cmd[count] = '\0';
      count = 0;

      if (in_cmd[0]=='+') { };            // handle BT commands
      return in_cmd;  // no empty commands
    }

    in_cmd[count] = b;
    count++;

    if (count>=MAX_COMMAND_SIZE) {
      count = MAX_COMMAND_SIZE-1;
    }
    
  }
  return NULL;
}


/*-----------------------------------------------------------------------------
  Command receiver :
-----------------------------------------------------------------------------*/

int  cmd_recv::count;
char cmd_recv::in_cmd[MAX_COMMAND_SIZE]; 

char *cmd_recv::recv_cmd()
{
  while (Serial.available()) {
    
    // get character :
    char b = (char)Serial.read();

    // '\n' is a command end :
    if (b=='\n') {
      in_cmd[count] = '\0';
      count = 0;
      return in_cmd;
    }

    in_cmd[count] = b;
    count++;

    if (count>=MAX_COMMAND_SIZE) {
      count = MAX_COMMAND_SIZE-1;
    }
    
  }
  return NULL;
}








