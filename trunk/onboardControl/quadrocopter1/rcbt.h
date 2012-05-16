#include <Arduino.h>

const int MAX_COMMAND_SIZE =  64;

class rcbt {
  public:
    static void init ();
    static void send_cmd  ( char *cmd );
    static char *recv_cmd ();
    static char *recv_cmd_com ();
    static SoftwareSerial *serial();
  protected:
    static int count;
    static char in_cmd[MAX_COMMAND_SIZE]; 
    static void btstate();
  };

class cmd_recv {
  public:
    static void init();
    static char *recv_cmd();
  protected:
    static int count;
    static char in_cmd[MAX_COMMAND_SIZE]; 
};


class cmd_recv_bt {
  public:
    static void init();
    static char *recv_cmd();
  protected:
    static int count;
    static char in_cmd[MAX_COMMAND_SIZE]; 
};
