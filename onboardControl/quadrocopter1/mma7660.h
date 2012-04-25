#include <Arduino.h>

class mma7660 {
  public:
    static void init      ( void );
    static void calibrate ( int count, int dt );
    static void get_data  ( float &x, float &y, float &z );
  protected:
    static void get_raw_data ( char &x, char &y, char &z );
    static void i2csend ( byte addr, byte value );  
    static float bias_x;
    static float bias_y;
    static float bias_z;
};


void mma7660_init(void);
void mma7660_get(char *x, char *y, char *z);
void mma7660_getf(float &x, float &y, float &z);
void mma7660_update();
