#include <Arduino.h>

class itg3200 {
  public:
    static void init       ( void );
    static void calibrate  ( int count, int dt );
    static void get_data   ( float &x, float &y, float &z );
  protected:
    static void get_raw_data ( int &x, int &y, int &z );
    static float bias_x;
    static float bias_y;
    static float bias_z;
  };

void itg3200_init(void);
void itg3200_get( int &x, int &y, int &z );
void itg3200_getfb( float &x, float &y, float &z );
void itg3200_getdeg( float &x, float &y, float &z );

