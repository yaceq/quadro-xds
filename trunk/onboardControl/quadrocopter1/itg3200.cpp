#include <Arduino.h>
#include <Wire.h>
#include "itg3200.h"

// ITG3200 Register Defines
#define WHO	0x00
#define	SMPL	0x15
#define DLPF	0x16
#define INT_C	0x17
#define INT_S	0x1A
#define	TMP_H	0x1B
#define	TMP_L	0x1C
#define	GX_H	0x1D
#define	GX_L	0x1E
#define	GY_H	0x1F
#define	GY_L	0x20
#define GZ_H	0x21
#define GZ_L	0x22
#define PWR_M	0x3E
#define GYRO_ADDRESS 0x68

float itg3200::bias_x = 0;
float itg3200::bias_y = 0;
float itg3200::bias_z = 0;
int itg3200::gx = 0;
int itg3200::gy = 0;
int itg3200::gz = 0;

void itg3200::init()
{
  delay(50);
  Serial.print("itg3200 gyro initialization...");
  Wire.beginTransmission(GYRO_ADDRESS);
  Wire.write(0x3E);
  Wire.write(0x80);  //send a reset to the device
  Wire.endTransmission(); //end transmission
  delay(50);

  Wire.beginTransmission(GYRO_ADDRESS);
  Wire.write((uint8_t)0x15);
  Wire.write((uint8_t)0x00);   //sample rate divider
  Wire.endTransmission(); //end transmission
  delay(50);

  Wire.beginTransmission(GYRO_ADDRESS);
  Wire.write((int)0x16);
  Wire.write((int)0x18); // ±2000 degrees/s (default value)
  Wire.endTransmission(); //end transmission
  delay(50);
  Serial.println("done.");
}


void itg3200::calibrate( int count, int dt )
{
  Serial.print("itg3200 calibration...");
  bias_x = bias_y = bias_z = 0;
  float ax=0, ay=0, az=0, x,y,z;
  for (int i=0; i<count/4; i++) { get_data(x,y,z); delay(dt); }
  for (int i=0; i<count; i++) {
    if ((i&7)==0) Serial.print(".");
    get_data(x,y,z);
    ax += x;
    ay += y;
    az += z;
    delay(dt);
  }
  bias_x = ax/count;
  bias_y = ay/count;
  bias_z = az/count;
  Serial.println("done.");
  Serial.print("...bias x : "); Serial.println(bias_x, 8);
  Serial.print("...bias y : "); Serial.println(bias_y, 8);
  Serial.print("...bias z : "); Serial.println(bias_z, 8);
}



void itg3200::get_raw_data ( int &x, int &y, int &z )
{
  x = y = z = 0;
  Wire.beginTransmission(GYRO_ADDRESS);
  Wire.write(0x1D);
  Wire.endTransmission();
  Wire.requestFrom(GYRO_ADDRESS, 6); 
  
  x =  Wire.read() << 8;
  x |= Wire.read();
  
  y =  Wire.read() << 8;
  y |= Wire.read();
  
  z =  Wire.read() << 8;
  z |= Wire.read();
  
  gx = x; gy = y; gz = z;
}


void itg3200::get_data ( float &x, float &y, float &z )
{
  int ix, iy, iz;
  get_raw_data(ix, iy, iz);
  x = ix / 14.375 - bias_x;
  y = iy / 14.375 - bias_y;
  z = iz / 14.375 - bias_z;
}



/*
static float bias_x = 0;
static float bias_y = 0; 
static float bias_z = 0;

void itg3200_init(void)
{
  delay(50);
  Serial.println("ITG3200 initialization...");
  Wire.beginTransmission(GYRO_ADDRESS);
  Wire.write(0x3E);
  Wire.write(0x80);  //send a reset to the device
  Wire.endTransmission(); //end transmission
  delay(50);

  Wire.beginTransmission(GYRO_ADDRESS);
  Wire.write((uint8_t)0x15);
  Wire.write((uint8_t)0x00);   //sample rate divider
  Wire.endTransmission(); //end transmission
  delay(50);

  Wire.beginTransmission(GYRO_ADDRESS);
  Wire.write((int)0x16);
  Wire.write((int)0x18); // ±2000 degrees/s (default value)
  Wire.endTransmission(); //end transmission
  delay(50);

  Serial.print("ITG3200 calibration...");
  delay(50);
  float ax=0, ay=0, az=0, x,y,z;
  for (int i=0; i<30; i++) { itg3200_getfb(x,y,z); delay(50); }
  for (int i=0; i<200; i++) {
    itg3200_getfb(x,y,z);
    ax += x;
    ay += y;
    az += z;
    delay(10);
  }
  bias_x = ax / 200.0f;
  bias_y = ay / 200.0f;
  bias_z = az / 200.0f;
  delay(50);
  Serial.println("OK");
  Serial.println(bias_x, DEC);
  Serial.println(bias_y, DEC);
  Serial.println(bias_z, DEC);
}


void itg3200_getfb( float &x, float &y, float &z )
{
  int ix,iy,iz;
  itg3200_get(ix, iy, iz);
  x = ix;  y = iy;  z = iz;
  x -= bias_x;
  y -= bias_y;
  z -= bias_z;
}

void itg3200_getdeg( float &x, float &y, float &z )
{
  itg3200_getfb(x,y,z);
  x /= 14.375;
  y /= 14.375;
  z /= 14.375;
}


void itg3200_get( int &x, int &y, int &z )
{
  x = y = z = 0;
  Wire.beginTransmission(GYRO_ADDRESS);
  Wire.write(0x1D);
  Wire.endTransmission();
  Wire.requestFrom(GYRO_ADDRESS, 6); 
  
  x =  Wire.read() << 8;
  x |= Wire.read();
  
  y =  Wire.read() << 8;
  y |= Wire.read();
  
  z =  Wire.read() << 8;
  z |= Wire.read();
}

*/
