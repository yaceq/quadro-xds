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


