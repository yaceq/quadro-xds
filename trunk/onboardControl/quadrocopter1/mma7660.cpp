#include <Arduino.h>
#include <Wire.h>
#include "mma7660.h"

#define MMA7660addr   0x4c
#define MMA7660_X     0x00
#define MMA7660_Y     0x01
#define MMA7660_Z     0x02
#define MMA7660_TILT  0x03
#define MMA7660_SRST  0x04
#define MMA7660_SPCNT 0x05
#define MMA7660_INTSU 0x06
#define MMA7660_MODE  0x07
#define MMA7660_SR    0x08
#define MMA7660_PDET  0x09
#define MMA7660_PD    0x0A
 

int count = 0;
int bufx[8], last_x;
int bufy[8], last_y;
int bufz[8], last_z;
static float bias_x = 0;
static float bias_y = 0;
static float gravity = 0;

void mma7660_send( uint8_t addr, uint8_t value )
{
  Wire.beginTransmission(MMA7660addr);
  Wire.write(addr);   
  Wire.write(value);
  Wire.endTransmission();
}


void mma7660_init(void)
{
  Serial.println("MMA7660 initialization...");
  mma7660_send(MMA7660_MODE, (uint8_t)0x00);
  mma7660_send(MMA7660_SR,   (uint8_t)0x00);
  mma7660_send(MMA7660_MODE, (uint8_t)0x01);
  
  Serial.print("MMA7660 calibration...");
  delay(50);
  float ax=0, ay=0, az=0, x,y,z;

  for (int i=0; i<30; i++)  {   mma7660_getf(x,y,z);    delay(50);  }
  for (int i=0; i<100; i++) {
    mma7660_getf(x,y,z);
    ax += x;
    ay += y;
    az += z;
    delay(30);
  }
  bias_x  = ax / 100.0f;
  bias_y  = ay / 100.0f;
  gravity = az / 100.0f;
  delay(50);
  Serial.println("OK");
  Serial.println(bias_x, DEC);
  Serial.println(bias_y, DEC);
  Serial.println(gravity, DEC);
}
 
void mma7660_getf(float &x, float &y, float &z)
{
  mma7660_update();
  /*x = y = z = 0;
  for (int i=0; i<8; i++) {
    x += bufx[i];
    y += bufy[i];
    z += bufz[i];
  }
  x /= 8;      y /= 8;      z /= 8;
  //*/
  x = last_x;  y = last_y;  z = last_z;
  x -= bias_x;
  y -= bias_y;
}


void mma7660_update()
{
  char x,y,z;
  mma7660_get(&x,&y,&z);
  last_x = x;
  last_y = y;
  last_z = z;
}

void mma7660_get(char *x, char *y, char *z)
{
  Wire.beginTransmission(MMA7660addr);
  Wire.write(uint8_t(0x00));  // register to read
  Wire.endTransmission();
  Wire.requestFrom(MMA7660addr, 3);    // request 3 bytes from slave device 0x4c
 
  if (Wire.available()) {
    *x = Wire.read();
    *y = Wire.read();
    *z = Wire.read();
    *x = ((char)((*x)<<2))/4;
    *y = ((char)((*y)<<2))/4;
    *z = ((char)((*z)<<2))/4;
    bufx[count&0x7] = *x;
    bufy[count&0x7] = *y;
    bufz[count&0x7] = *z;
    count++;
  }
}


