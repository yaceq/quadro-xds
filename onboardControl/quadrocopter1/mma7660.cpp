#include <Arduino.h>
#include <Wire.h>

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
 


void mma7660_send( uint8_t addr, uint8_t value )
{
  Wire.beginTransmission(MMA7660addr);
  Wire.write(addr);   
  Wire.write(value);
  Wire.endTransmission();
}


void mma7660_init(void)
{
  mma7660_send(MMA7660_MODE, (uint8_t)0x00);
  mma7660_send(MMA7660_SR,   (uint8_t)0x00);
  mma7660_send(MMA7660_MODE, (uint8_t)0x01);
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
  }
}


