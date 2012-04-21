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
 
 
class Acceleration
{
 public:
 char x;
 char y;
 char z;
};
 
void sendByte( byte addr, byte value )
{
  Wire.beginTransmission( MMA7660addr);
  Wire.write(addr);   
  Wire.write(value);
  Wire.endTransmission();
}

 
void mma7660_init(void)
{
  Wire.begin();
  sendByte(MMA7660_MODE, (byte)0x00);
  sendByte(MMA7660_SR,   (byte)0x00);
  sendByte(MMA7660_MODE, (byte)0x01);
}
 
void setup()
{
  mma7660_init();        // join i2c bus (address optional for master)
  Serial.begin(9600);  // start serial for output
}


int itercount = 0;
 
void Ecom()
{
  unsigned char val[3];
  int count = 0;
  val[0] = val[1] = val[2] = 64;

  Wire.beginTransmission(MMA7660addr);
  Wire.write(byte(0x00));  // register to read
  Wire.endTransmission();
  Wire.requestFrom(MMA7660addr, 3);    // request 3 bytes from slave device 0x4c
 
  if (Wire.available()) {
    val[0] = Wire.read();
    val[1] = Wire.read();
    val[2] = Wire.read();
  }
 
  // transform the 7 bit signed number into an 8 bit signed number.
  Acceleration ret;
 
  ret.x = ((char)(val[0]<<2))/4;
  ret.y = ((char)(val[1]<<2))/4;
  ret.z = ((char)(val[2]<<2))/4;
  Serial.print("accel: ");
  Serial.print(itercount, DEC);
  Serial.print(" : ");
  Serial.print(ret.x, DEC);   // print the reading
  Serial.print(" ");
  Serial.print(ret.y, DEC);   // print the reading
  Serial.print(" ");
  Serial.print(ret.z, DEC);   // print the reading
  Serial.println("");
  itercount++;
}
 
char reading = 0;
 
void loop()
{
  Ecom();
  delay(20);
}
