#include <Wire.h>
 
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
 
void setup()
{
  Wire.begin();
  Gyro_Init();
  Serial.begin(9600);
}
 
void loop()
{
//  Serial.print("Gyrox: ");
 
 // Serial.println(ITG3200Read(TMP_H,TMP_L),DEC);
  Serial.print(ITG3200Read(GX_H,GX_L),DEC); Serial.print("  ");
  Serial.print(ITG3200Read(GY_H,GY_L),DEC); Serial.print("  ");
  Serial.print(ITG3200Read(GZ_H,GZ_L),DEC); Serial.print("  ");
  Serial.println("");
 
//  Serial.println(ITG3200Readbyte(WHO),HEX);    
//  Serial.println(ITG3200Readbyte(0x16),BIN);  
//  Serial.println(ITG3200Readbyte(0x15),BIN);  
 
//  Serial.println(ITG3200Readbyte(0x3E),BIN);    
 
//  Serial.println("*************");
 
 
	    delay(10);
}
 
 
char ITG3200Readbyte(unsigned char address)
{
   char data;
 
	  Wire.beginTransmission(GYRO_ADDRESS);
	  Wire.write((address));
	  Wire.endTransmission();
	  Wire.requestFrom(GYRO_ADDRESS,1);
	  if (Wire.available()>0)
	    {
	    data = Wire.read();
	    }
	    return data;
 
 
	   Wire.endTransmission();
}
 
char ITG3200Read(unsigned char addressh,unsigned char addressl)
{
   char data;
 
	  Wire.beginTransmission(GYRO_ADDRESS);
	  Wire.write((addressh));
	    Wire.endTransmission();
	  Wire.requestFrom(GYRO_ADDRESS,1);
	  if (Wire.available()>0)
	    {
	    data = Wire.read();
	    }
	  Wire.beginTransmission(GYRO_ADDRESS);
	    Wire.write((addressl));
	    Wire.endTransmission();
	    if (Wire.available()>0)
	    {
	    data |= Wire.read()<<8;
	    }
	    return data;
 
 
//	   Wire.endTransmission();
}
 
 
 
void Gyro_Init(void)
{
  Wire.beginTransmission(GYRO_ADDRESS);
  Wire.write(0x3E);
  Wire.write(0x80);  //send a reset to the device
  Wire.endTransmission(); //end transmission
 
 
  Wire.beginTransmission(GYRO_ADDRESS);
  Wire.write((uint8_t)0x15);
  Wire.write((uint8_t)0x00);   //sample rate divider
  Wire.endTransmission(); //end transmission
 
  Wire.beginTransmission(GYRO_ADDRESS);
  Wire.write((int)0x16);
  Wire.write((int)0x18); // ±2000 degrees/s (default value)
  Wire.endTransmission(); //end transmission
 
//  Wire.beginTransmission(GYRO_ADDRESS);
//  Wire.write(0x17);
//  Wire.write(0x05);   // enable send raw values
//  Wire.endTransmission(); //end transmission
 
//  Wire.beginTransmission(GYRO_ADDRESS);
//  Wire.write(0x3E);
//  Wire.write(0x00);
//  Wire.endTransmission(); //end transmission
}
