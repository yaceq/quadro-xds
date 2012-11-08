// Gyroscope ITG3205 
#define GYRO 0x68 // Defined sensor address, AD0 is connected to GND port, sensor address is a binary number 11101000 (please refer to your interface board schematics)
#define G_SMPLRT_DIV 0x15  //Sampling rate register address
#define G_DLPF_FS 0x16     //Detection sensitivity and low-pass filter set
#define G_INT_CFG 0x17     //Configuration register
#define G_PWR_MGM 0x3E     //Sensor data register starting address, including the temperature of its 3-axis angular velocity

#define G_TO_READ 8        //each of x, y, z-axis of two bytes, plus another two byte temperature

// Gyro error correction offset 
int16 g_offx = 0;
int16 g_offy = 0;
int16 g_offz = 0;

////////////////////////////////////////////////////////////////////////////////////
//函数原型:  void initGyro(void)             	     
//参数说明:  无                                        
//返回值:    无                                                               
//说明:      初始化ITG3205陀螺仪
///////////////////////////////////////////////////////////////////////////////////
void initGyro(void)
{
  /*****************************************
   * ITG 3200
   * Power management settings：
   * Clock select = internal oscillator
   * No reset, no sleep mode
   * Standby mode
   * Sampling rate = 125Hz
   * Parameter is + / - 2000度/秒
   * Low-pass filter=5HZ
   * Without interruption
   ******************************************/
  writeTo(GYRO, G_PWR_MGM, 0x00);
  writeTo(GYRO, G_SMPLRT_DIV, 0x07); // EB, 50, 80, 7F, DE, 23, 20, FF
  writeTo(GYRO, G_DLPF_FS, 0x1E); // +/- 2000 dgrs/sec, 1KHz, 1E, 19
  writeTo(GYRO, G_INT_CFG, 0x00);
}

////////////////////////////////////////////////////////////////////////////////////
//Function prototype    : void getGyroscopeData(int16 * result)           	     
//Parameter Description : * result : Gyro data pointer                                      
//The return value      : No                                                               
//Explain               : Read ITG3205 gyro raw data
///////////////////////////////////////////////////////////////////////////////////
void getGyroscopeData(int16 * result)
{
  /**************************************
   * 陀螺仪ITG- 3200的I2C
   * 寄存器：
   * temp MSB = 1B, temp LSB = 1C
   * x axis MSB = 1D, x axis LSB = 1E
   * y axis MSB = 1F, y axis LSB = 20
   * z axis MSB = 21, z axis LSB = 22
   *************************************/

  uint8 regAddress = 0x1B;
  int16 temp, x, y, z;
  uint8 buff[G_TO_READ];

  readFrom(GYRO, regAddress, G_TO_READ, buff); //读取陀螺仪ITG3200的数据

  result[0] = (((int16)buff[2] << 8) | buff[3]) + g_offx;
  result[1] = (((int16)buff[4] << 8) | buff[5]) + g_offy;
  result[2] = (((int16)buff[6] << 8) | buff[7]) + g_offz;
  result[3] = ((int16)buff[0] << 8) | buff[1]; // 温度
}
