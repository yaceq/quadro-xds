// 陀螺仪 ITG3205 
#define GYRO 0x68 // 定义传感器地址,将AD0连接到GND口，传感器地址为二进制数11101000 (请参考你接口板的原理图)
#define G_SMPLRT_DIV 0x15  //采样率寄存器地址
#define G_DLPF_FS 0x16     //检测灵敏度及其低通滤波器设置
#define G_INT_CFG 0x17     //配置寄存器
#define G_PWR_MGM 0x3E     //传感器数据寄存器起始地址，包括温度及其3轴角速度

#define G_TO_READ 8 // x,y,z 每个轴2个字节，另外再加上2个字节的温度

// 陀螺仪误差修正的偏移量 
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
   * 电源管理设定：
   * 时钟选择 =内部振荡器
   * 无复位, 无睡眠模式
   * 无待机模式
   * 采样率 = 125Hz
   * 参数为+ / - 2000度/秒
   * 低通滤波器=5HZ
   * 没有中断
   ******************************************/
  writeTo(GYRO, G_PWR_MGM, 0x00);
  writeTo(GYRO, G_SMPLRT_DIV, 0x07); // EB, 50, 80, 7F, DE, 23, 20, FF
  writeTo(GYRO, G_DLPF_FS, 0x1E); // +/- 2000 dgrs/sec, 1KHz, 1E, 19
  writeTo(GYRO, G_INT_CFG, 0x00);
}

////////////////////////////////////////////////////////////////////////////////////
//函数原型:  void getGyroscopeData(int16 * result)           	     
//参数说明:  * result : 陀螺仪数据指针                                      
//返回值:    无                                                               
//说明:      读取ITG3205陀螺仪原始数据
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
