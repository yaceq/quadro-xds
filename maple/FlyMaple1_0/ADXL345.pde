
////////加速度传感器 ADXL345 功能函数/////////////////////////////
#define ACC (0x53)    //定义ADXL345地址，ALT ADDRESS 管脚接地
#define A_TO_READ (6) //读取每次占用的字节数 (每个坐标轴占两个字节)
#define XL345_DEVID   0xE5 //ADXL345 加速度地址，需要注意芯片有一个地址选择线将AD0连接到GND口
// ADXL345控制寄存器
#define ADXLREG_BW_RATE      0x2C
#define ADXLREG_POWER_CTL    0x2D
#define ADXLREG_DATA_FORMAT  0x31
#define ADXLREG_FIFO_CTL     0x38
#define ADXLREG_BW_RATE      0x2C
#define ADXLREG_TAP_AXES     0x2A
#define ADXLREG_DUR          0x21

//ADXL345数据寄存器
#define ADXLREG_DEVID        0x00
#define ADXLREG_DATAX0       0x32
#define ADXLREG_DATAX1       0x33
#define ADXLREG_DATAY0       0x34
#define ADXLREG_DATAY1       0x35
#define ADXLREG_DATAZ0       0x36
#define ADXLREG_DATAZ1       0x37

// 加速度传感器误差修正的偏移量
int16 a_offx = 0;
int16 a_offy = 0;
int16 a_offz = 0;

////////////////////////////////////////////////////////////////////////////////////
//函数原型:  void writeTo(uint8 DEVICE, uint8 address, uint8 val)             	     
//参数说明:  DEVICE: I2C设备地址
//           address:操作寄存器地址 
//           val:写入寄存器值
//返回值:    无                                                               
//说明:      通过I2C总线将val写入到对应地址寄存器中
///////////////////////////////////////////////////////////////////////////////////
void writeTo(uint8 DEVICE, uint8 address, uint8 val) 
{
  // all i2c transactions send and receive arrays of i2c_msg objects 
  i2c_msg msgs[1]; // we dont do any bursting here, so we only need one i2c_msg object
 uint8 msg_data[2];
  
  msg_data = {address,val};  //写两个数据，一个地址，一个值
  msgs[0].addr = DEVICE;
  msgs[0].flags = 0; // 写操作
  msgs[0].length = 2; //写两个数据
  msgs[0].data = msg_data;
  i2c_master_xfer(I2C1, msgs, 1,0);  //
}

////////////////////////////////////////////////////////////////////////////////////
//函数原型:  void readFrom(uint8 DEVICE, uint8 address, uint8 num, uint8 *msg_data)              	     
//参数说明:  DEVICE: I2C设备地址
//           address:操作寄存器地址 
//           num:读取数量
//           *msg_data:读取数据存放指针
//返回值:    无                                                               
//说明:      通过I2C总线读取数据
///////////////////////////////////////////////////////////////////////////////////
void readFrom(uint8 DEVICE, uint8 address, uint8 num, uint8 *msg_data) 
{
  i2c_msg msgs[1]; 
  msg_data[0] = address;
  
  msgs[0].addr = DEVICE;
  msgs[0].flags = 0; //标志为0，是写操作
  msgs[0].length = 1; // just one byte for the address to read, 0x00
  msgs[0].data = msg_data;
  i2c_master_xfer(I2C1, msgs, 1,0);
  
  msgs[0].addr = DEVICE;
  msgs[0].flags = I2C_MSG_READ; //读取
  msgs[0].length = num; // 读取字节数
  msgs[0].data = msg_data;
  i2c_master_xfer(I2C1, msgs, 1,0);
}
////////////////////////////////////////////////////////////////////////////////////
//函数原型:  void initAcc(void)             	     
//参数说明:  无                                        
//返回值:    无                                                               
//说明:      初始化ADXL345加速度计
///////////////////////////////////////////////////////////////////////////////////
void initAcc(void) 
{
    //all i2c transactions send and receive arrays of i2c_msg objects 
  i2c_msg msgs[1]; // we dont do any bursting here, so we only need one i2c_msg object
  uint8 msg_data[2];
  msg_data = {0x00,0x00};
  msgs[0].addr = ACC;
  msgs[0].flags = 0; 
  msgs[0].length = 1; // just one byte for the address to read, 0x00
  msgs[0].data = msg_data;
  
  i2c_master_xfer(I2C1, msgs, 1,0);
  msgs[0].addr = ACC;
  msgs[0].flags = I2C_MSG_READ; 
  msgs[0].length = 1; // just one byte for the address to read, 0x00
  msgs[0].data = msg_data;
  i2c_master_xfer(I2C1, msgs, 1,0);
  // now we check msg_data for our 0xE5 magic number 
  uint8 dev_id = msg_data[0];
  //SerialUSB.print("Read device ID from xl345: ");
  //SerialUSB.println(dev_id,HEX);
  
  if (dev_id != XL345_DEVID) 
  {
    SerialUSB.println("Error, incorrect xl345 devid!");
    SerialUSB.println("Halting program, hit reset...");
    waitForButtonPress(0);
  }
  //调用 ADXL345
  writeTo(ACC, ADXLREG_POWER_CTL, 0x00); //清零 
  writeTo(ACC, ADXLREG_POWER_CTL, 0xff);//休眠
  writeTo(ACC, ADXLREG_POWER_CTL, 0x08); //仅开启工作模式
  //设定在 +-2g 时的默认读数
}
////////////////////////////////////////////////////////////////////////////////////
//函数原型:  void getAccelerometerData(int16 * result)            	     
//参数说明:  * result: 读取加速值指针                                        
//返回值:    无                                                               
//说明:      读取ADXL345加速度计原始数据
///////////////////////////////////////////////////////////////////////////////////
void getAccelerometerData(int16 * result) 
{
  int16 regAddress = ADXLREG_DATAX0;    //加速度传感器ADXL345第一轴的数据地址
  uint8 buff[A_TO_READ];

  readFrom(ACC, regAddress, A_TO_READ, buff); //读取加速度传感器ADXL345的数据

  //每个轴的读数有10位分辨率，即2个字节.  
  //我们要转换两个bytes为一个int变量
  result[0] = (((int16)buff[1]) << 8) | buff[0] + a_offx;   
  result[1] = (((int16)buff[3]) << 8) | buff[2] + a_offy;
  result[2] = (((int16)buff[5]) << 8) | buff[4] + a_offz;
}


