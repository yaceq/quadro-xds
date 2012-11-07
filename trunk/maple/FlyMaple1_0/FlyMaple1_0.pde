#include <stdio.h>
#include "wirish.h"
#include "i2c.h"

extern uint16 MotorData[6];  //电机控制寄存器 

extern volatile unsigned int chan1PPM;  //PPM捕获值寄存器
extern volatile unsigned int chan2PPM;
extern volatile unsigned int chan3PPM;
extern volatile unsigned int chan4PPM;


char str[512]; 

////////////////////////////////////////////////////////////////////////////////////
//函数原型:  void setup()               	     
//参数说明:  无                                        
//返回值:    无                                                               
//说明:      FlyMaple板 初始化函数
///////////////////////////////////////////////////////////////////////////////////
void setup()
{
  motorInit();       //电机控制初始化 
  capturePPMInit();  //捕获遥控器接收机PPM输入信号功能初始化 

  //configure I2C port 1 (pins 5, 9) with no special option flags (second argument)
  i2c_master_enable(I2C1, 0);  //设置I2C1接口，主机模式
  
  initAcc();            //初始化加速度计
  initGyro();           //初始化陀螺仪
  bmp085Calibration();  //初始化气压高度计
  compassInit(false);   //初始化罗盘
  //compassCalibrate(1);  //校准一次罗盘，gain为1.3Ga
  //commpassSetMode(0);  //设置为连续测量模式 
  
}
////////////////////////////////////////////////////////////////////////////////////
//函数原型:  void loop()            	     
//参数说明:  无                                        
//返回值:    无                                                               
//说明:      主函数，程序主循环
///////////////////////////////////////////////////////////////////////////////////
void loop()
{
  //***************ADXL345加速度读取测试例子*****************************
  int16 acc[3];
  getAccelerometerData(acc);  //读取加速度
  SerialUSB.print("Xacc=");
  SerialUSB.print(acc[0]);
  SerialUSB.print("    ");
  SerialUSB.print("Yacc=");  
  SerialUSB.print(acc[1]);
  SerialUSB.print("    ");
  SerialUSB.print("Zacc=");  
  SerialUSB.print(acc[2]);
  SerialUSB.print("    ");
  //delay(100);
  //***********************************************************/

    
  //***************ITG3205加速度读取测试例子*****************************
    int16 gyro[4];
    getGyroscopeData(gyro);    //读取陀螺仪
    
    SerialUSB.print("Xg=");
    SerialUSB.print(gyro[0]);
    SerialUSB.print("    ");
    SerialUSB.print("Yg=");  
    SerialUSB.print(gyro[1]);
    SerialUSB.print("    ");
    SerialUSB.print("Zg=");  
    SerialUSB.print(gyro[2]);
    SerialUSB.print("    ");
    //SerialUSB.print("temperature=");  
    //SerialUSB.print(gyro[3]);
    //SerialUSB.print("    ");
    //delay(100);
  //*********************************************************************/  
  //****************************BMP085 气压计测试例子******************
  int16 temperature = 0;
  int32 pressure = 0;
  int32 centimeters = 0;
  temperature = bmp085GetTemperature(bmp085ReadUT());
  pressure = bmp085GetPressure(bmp085ReadUP());
  centimeters = bmp085GetAltitude(); //获得海拔高度，单位厘米
  //SerialUSB.print("Temperature: ");
  SerialUSB.print(temperature, DEC);
  SerialUSB.print(" *0.1 deg C ");
  //SerialUSB.print("Pressure: ");
  SerialUSB.print(pressure, DEC);
  SerialUSB.print(" Pa ");
  SerialUSB.print("Altitude: ");
  SerialUSB.print(centimeters, DEC);
   SerialUSB.print(" cm ");
  //SerialUSB.println("    ");
  //delay(1000);
  //********************************************************************/
  //******************************HMC5883罗盘测试****************************
  float Heading;
  Heading = compassHeading();
  //SerialUSB.print("commpass: ");
  SerialUSB.print(Heading, DEC);
  SerialUSB.println(" degree");
  delay(100);
 //***************************************************************************/
  /*************************电机驱动测试***************************************************************
  MotorData[0] = 1;  //首先将PWM设置为最低以开启电调
  MotorData[1] = 1;
  MotorData[2] = 1;
  MotorData[3] = 1;
  MotorData[4] = 1;
  MotorData[5] = 1; 
  motorCcontrol();   //计算各个电机控制量之差,将这个值用于定时器产生中断改变相应电机脉冲高电平时间
  delay(1000); 
  MotorData[0] = 500;  //控制6个电调使电机按照一半速度运行
  MotorData[1] = 500;
  MotorData[2] = 500;
  MotorData[3] = 500;
  MotorData[4] = 500;
  MotorData[5] = 500; 
  motorCcontrol();   //计算各个电机控制量之差,将这个值用于定时器产生中断改变相应电机脉冲高电平时间
  while(1);  
  *********************************************************************************************************/
  /********************无线遥控器RC 的 PPM 捕获测试****************************************************
  SerialUSB.print("PPM Channel 1: ");
  SerialUSB.print(chan1PPM, DEC);
  SerialUSB.print("  ");  
  SerialUSB.print("PPM Channel 2: ");
  SerialUSB.print(chan2PPM, DEC);
  SerialUSB.print("  ");  
  SerialUSB.print("PPM Channel 3: ");
  SerialUSB.print(chan3PPM, DEC);
  SerialUSB.print("  ");  
  SerialUSB.print("PPM Channel 4: ");
  SerialUSB.print(chan4PPM, DEC);
  SerialUSB.println("  ");  
  delay(100);
  ***************************************************************************************************/
}
