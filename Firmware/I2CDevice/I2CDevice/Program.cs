using System;
using System.Threading;
using Microsoft.SPOT;
using Microsoft.SPOT.Hardware;
using SecretLabs.NETMF.Hardware;
using SecretLabs.NETMF.Hardware.Netduino;

namespace I2CDev
{
    public class Program
    {
        public static void Main()
        {
            I2CPlug gyro = new I2CPlug(0x68);

            gyro.WriteToRegister(0x3e, 0x80);
            Thread.Sleep(100);
            gyro.WriteToRegister(0x15, 0x00);
            Thread.Sleep(100);
            gyro.WriteToRegister(0x16, 0x19);
            Thread.Sleep(100);
            gyro.WriteToRegister(0x17, 0x01);
            Thread.Sleep(100);

            byte[] readBuffer = new byte[1] { 0 };
            int i = 0;
            for (; ; )
            {
                gyro.ReadFromRegister(0x1a, readBuffer);
                if ( (byte)(readBuffer[0] & 0x01) == 0x01)
                {
                    Debug.Print("Iteration : " + i.ToString() + "\n");
                    gyro.ReadFromRegister(0x00, readBuffer);
                    Debug.Print("ID : " + readBuffer[0].ToString() + "\n");
                    gyro.ReadFromRegister(0x1d, readBuffer);
                    Debug.Print(readBuffer[0].ToString() + "\n");
                    gyro.ReadFromRegister(0x1e, readBuffer);
                    Debug.Print(readBuffer[0].ToString() + "\n");
                    gyro.ReadFromRegister(0x1f, readBuffer);
                    Debug.Print(readBuffer[0].ToString() + "\n");
                    gyro.ReadFromRegister(0x20, readBuffer);
                    Debug.Print(readBuffer[0].ToString() + "\n");
                    gyro.ReadFromRegister(0x21, readBuffer);
                    Debug.Print(readBuffer[0].ToString() + "\n");
                    gyro.ReadFromRegister(0x22, readBuffer);
                    Debug.Print(readBuffer[0].ToString() + "\n");
                    
                }
                i++;
                //Thread.Sleep(1000);
            }
            
        }

    }
}
