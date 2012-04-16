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
            gyro.WriteToRegister(0x16, 0x18);
            Thread.Sleep(100);
            //gyro.WriteToRegister(0x17, 0x05);

            byte[] readBuffer = new byte[1] { 0 };
            int i = 0;
            for (; ; )
            {
                Debug.Print("Iteration : " + i.ToString() + "\n");
                gyro.ReadFromRegister(0x00, readBuffer);
                Debug.Print("ID : " + readBuffer[0].ToString() + "\n");
                gyro.ReadFromRegister(0x1d, readBuffer);
                Debug.Print(readBuffer[0].ToString() + "\n");
                gyro.ReadFromRegister(0x1e, readBuffer);
                Debug.Print(readBuffer[0].ToString() + "\n");
                Thread.Sleep(100);
                i++;
            }
            
        }

    }
}
