using System;
using System.Threading;
using Microsoft.SPOT;
using Microsoft.SPOT.Hardware;
using SecretLabs.NETMF.Hardware;
using SecretLabs.NETMF.Hardware.Netduino;

namespace Firmware {
	public class Program {

		private const byte	WHO		= 0x00;
		private const byte	SMPL	= 0x15;
		private const byte	DLPF	= 0x16;
		private const byte	INT_C	= 0x17;
		private const byte	INT_S	= 0x1A;
		private const byte	TMP_H	= 0x1B;
		private const byte	TMP_L	= 0x1C;
		private const byte	GX_H	= 0x1D;
		private const byte	GX_L	= 0x1E;
		private const byte	GY_H	= 0x1F;
		private const byte	GY_L	= 0x20;
		private const byte	GZ_H	= 0x21;
		private const byte	GZ_L	= 0x22;
		private const byte	PWR_M	= 0x3E;
		public static void Main ()
		{
			// write your code here
			//GyroscopePlug	gyro = new GyroscopePlug();
			
			//I2CBus bus = I2CBus.GetInstance();

			//I2CDevice.Configuration config = new I2CDevice.Configuration(0x68, 400);

			//bus.WriteRegister(config, PWR_M, 0x80, 1000);
			//Thread.Sleep(100);
			//bus.WriteRegister(config, SMPL, 0x00, 1000);
			//Thread.Sleep(100);
			//bus.WriteRegister(config, DLPF, 0x18, 1000);
			//Thread.Sleep(100);

			//byte[] buf = new byte[] { 0 };

			OutputPort led = new OutputPort(Pins.ONBOARD_LED, false);

			while (true) {


				led.Write(true); // turn on the LED
				Thread.Sleep(250); // sleep for 250ms
				led.Write(false); // turn off the LED
				Thread.Sleep(250); // sleep for 250ms

			    //Console.Clear();
			    //Debug.c
				//bus.ReadRegister(config, INT_S, buf, 1000);
				//if ((buf[0] & 0x01)==0x01) {

				//    int x, y, z;
				//    bus.ReadRegister(config, WHO, buf, 1000);	 

				//    bus.ReadRegister(config, GX_H, buf, 1000);	 
				//    x = buf[0]<<8;
				//    bus.ReadRegister(config, GX_L, buf, 1000);	 
				//    x |= buf[0];

				//    bus.ReadRegister(config, GY_H, buf, 1000);	 
				//    y = buf[0]<<8;
				//    bus.ReadRegister(config, GY_L, buf, 1000);	 
				//    y |= buf[0];

				//    bus.ReadRegister(config, GZ_H, buf, 1000);	 
				//    z = buf[0]<<8;
				//    bus.ReadRegister(config, GZ_L, buf, 1000);	 
				//    z |= buf[0];

				//    Debug.Print("Gyro: " + x.ToString() + " " + y.ToString() +" "+ z.ToString() );
				//} else {
				//    Debug.Print("FUCK!");
				//}

			}


		}

	}
}
