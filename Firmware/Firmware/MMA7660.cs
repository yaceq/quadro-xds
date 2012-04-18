using System;
using System.Threading;
using Microsoft.SPOT;
using Microsoft.SPOT.Hardware;
using SecretLabs.NETMF.Hardware;
using SecretLabs.NETMF.Hardware.Netduino;


namespace Firmware {
	public class MMA7660 {

		public const byte MMA7660addr    = 0x4c;
		public const byte MMA7660_X      = 0x00;
		public const byte MMA7660_Y      = 0x01;
		public const byte MMA7660_Z      = 0x02;
		public const byte MMA7660_TILT   = 0x03;
		public const byte MMA7660_SRST   = 0x04;
		public const byte MMA7660_SPCNT  = 0x05;
		public const byte MMA7660_INTSU  = 0x06;
		public const byte MMA7660_MODE   = 0x07;
		public const byte MMA7660_SR     = 0x08;
		public const byte MMA7660_PDET   = 0x09;
		public const byte MMA7660_PD     = 0x0A;		

		/*I2CDevice.Configuration	config;  
		I2CBus					bus;  */
		I2CPlug		i2c;

		public float AccelX { get; protected set; }
		public float AccelY { get; protected set; }
		public float AccelZ { get; protected set; }

		public MMA7660 ()
		{
			/*config = new I2CDevice.Configuration( MMA7660addr, 400 );
			bus = I2CBus.GetInstance();*/
		}


		public void Init ()
		{
			i2c = new I2CPlug( MMA7660addr, 100 );
			writemem( MMA7660_MODE, 0x00 );
			writemem( MMA7660_SR,	0x07 );
			writemem( MMA7660_MODE, 0x01 );
			Thread.Sleep( 100 );  // startup 
		}


		public void Update ()
		{
			 AccelX = (float)((char)( readmem(MMA7660_X) << 2 ))/4;
			 AccelY = (float)((char)( readmem(MMA7660_Y) << 2 ))/4;
			 AccelZ = (float)((char)( readmem(MMA7660_Z) << 2 ))/4;
		}



		protected void writemem ( byte addr, byte value ) {
			i2c.WriteToRegister( addr, value );
			//bus.WriteRegister( config, addr, value, 1000 );
		}

		protected byte readmem ( byte addr ) {
			try {
				return i2c.ReadFromRegister( addr );
			} catch (Exception ex) {
				writemem( MMA7660_MODE, 0x00 );
				writemem( MMA7660_SR,	0x07 );
				writemem( MMA7660_MODE, 0x01 );
				Thread.Sleep( 100 );  // startup 
				return i2c.ReadFromRegister( addr );
			}
			//return bus.ReadRegister( config, addr, 1000 );
		}

	}
}
