using System;
using System.Text;
using Microsoft.SPOT;
using Microsoft.SPOT.Hardware;

namespace Firmware {
	public class GyroscopePlug {

		private const int	GYRO_ADDRESS = 0x68;

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

		//I2CDevice.Configuration config;

		public GyroscopePlug () 
		{
			//config = new I2CDevice.Configuration(GYRO_ADDRESS, 400);
			 
			//WriteRegister( PWR_M, 0x80 );
			//WriteRegister( SMPL,  0x00 );
			//WriteRegister( DLPF,  0x18 );
		}

		public void GetData ()
		{
			//byte[] output = new byte[1];
			//ReadFromRegister( GX_H, output );
			//Debug.Print( output.ToString() );
			//ReadFromRegister( GX_L, output );
			//Debug.Print( output.ToString() );
		}
		
	}
}
