using System;
using System.Threading;

using Microsoft.SPOT;
using Microsoft.SPOT.Hardware;

using GHIElectronics.NETMF.FEZ;
using Firmware;

namespace LedBlinkP2 {
	public class Program {
		public static void Main ()
		{
			ITG3200	gyro = new ITG3200();

			/*Wire.begin();      // if experiencing gyro problems/crashes while reading XYZ values
								// please read class constructor comments for further info.*/

			Debug.Print("Starting...");

			Thread.Sleep(1000);
			// Use ITG3200_ADDR_AD0_HIGH or ITG3200_ADDR_AD0_LOW as the ITG3200 address 
			// depending on how AD0 is connected on your breakout board, check its schematics for details

			gyro.init( ITG3200.ITG3200_ADDR_AD0_LOW ); 
  
			Debug.Print("Gyro calibrating...");
			  
			gyro.zeroCalibrate(2500, 2);
			  
			Debug.Print("Done.");


			while (true) {

				while (gyro.isRawDataReady()) {

					float x,y,z;

					gyro.readGyro( out x, out y, out z ); 
					Debug.Print("Gyro: " + x.ToString() + " "
										 + y.ToString()	+ " "
										 + z.ToString() );

					Thread.Sleep(500);
				}				

			}
		}

	}
}
