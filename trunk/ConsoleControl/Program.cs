using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.IO.Ports;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Input;
using System.Globalization;


namespace ConsoleControl {
	class Program {
		static void Main ( string[] args )
		{
			Thread.CurrentThread.CurrentCulture = CultureInfo.InvariantCulture;

			SerialPort	port = null;
			
			do {
				try {
					port = new SerialPort( "COM6", 9600, Parity.None, 8, StopBits.One );
					port.NewLine = "\r\n";
					port.Open();

					Console.WriteLine();
					Console.WriteLine("Serial port is open!");

				} catch ( Exception ex ) {
					Console.Write(".");
					Thread.Sleep(250);
					port = null;
				}
			} while (port==null);


			while (true) {
				var s = port.ReadLine();
				Console.WriteLine(s);

				if (s=="DONE") {
					break;
				}
			}

			byte stab_r = 200;
			byte stab_g = 200;

			int roll_bias = 0;
			int pitch_bias = 0;
			int yaw_bias = 0;

			try {
				while (true) {

					var gps = GamePad.GetState(0);
					var bufo = new byte[8];
					var bufi = new byte[8];

					if (gps.DPad.Left  == ButtonState.Pressed) stab_r = (byte)MathHelper.Clamp( stab_r - 1, 1, 255 );
					if (gps.DPad.Right == ButtonState.Pressed) stab_r = (byte)MathHelper.Clamp( stab_r + 1, 1, 255 );
					if (gps.DPad.Down  == ButtonState.Pressed) stab_g = (byte)MathHelper.Clamp( stab_g - 1, 1, 255 );
					if (gps.DPad.Up    == ButtonState.Pressed) stab_g = (byte)MathHelper.Clamp( stab_g + 1, 1, 255 );
					if (gps.IsButtonDown(Buttons.X)) roll_bias--;
					if (gps.IsButtonDown(Buttons.B)) roll_bias++;
					if (gps.IsButtonDown(Buttons.Y)) pitch_bias++;
					if (gps.IsButtonDown(Buttons.A)) pitch_bias--;
					if (gps.IsButtonDown(Buttons.LeftShoulder)) yaw_bias++;
					if (gps.IsButtonDown(Buttons.RightShoulder)) yaw_bias--;

					bufo[0] = (byte)(              200 * gps.Triggers.Left );		//	throttle
					bufo[1] = (byte)( roll_bias  + 200 * (gps.ThumbSticks.Right.X+1)/2 );		//	throttle
					bufo[2] = (byte)( pitch_bias + 200 * (gps.ThumbSticks.Right.Y+1)/2 );		//	throttle
					bufo[3] = (byte)( yaw_bias   + 200 * (gps.ThumbSticks.Left.X+1)/2 );		//	throttle
					bufo[4] = (byte)( stab_r );
					bufo[5] = (byte)( stab_g );
					port.Write( bufo, 0, 6 );

					Console.Write( "{0,3:X} {1,3:X} {2,3:X} {3,3:X} {4,3:D} {5,3:D} {6} {7} {8}: ", bufo[0], bufo[1], bufo[2], bufo[3], bufo[4], bufo[5], roll_bias, pitch_bias, yaw_bias );



					//port.Read( bufi, 0, 4 );
					var s = port.ReadLine();
					//port.

					//Console.Write( "{0,3:X} {0,3:X} {0,3:X} {0,3:X} : ", bufi[0], bufi[1], bufi[2], bufi[3] );

					Console.WriteLine(s);

					Thread.Sleep(10);
				}

			} catch (Exception ex) {
				Console.WriteLine("");
				Console.WriteLine("Exception:");
				Console.WriteLine("{0}", ex.Message);
				Console.WriteLine("Press Enter...");
				Console.ReadLine();
				port.Dispose();
				port = null;
			}
		}
	}
}
