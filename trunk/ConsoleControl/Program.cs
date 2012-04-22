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



			try {
				while (true) {

					var gps = GamePad.GetState(0);
					var bufo = new byte[8];
					var bufi = new byte[8];

					bufo[0] = (byte)( 200 * gps.Triggers.Left );		//	throttle
					bufo[1] = (byte)( 200 * (gps.ThumbSticks.Right.X+1)/2 );		//	throttle
					bufo[2] = (byte)( 200 * (gps.ThumbSticks.Right.Y+1)/2 );		//	throttle
					bufo[3] = (byte)( 200 * (gps.ThumbSticks.Left.X+1)/2 );		//	throttle
					port.Write( bufo, 0, 4 );

					Console.Write( "{0,3:X} {0,3:X} {0,3:X} {0,3:X} : ", bufo[0], bufo[1], bufo[2], bufo[3] );



					port.Read( bufi, 0, 4 );
					//port.

					Console.Write( "{0,3:X} {0,3:X} {0,3:X} {0,3:X} : ", bufi[0], bufi[1], bufi[2], bufi[3] );

					Console.WriteLine();

					Thread.Sleep(20);
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
