using System;
using System.Threading;
using Microsoft.SPOT;
using Microsoft.SPOT.Hardware;
using SecretLabs.NETMF.Hardware;
using SecretLabs.NETMF.Hardware.Netduino;
using System.IO.Ports;

namespace WifiTest {
	public class Program {
		public static void Main ()
		{
			// write your code here

			SerialPortHelper	serial = new SerialPortHelper(SerialPorts.COM1, 115200, Parity.None, 8, StopBits.One);

			OutputPort led = new OutputPort(Pins.ONBOARD_LED, false);

			while (true) {
				led.Write(true); // turn on the LED
				Thread.Sleep(100); // sleep for 250ms
				led.Write(false); // turn off the LED
				Thread.Sleep(100); // sleep for 250ms

				string s = serial.ReadLine();
				Debug.Print(">" + s);
				serial.PrintLine(s);
			}


		}

	}
}
