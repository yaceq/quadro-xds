using System;
using System.Threading;
using Microsoft.SPOT;
using Microsoft.SPOT.Hardware;
using SecretLabs.NETMF.Hardware;
using SecretLabs.NETMF.Hardware.Netduino;

namespace LedBlink {
	public class Program {
		public static void Main ()
		{
			// write your code here
			OutputPort led = new OutputPort(Pins.ONBOARD_LED, false);

			while (true) {
				led.Write(true); // turn on the LED
				Thread.Sleep(100); // sleep for 250ms
				led.Write(false); // turn off the LED
				Thread.Sleep(100); // sleep for 250ms
			}

		}

	}
}
