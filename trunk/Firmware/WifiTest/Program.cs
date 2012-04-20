using System;
using System.Threading;
using Microsoft.SPOT;
using Microsoft.SPOT.Hardware;
using SecretLabs.NETMF.Hardware;
using SecretLabs.NETMF.Hardware.Netduino;
using System.IO.Ports;

namespace WifiTest {
	public class Program {

		public static void delay(int msec) {
			Thread.Sleep(msec);
		}

		public static void Main ()
		{
			// write your code here

			SerialPortHelper	serial = new SerialPortHelper(SerialPorts.COM1, 115200, Parity.None, 8, StopBits.One);

			delay(1000);
			serial.PrintLine("ATE0");
			delay(1000);
			serial.PrintLine("AT");
			delay(1000);
			serial.PrintLine("AT+WWPA=U17qGmb6eyHZMUv");
			delay(3000);
			serial.PrintLine("AT+NDHCP=0");
			delay(3000);
			serial.PrintLine("AT+NSET=192.168.1.25,255.255.255.0,192.168.1.1");
			delay(3000);
			serial.PrintLine("AT+WAUTO=0,WiFi,,12");
			delay(3000);
			serial.PrintLine("AT+NAUTO=1,1,,8011");
			delay(3000);
			serial.PrintLine("ATA");
			delay(3000);


			while (true) {
				serial.PrintLine("Ololo");
				delay(5000);
			}


		}

	}
}
