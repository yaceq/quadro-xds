﻿using System;
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

		static float Curve( float value, float power ) 
		{
			float sign = Math.Sign(value);
			value = Math.Abs( value );
			value = (float)Math.Pow( value, power );
			return sign * value;
		}


		static List<string> commandList = new List<string>();
		static string incomingCommand = "";


		static void PushInCommand  ( string s ) { commandList.Add( s ); }
		static string PopInCommand ( ) {
			if (commandList.Count==0) {
				return null;
			} else {
				var s = commandList[0];
				commandList.RemoveAt(0);
				return s;
			}
		}


		private static void DataReceviedHandler( object sender, SerialDataReceivedEventArgs e )
		{
			SerialPort sp = (SerialPort)sender;
			string indata = sp.ReadExisting();

			foreach (char ch in indata) {
				if (ch=='\n') {
					PushInCommand( incomingCommand );
					incomingCommand = "";
				} else {
					incomingCommand += ch;
				}
			}
		}


		static void Main ( string[] args )
		{
			Thread.CurrentThread.CurrentCulture = CultureInfo.InvariantCulture;

			SerialPort	port = null;
			string portName  = "COM6";

			Console.WriteLine("waiting for connection on " + portName);
			
			do {
				
				try {

					port = new SerialPort( portName, 57600, Parity.None, 8, StopBits.One );
					port.Open();
					port.DataReceived += new SerialDataReceivedEventHandler(DataReceviedHandler);

					Console.WriteLine();
					Console.WriteLine("serial port is open.");

				} catch ( Exception ex ) {
					Console.Write(".");
					Thread.Sleep(250);
					port = null;
				}

			} while (port==null);


			Thread.Sleep(1000);


			try {
				while (true) {

					var gps = GamePad.GetState(0);

					if ( gps.IsButtonDown(Buttons.Back) ) {
						break;
					}

					/*if (gps.DPad.Left  == ButtonState.Pressed) stab_r = (byte)MathHelper.Clamp( stab_r - 1, 1, 255 );
					if (gps.DPad.Right == ButtonState.Pressed) stab_r = (byte)MathHelper.Clamp( stab_r + 1, 1, 255 );
					if (gps.DPad.Down  == ButtonState.Pressed) stab_g = (byte)MathHelper.Clamp( stab_g - 1, 1, 255 );
					if (gps.DPad.Up    == ButtonState.Pressed) stab_g = (byte)MathHelper.Clamp( stab_g + 1, 1, 255 );
					if (gps.IsButtonDown(Buttons.X)) roll_bias--;
					if (gps.IsButtonDown(Buttons.B)) roll_bias++;
					if (gps.IsButtonDown(Buttons.Y)) pitch_bias++;
					if (gps.IsButtonDown(Buttons.A)) pitch_bias--;
					if (gps.IsButtonDown(Buttons.LeftShoulder)) yaw_bias++;
					if (gps.IsButtonDown(Buttons.RightShoulder)) yaw_bias--;*/

					int throttle	=	(char)( 127 * Curve( gps.Triggers.Left,			0.5f ) ) & 0xFF;
					int roll		=	(char)( 127 * Curve( gps.ThumbSticks.Right.X,	0.5f ) ) & 0xFF;
					int pitch		=	(char)( 127 * Curve( gps.ThumbSticks.Right.Y,	0.5f ) ) & 0xFF;
					int yaw			=	(char)( 127 * Curve( gps.ThumbSticks.Left.X,	0.5f ) ) & 0xFF;

					string outCmd		=	string.Format("X{0,3:X}{1,3:X}{2,3:X}{3,3:X}", throttle, roll, pitch, yaw );

					Console.Write("OUT CMD : {0}  - ", outCmd );
					port.WriteLine( outCmd + "\n" );

					var inCmd = PopInCommand();

					if (inCmd!=null) {
						Console.WriteLine("IN CMD  : {0}", inCmd );
						while ( PopInCommand()!=null ) {
							// skip the rest of the commands...
						}
					} else {
						Console.WriteLine();
					}

					Thread.Sleep(30);
				}

			} catch (Exception ex) {
				Console.WriteLine("");
				Console.WriteLine("Exception:");
				Console.WriteLine("{0}", ex.Message);
				Console.WriteLine("Press Enter...");
				Console.ReadLine();
				if (port.IsOpen) {
					port.Close();
				}
				port.Dispose();
				port = null;
			}


			if (port.IsOpen) {
				port.Close();
			}
			port.Dispose();
			port = null;
		}
	}
}
