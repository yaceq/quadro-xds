using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Audio;
using Microsoft.Xna.Framework.Content;
using Microsoft.Xna.Framework.GamerServices;
using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework.Input;
using Microsoft.Xna.Framework.Media;
using System.Globalization;
using System.IO.Ports;
using Misc;

namespace QuadroTracker {
	/// <summary>
	/// This is the main type for your game
	/// </summary>
	public class QuadroTracker : Microsoft.Xna.Framework.Game {
		GraphicsDeviceManager graphics;
		SpriteBatch spriteBatch;

			float trim_roll		= 90;
			float trim_pitch	= 90;
			float trim_yaw		= 90;

		public Tracker	tracker;

		public QuadroTracker ()
		{
			Thread.CurrentThread.CurrentCulture = CultureInfo.InvariantCulture;

			graphics = new GraphicsDeviceManager( this );
			Content.RootDirectory = "Content";

			this.AddServiceAndComponent( new DebugRender( this ) );
			this.AddServiceAndComponent( new DebugStrings( this, "debugFont", 1 ) );

			graphics.PreferredBackBufferWidth	=	640;
			graphics.PreferredBackBufferHeight	=	480;
			graphics.PreferMultiSampling		=	true;

			this.IsMouseVisible = true;

			Content.RootDirectory = "Content";

			port = new SerialPort( "COM6", 57600, Parity.None, 8, StopBits.One );
			port.Open();
			port.DataReceived += new SerialDataReceivedEventHandler(DataReceviedHandler);

			Console.WriteLine();
			Console.WriteLine("serial port is open.");
		}



		protected override void OnExiting ( object sender, EventArgs args )
		{
			port.Close();
			base.OnExiting( sender, args );
		}



		static float Curve( float value, float power ) 
		{
			float sign = Math.Sign(value);
			value = Math.Abs( value );
			value = (float)Math.Pow( value, power );
			return sign * value;
		}

		
		#region COM port stuff
		SerialPort	port = null;

		List<string> commandList = new List<string>();
		string incomingCommand = "";


		void PushInCommand  ( string s ) 
		{ 
			commandList.Add( s ); 
		}


		string PopInCommand ( ) 
		{
			if (commandList.Count==0) {
				return null;
			} else {
				var s = commandList[0];
				commandList.RemoveAt(0);
				return s;
			}
		}


		private void DataReceviedHandler( object sender, SerialDataReceivedEventArgs e )
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
		#endregion



		protected override void Initialize ()
		{
			base.Initialize();
		}



		protected override void LoadContent ()
		{
			spriteBatch = new SpriteBatch( GraphicsDevice );
		}



		protected override void UnloadContent ()
		{
		}


		struct RawImuData {
			public int xRot;
			public int yRot;
			public int zRot;
		}


		List<RawImuData>	calibrationData = new List<RawImuData>();


		protected override void Update ( GameTime gameTime )
		{
			var dr = this.GetService<DebugRender>();
			var ds = this.GetService<DebugStrings>();

			dr.Clear();
			ds.Clear();

			string cmd = PopInCommand();

			if (cmd!=null) {
				Console.WriteLine("CMD: {0}", cmd);
				if (cmd[0]=='X') {
					var list	= cmd.Split(new []{' '}, StringSplitOptions.RemoveEmptyEntries);
					RawImuData rawImuData = new RawImuData();
					rawImuData.xRot	= Int16.Parse(list[1], NumberStyles.HexNumber);
					rawImuData.yRot	= Int16.Parse(list[2], NumberStyles.HexNumber);
					rawImuData.zRot	= Int16.Parse(list[3], NumberStyles.HexNumber);
				}								
			}

			port.WriteLine("X 0 0 0 0\n");

			Thread.Sleep(40);

			base.Update( gameTime );
		}



		protected override void Draw ( GameTime gameTime )
		{
			//UpdateTracking( (float)gameTime.ElapsedGameTime.TotalSeconds );

			GraphicsDevice.Clear( Color.Black );

			var dr = this.GetService<DebugRender>();
			var ds = this.GetService<DebugStrings>();
			// TODO: Add your drawing code here

			var view = Matrix.CreateLookAt( Vector3.Backward * 3 + Vector3.Up * 1.5f, Vector3.Zero, Vector3.Up );
			var proj = Matrix.CreatePerspectiveFieldOfView( MathHelper.ToRadians(70), GraphicsDevice.Viewport.AspectRatio, 0.1f, 100 );


			


			dr.SetMatrix( view, proj );
			dr.DrawGrid(10);


			base.Draw( gameTime );
		}
	}
}
