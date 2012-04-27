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

			float trim_roll	= 0;
			float trim_pitch	= 0;
			float trim_yaw	= 0;

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

		SerialPort	port = null;

		List<string> commandList = new List<string>();
		string incomingCommand = "";

		void PushInCommand  ( string s ) { commandList.Add( s ); }
		string PopInCommand ( ) {
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




		/// <summary>
		/// Allows the game to perform any initialization it needs to before starting to run.
		/// This is where it can query for any required services and load any non-graphic
		/// related content.  Calling base.Initialize will enumerate through any components
		/// and initialize them as well.
		/// </summary>
		protected override void Initialize ()
		{
			// TODO: Add your initialization logic here

			base.Initialize();
		}

		/// <summary>
		/// LoadContent will be called once per game and is the place to load
		/// all of your content.
		/// </summary>
		protected override void LoadContent ()
		{
			// Create a new SpriteBatch, which can be used to draw textures.
			spriteBatch = new SpriteBatch( GraphicsDevice );

			// TODO: use this.Content to load your game content here
		}

		/// <summary>
		/// UnloadContent will be called once per game and is the place to unload
		/// all content.
		/// </summary>
		protected override void UnloadContent ()
		{
			// TODO: Unload any non ContentManager content here
		}

		/// <summary>
		/// Allows the game to run logic such as updating the world,
		/// checking for collisions, gathering input, and playing audio.
		/// </summary>
		/// <param name="gameTime">Provides a snapshot of timing values.</param>
		protected override void Update ( GameTime gameTime )
		{
			var dr = this.GetService<DebugRender>();
			var ds = this.GetService<DebugStrings>();

			dr.Clear();
			ds.Clear();

			time += (float)gameTime.ElapsedGameTime.TotalSeconds;

			var gps = GamePad.GetState(0);

			// Allows the game to exit
			if (Keyboard.GetState().IsKeyDown(Keys.Escape) || gps.IsButtonDown(Buttons.Back)) {
				this.Exit();
			}

			if (gps.IsButtonDown(Buttons.A)) calibrating = true;
			if (gps.IsButtonDown(Buttons.B)) calibrating = false;

			if (gps.DPad.Left  == ButtonState.Pressed)	 trim_roll	-= 0.3f;
			if (gps.DPad.Right == ButtonState.Pressed)	 trim_roll	+= 0.3f;
			if (gps.DPad.Down  == ButtonState.Pressed)	 trim_pitch	-= 0.3f;
			if (gps.DPad.Up    == ButtonState.Pressed)	 trim_pitch	+= 0.3f;
			if (gps.IsButtonDown(Buttons.LeftShoulder))  trim_yaw	+= 0.3f;
			if (gps.IsButtonDown(Buttons.RightShoulder)) trim_yaw	-= 0.3f;
			/*if (gps.IsButtonDown(Buttons.X)) roll_bias--;
			if (gps.IsButtonDown(Buttons.B)) roll_bias++;
			if (gps.IsButtonDown(Buttons.Y)) pitch_bias++;
			if (gps.IsButtonDown(Buttons.A)) pitch_bias--;
			if (gps.IsButtonDown(Buttons.LeftShoulder)) yaw_bias++;
			if (gps.IsButtonDown(Buttons.RightShoulder)) yaw_bias--;*/

			int throttle	=	(int)MathHelper.Clamp( (127 * Curve( gps.Triggers.Left,		  1.0f )),		 	    -127, 127 ) & 0xFF;
			int roll		=	(int)MathHelper.Clamp( (127 * Curve( gps.ThumbSticks.Right.X, 0.7f )) + trim_roll,  -127, 127 ) & 0xFF;
			int pitch		=	(int)MathHelper.Clamp( (127 * Curve( gps.ThumbSticks.Right.Y, 0.7f )) + trim_pitch, -127, 127 ) & 0xFF;
			int yaw			=	(int)MathHelper.Clamp( (127 * Curve( gps.ThumbSticks.Left.X,  0.7f )) + trim_yaw,   -127, 127 ) & 0xFF;

			string outCmd		=	string.Format("X{0,3:X}{1,3:X}{2,3:X}{3,3:X}", throttle, roll, pitch, yaw );

			Console.Write("OUT CMD : {0}  - ", outCmd );
			port.WriteLine( outCmd + "\n" );

			var inCmd = PopInCommand();

			if (inCmd!=null) {
				Console.WriteLine("IN CMD  : {0}", inCmd );

				HandleInCommand( inCmd );

				while ( PopInCommand()!=null ) {
					// skip the rest of the commands...
				}
			} else {
				Console.WriteLine();
			}

			base.Update( gameTime );
		}



		List<Vector3>	accelData;
		List<Vector3>	gyroData ;

		Vector3	currentAccel;
		Vector3 currentGyro;
		Vector3	biasAccel;
		Vector3 biasGyro;

		float time = 0;
		float lastIMtime = 0;
		float deltaIMtime = 0;
		bool  calibrating = true;

		void HandleInCommand( string s )
		{
			float kG = 1.5f*9.8f/32;

			var list = s.Split(new[]{' '}, StringSplitOptions.RemoveEmptyEntries);
			if (list[0]=="X") {
				deltaIMtime = time - lastIMtime;
				lastIMtime = time;

				currentGyro.X  = MathHelper.ToRadians(float.Parse(list[1]) / 14.375f);
				currentGyro.Y  = MathHelper.ToRadians(float.Parse(list[2]) / 14.375f);
				currentGyro.Z  = MathHelper.ToRadians(float.Parse(list[3]) / 14.375f);
				currentAccel.X = float.Parse(list[4]) * kG ;
				currentAccel.Y = float.Parse(list[5]) * kG ;
				currentAccel.Z = float.Parse(list[6]) * kG ;

				currentAccel -= biasAccel;
				currentGyro -= biasGyro;

				if (calibrating) {
					if (accelData==null) {
						biasAccel = Vector3.Zero;
						biasGyro  = Vector3.Zero;
						accelData = new List<Vector3>();
						gyroData  = new List<Vector3>();
						accelData.Add( currentAccel );
						gyroData.Add( currentGyro );
					} else {
						accelData.Add( currentAccel );
						gyroData.Add( currentGyro );
					}
				} else {
					if (accelData!=null) {
						foreach (var v in accelData) { biasAccel += v; }
						foreach (var v in gyroData)  { biasGyro += v; }
						biasAccel /= accelData.Count;
						biasGyro  /= gyroData.Count;
						accelData = null;
						gyroData  = null;
						copter = Matrix.Identity;
					} else {

						float dt = 	deltaIMtime;

						Quaternion qx = Quaternion.CreateFromAxisAngle( Vector3.UnitX, currentGyro.X * dt );
						Quaternion qy = Quaternion.CreateFromAxisAngle( Vector3.UnitY, currentGyro.Y * dt );
						Quaternion qz = Quaternion.CreateFromAxisAngle( Vector3.UnitZ, currentGyro.Z * dt );
						Quaternion q = qx + qy + qz;
						q.Normalize();

						copter *= Matrix.CreateFromQuaternion( q );
						velocity += currentAccel * dt;

						//copter.Translation += velocity * dt;

					}
				}
			}
		}


		Vector3 velocity = Vector3.Zero;
		Matrix	copter  = Matrix.Identity;




		/// <summary>
		/// This is called when the game should draw itself.
		/// </summary>
		/// <param name="gameTime">Provides a snapshot of timing values.</param>
		protected override void Draw ( GameTime gameTime )
		{
			GraphicsDevice.Clear( Color.Black );

			var dr = this.GetService<DebugRender>();
			var ds = this.GetService<DebugStrings>();
			// TODO: Add your drawing code here

			var view = Matrix.CreateLookAt( Vector3.Backward * 5 + Vector3.Up * 3, Vector3.Zero, Vector3.Up );
			var proj = Matrix.CreatePerspectiveFieldOfView( MathHelper.ToRadians(70), GraphicsDevice.Viewport.AspectRatio, 0.1f, 100 );

			dr.SetMatrix( view, proj );
			dr.DrawGrid(10);

			dr.DrawBasis( copter, 0.3f );
			dr.DrawRing( copter, 0.2f, Color.Yellow );

			if (calibrating) {
				ds.Add(string.Format("Calibrating : {0} samples", accelData==null ? 0 : accelData.Count));
			} else {
				ds.Add(string.Format("Gyro bias  : {0}", biasGyro));
				ds.Add(string.Format("Accel bias : {0}", biasAccel));
				ds.Add(string.Format("IM dT : {0}", deltaIMtime));
			}

			base.Draw( gameTime );
		}
	}
}
