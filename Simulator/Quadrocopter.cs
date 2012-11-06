using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Audio;
using Microsoft.Xna.Framework.Content;
using Microsoft.Xna.Framework.GamerServices;
using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework.Input;
using Microsoft.Xna.Framework.Media;
using System.Threading;
using BEPUphysics;
using BEPUphysics.Collidables;
using BEPUphysics.DataStructures;
using BEPUphysics.CollisionShapes.ConvexShapes;
using BEPUphysicsDrawer.Models;
using BEPUphysics.Entities.Prefabs;
using BEPUphysics.EntityStateManagement;
using BEPUphysics.PositionUpdating;
using Forms = System.Windows.Forms;
//using ViconDataStreamSDK.DotNET;
using System.IO.Ports;
using System.ComponentModel;
using Misc;



namespace Simulator {
	public class Quadrocopter {

		[Category("Tracking")]	public bool			TrackObject { set; get; }


		[Category("Motors")]	public float	RollK { set; get; }
		[Category("Motors")]	public float	RollD { set; get; }


		[Category("Motors")]	public volatile int	Rotor1;
		[Category("Motors")]	public volatile int	Rotor2;
		[Category("Motors")]	public volatile int	Rotor3;
		[Category("Motors")]	public volatile int	Rotor4;
        public string Name { protected set; get; }

		public int ThrustTest =  0;

		Model	frame;
		Model	propellerA;
		Model	propellerB;

		Game	game;
		World	world;
		Matrix	worldTransform;

		public StreamWriter	logWriter = null;
		public bool firstLine = false;



		public Quadrocopter( Game game, World world, Vector3 position, string name )
		{
            Name = name;

			this.game		=	game;
			this.world		=	world;

			frame			=	game.Content.Load<Model>( @"scenes\quadFrame" );
			propellerA		=	game.Content.Load<Model>( @"scenes\propellerA" );
			propellerB		=	game.Content.Load<Model>( @"scenes\propellerB" );

			worldTransform	=	Matrix.Identity;
		}



		float oldYaw, oldPitch, oldRoll;
		Vector3	oldUp;

		List<float> vFYaw	= new List<float>();
		List<float> vFPitch	= new List<float>();
		List<float> vFRoll	= new List<float>();

		public void Update ( float dt )
		{
			var ds	= game.GetService<DebugStrings>();
			var cfg = game.GetService<Settings>().Configuration;
			var dr	= game.GetService<DebugRender>();

			UpdateFromTracker();

			float yaw, pitch, roll;
			worldTransform.ToAngles( out yaw, out pitch, out roll );

			yaw   = MathHelper.ToDegrees( yaw   );
			pitch = MathHelper.ToDegrees( pitch );
			roll  = MathHelper.ToDegrees( roll  );

			float vYaw   = ( yaw   - oldYaw   ) / dt ;
			float vPitch = ( pitch - oldPitch ) / dt ;
			float vRoll  = ( roll  - oldRoll  ) / dt ;

			oldYaw   = yaw;
			oldPitch = pitch;
			oldRoll  = roll;

		#if false

			ds.Add(string.Format( "YAW   = {0,10:F3} / {1,10:F3}", yaw   , vYaw   ) );
			ds.Add(string.Format( "PITCH = {0,10:F3} / {1,10:F3}", pitch , vPitch ) );
			ds.Add(string.Format( "ROLL  = {0,10:F3} / {1,10:F3}", roll  , vRoll  ) );

			var gps = GamePad.GetState(0);

			float thrust	= gps.Triggers.Right;

			Vector2	rud		= gps.ThumbSticks.Right;
			Vector2	rud2	= gps.ThumbSticks.Left;

			bool  engine	= thrust > 0.02f;
			float rollT		= - ( roll  * cfg.RollK  + rud.X  + vRoll  * cfg.RollD  );
			float pitchT	= - ( pitch * cfg.PitchK + rud.Y  + vPitch * cfg.PitchD );
			float yawT		= 0;//- ( yaw   * cfg.YawK   + rud2.X + vYaw   * cfg.YawD   ) * thrust;

			float t1		= MathHelper.Clamp( thrust + rollT + pitchT + yawT, 0, 1 );
			float t2		= MathHelper.Clamp( thrust + rollT - pitchT - yawT, 0, 1 );
			float t3		= MathHelper.Clamp( thrust - rollT - pitchT + yawT, 0, 1 );
			float t4		= MathHelper.Clamp( thrust - rollT + pitchT - yawT, 0, 1 );*/
		#else
			var gps = GamePad.GetState(0);

			float thrust	= gps.Triggers.Right;

			var up			= worldTransform.Up;
			var omega		= Vector3.Cross( up, ( up - oldUp ) / dt );
			var localOmega	= Vector3.TransformNormal( omega, Matrix.Invert(worldTransform) );
			oldUp			= up;

			var targetUp	= Vector3.TransformNormal( Vector3.Up, Matrix.Invert(worldTransform) );

			var torque		= cfg.StabK * Vector3.Cross( targetUp, Vector3.Up )
							+ cfg.StabD * (float)Math.Exp( -localOmega.LengthSquared() ) * localOmega;

			var e1			= 0.15f * ( new Vector3( -1, 0, +1 ).Normalized() );
			var e2			= 0.15f * ( new Vector3( +1, 0, +1 ).Normalized() );

			var f1			= Vector3.Dot( e2, torque ); 
			var f2			= Vector3.Dot( e1, torque ); 

			//var testTorq	= e1 * f1 + e2 * f2 + e3 * f3 + e4 * f4;

			float t1		= MathHelper.Clamp( thrust + f1, 0, 1 );
			float t2		= MathHelper.Clamp( thrust + f2, 0, 1 );
			float t3		= MathHelper.Clamp( thrust - f1, 0, 1 );
			float t4		= MathHelper.Clamp( thrust - f2, 0, 1 );//*/


			dr.DrawVector( worldTransform.Translation, Vector3.TransformNormal( torque	 , worldTransform  ), Color.Red );
			dr.DrawVector( worldTransform.Translation, Vector3.TransformNormal( omega	 , worldTransform  ), Color.Yellow );
			dr.DrawVector( worldTransform.Translation, Vector3.TransformNormal( targetUp , worldTransform  ), Color.Blue );
			dr.DrawVector( worldTransform.Translation, Vector3.TransformNormal( up		 , Matrix.Identity ), Color.Green );

			/*if (logWriter!=null) {
			    if (firstLine) {
			        logWriter.WriteLine("# TX TY TZ   UpX UpY UpZ");
			        firstLine = false;	
			    }

			    logWriter.WriteLine("{0} {1} {2}   {3} {4} {5}   {6} {7} {8}   {9}   {10} {11} {12} {13}   {14} {15} {16} {17}",	
			        worldTransform.Translation.X, worldTransform.Translation.Y, worldTransform.Translation.Z, 
			        yaw, pitch, roll,
			        vYaw, vPitch, vRoll,
			        thrust, 
			        t1, t2, t3, t4, 
			        Rotor1, Rotor2, Rotor3, Rotor4 );
			} */
			

		#endif

			Rotor1	=	(int) ( 50 + 80 * t1 );
			Rotor2	=	(int) ( 50 + 80 * t2 );
			Rotor3	=	(int) ( 50 + 80 * t3 );
			Rotor4	=	(int) ( 50 + 80 * t4 );
			
			bool  engine	= thrust > 0.02f;

			if (!engine) {
				Rotor1 = Rotor2 = Rotor3 = Rotor4 = 0;
			}

			if (logWriter!=null) {
			    if (firstLine) {
			        logWriter.WriteLine("# X Y Z   Yaw Pitch Roll   VYaw VPitch VRoll   Thrust   T1 T2 T3 T4   PWM1 PWM2 PWM3 PWM4");
			        firstLine = false;	
			    }

			    logWriter.WriteLine("{0} {1} {2}   {3} {4} {5}   {6} {7} {8}   {9}   {10} {11} {12} {13}   {14} {15} {16} {17}",	
			        worldTransform.Translation.X, worldTransform.Translation.Y, worldTransform.Translation.Z, 
			        yaw, pitch, roll,
			        vYaw, vPitch, vRoll,
			        thrust, 
			        t1, t2, t3, t4, 
			        Rotor1, Rotor2, Rotor3, Rotor4 );
			}
		}



		public void UpdateFromTracker ()
		{
			if ( TrackObject && world.Tracker!=null ) {
				var frame = world.Tracker.GetFrame();
				var subject = frame[ Name ];

				if (subject!=null) {
					var segment = subject[ Name ];

					var xform = segment.Transform;

					xform.Translation = xform.Translation * new Vector3( 0.01f, 0.01f, 0.01f );

					if (xform.Right.Length()<0.001f || xform.Forward.Length()<0.001f || xform.Up.Length()<0.001f ) {
						TrackObject = false;
					}

					lock (this) {
						worldTransform = xform;
					}
				}
			}
		}


		Thread takeofThread;

		public void RunTakeoffThrustEstimation() {
			takeofThread = new Thread( TakeoffThrustEstimation );
			takeofThread.Start();
		}


		public void TakeoffThrustEstimation ()
		{
			float currentHeight = Vector3.Dot( worldTransform.Translation, Vector3.Up );

			Console.WriteLine("Takeoff thrust estimation procedure");

			bool raising = true;

			lock (this) {
				Rotor1 = Rotor2 = Rotor3 = Rotor4 = 12;
			}

			Console.WriteLine("Thrust set to zero");
			Thread.Sleep(2000);
			

			while (true) {

				Thread.Sleep(50);

				lock (this) {
					
					Rotor1 = Rotor1 + ( raising ? 1 : -50 );
					Rotor2 = Rotor1;
					Rotor3 = Rotor1;
					Rotor4 = Rotor1;

					Console.WriteLine("Thrust : {0}", Rotor1);

					float height = Vector3.Dot( worldTransform.Translation, Vector3.Up );
					if ( (height - currentHeight) > 0.02 && raising ) {
						raising = false;
						Console.WriteLine( "Takeoff thrust detected : {0}", Rotor1 );
					}

					if (Rotor1>179) {
						raising = false;
						Console.WriteLine( "Unable to takeoff!" );
					}
				}

				if (Rotor1<=10) {
					break;
				}
			}

			Console.WriteLine("Done");
		}



		public void Draw ( float dt, Matrix view, Matrix proj )
		{
			float	rot1 = 3121 % 360;
			float	rot2 = 1135 % 360;
			float	rot3 = 6546 % 360;
			float	rot4 = 1231 % 360;

			Vector3	arm1	=	(float)Math.Sqrt(2)/2 * ( Vector3.Right + Vector3.Forward  );
			Vector3	arm2	=	(float)Math.Sqrt(2)/2 * ( Vector3.Right + Vector3.Backward );
			Vector3	arm3	=	(float)Math.Sqrt(2)/2 * ( Vector3.Left  + Vector3.Backward );
			Vector3	arm4	=	(float)Math.Sqrt(2)/2 * ( Vector3.Left  + Vector3.Forward  );

			SimulatorGame.DrawModel( frame, worldTransform, view, proj );

			SimulatorGame.DrawModel( propellerA, Matrix.CreateRotationY( rot1/3.14f) * Matrix.CreateTranslation( arm1 * 0.15f ) * worldTransform, view, proj );
			SimulatorGame.DrawModel( propellerB, Matrix.CreateRotationY(-rot2/3.14f) * Matrix.CreateTranslation( arm2 * 0.15f ) * worldTransform, view, proj );
			SimulatorGame.DrawModel( propellerA, Matrix.CreateRotationY( rot3/3.14f) * Matrix.CreateTranslation( arm3 * 0.15f ) * worldTransform, view, proj );
			SimulatorGame.DrawModel( propellerB, Matrix.CreateRotationY(-rot4/3.14f) * Matrix.CreateTranslation( arm4 * 0.15f ) * worldTransform, view, proj );
		}
		


		Thread  commThread;
		bool	commAbortRequest = false;
		SerialPort 	port;



		public void RunCommunicationProtocol ()
		{
			commAbortRequest = false;
			commThread = new Thread( CommunicationThreadFunc );
			commThread.Start();
		}



		public void StopCommunicationProtocol ()
		{
			commAbortRequest = true;
			//commThread.
		}



		void CommunicationThreadFunc ()
		{
			bool engineCut = false;

			Console.WriteLine("Communication thread started.");

			try {

				var cfg = game.GetService<Settings>().Configuration;

				Console.WriteLine("Port " + cfg.Port);

				port = new SerialPort( cfg.Port, cfg.BaudRate );
				port.ReadTimeout = SerialPort.InfiniteTimeout;
				port.Open();

				Console.WriteLine("Opened.");

				//byte[] buf = new byte[7];
				string s;

				while (!commAbortRequest) {

					if ( GamePad.GetState(0).IsButtonDown( Buttons.B ) ) {
						engineCut = true;
					}
					if ( GamePad.GetState(0).IsButtonDown( Buttons.A ) ) {
						engineCut = false;
					}

					if ( game.GetService<World>().Tracker==null || !TrackObject ) {
						engineCut = true;
					}

					lock (this) {
						if (engineCut) {
							Rotor1 = Rotor2 = Rotor3 = Rotor4 = 10;
						}
						//if (ThrustTest>0) {
						//    Rotor1 = Rotor2 = Rotor3 = Rotor4 = ThrustTest;
						//}
						Rotor1 = Math.Min(180, Math.Max( Rotor1, 10 ));
						Rotor2 = Math.Min(180, Math.Max( Rotor2, 10 ));
						Rotor3 = Math.Min(180, Math.Max( Rotor3, 10 ));
						Rotor4 = Math.Min(180, Math.Max( Rotor4, 10 ));
						s = string.Format("X{0,2:X2}{1,2:X2}{2,2:X2}{3,2:X2}\n", (byte)Rotor1, (byte)Rotor2, (byte)Rotor3, (byte)Rotor4 );
					}

					port.Write( s );
					Thread.Sleep(30);
				}

			} catch (Exception ex) {
				Forms.MessageBox.Show( ex.Message );
				commAbortRequest = true;
				return;
			}

			Console.WriteLine("Communication thread aborted.");
		}

		//void ApplyForceLL ( ref Box box, float dt, Vector3 point, Vector3 localForce )
		//{
		//    var m = box.WorldTransform;
		//    var p = Vector3.Transform(point, m);
		//    var f = Vector3.TransformNormal(localForce * dt, m);

		//    float groundEffect = 1 + 0.4f * (float)Math.Exp( -2*p.Y ) / MathHelper.E;

		//    box.ApplyImpulse( p, f * groundEffect );
		//}


		//void ApplyForceLG ( ref Box box, float dt, Vector3 point, Vector3 globalForce )
		//{
		//    var m = box.WorldTransform;
		//    box.ApplyImpulse( Vector3.Transform(point, m), globalForce * dt );
		//}
	}
}
