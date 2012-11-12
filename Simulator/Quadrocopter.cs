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

		SerialPort 	port;
		public StreamWriter	logWriter = null;
		public bool firstLine = false;

		bool engineCut = false;


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

		public Box	box;

		public float thrustTest2 = 0;
		public Vector3 oldLocalOmega = Vector3.Zero;
		Vector3 oldPosition;
		Vector3	oldVelocity;

		bool	getTakeoffThrust = true;
		Vector3	landingPoint;
		float	takeoffThrust;

		bool	targetPointMode = false;

		Vector3	oldForward = Vector3.Forward;

		/// <summary>
		/// Updates intire state of the quad-rotor
		/// </summary>
		/// <param name="dt"></param>
		public void Update ( float dt )
		{
			var ds	= game.GetService<DebugStrings>();
			var cfg = game.GetService<Settings>().Configuration;
			var dr	= game.GetService<DebugRender>();

			if (port==null) {
				ds.Add( string.Format( "{0} disconnected", Name ), Color.Red );
			} else {
				ds.Add( string.Format( "{0} connected on {1}", Name, port.PortName ), Color.Yellow );
			}

			if (engineCut) {
				ds.Add( "Engine cut", Color.Red );
			}

			if (getTakeoffThrust) {
				ds.Add( "Takeoff thrust required", Color.Red );
			} else {
				ds.Add( "Takeoff thrust " + takeoffThrust.ToString(), Color.White );
			}

			if (targetPointMode) {
				ds.Add( "Target point", Color.LightYellow );
			}

			//
			//	Tracker/Simulation update :
			//
			UpdateFromTracker(dt);

			//
			//	Stabilization :
			//
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

			Vector3	position		=	worldTransform.Translation;
			Vector3	velocity		=	( position - oldPosition ) / dt;
			Vector3 acceleration	=	( velocity - oldVelocity ) / dt;
			oldVelocity				=	velocity;
			oldPosition				=	position;

			var gps = GamePad.GetState(0);

			if (gps.DPad.Right == ButtonState.Pressed)				{ cfg.TrimRoll  -= 0.01f; }
			if (gps.DPad.Left  == ButtonState.Pressed)				{ cfg.TrimRoll  += 0.01f; }
			if (gps.DPad.Up    == ButtonState.Pressed)				{ cfg.TrimPitch -= 0.01f; }
			if (gps.DPad.Down  == ButtonState.Pressed)				{ cfg.TrimPitch += 0.01f; }
			if (gps.Buttons.LeftShoulder   == ButtonState.Pressed)	{ cfg.Yaw -= 0.01f;		  }
			if (gps.Buttons.RightShoulder  == ButtonState.Pressed)	{ cfg.Yaw += 0.01f;		  }
			if (gps.Buttons.Y  == ButtonState.Pressed)				{ targetPointMode = true; }
			if (gps.Buttons.X  == ButtonState.Pressed)				{ targetPointMode = false	; }
			if (gps.Buttons.B  == ButtonState.Pressed)				{ targetPointMode = false	; }


			float thrust	= gps.Triggers.Right;
			var up			= worldTransform.Up;
			var forward		= worldTransform.Forward.Flattern(Vector3.Up);

			var aimUpVector	= Vector3.Up;	

			var addTorque	= Vector3.Zero;

			if ( targetPointMode ) {

				if (cfg.TargetVector==Vector3.Zero) {
					cfg.TargetVector = Vector3.Forward;
				}
				cfg.TargetVector = cfg.TargetVector.Flattern( Vector3.Up ).Normalized();

				dr.DrawRing( cfg.TargetPoint, 0.3f, Color.Yellow );
				dr.DrawVector( cfg.TargetPoint, cfg.TargetVector * 1.33f, Color.Yellow );

				var	offset	=	position - cfg.TargetPoint;

				offset.ClampLength(1);

				thrust		=	cfg.TakeoffThrust + ( offset.Y * cfg.StabAltitudeK  + velocity.Y * cfg.StabAltitudeD );

				var	upShift	=	offset * cfg.StabPointK 
							+	velocity * ( cfg.StabPointD )
							+	cfg.StabPointI * acceleration;

				/*if (velocity.Length()>5) {
					upShift =	Vector3.Zero;
				} */

				aimUpVector +=	upShift.Flattern(Vector3.Up);//.ClampLength(0.7f);
				//thrust *= aimUpVector.Length();
				aimUpVector.Normalize();

				thrust /= MathHelper.Clamp(up.Y, 0.6f, 1);
				//addTorque.Z	=	-upShift.X;
				//addTorque.X	=	 upShift.Z;
				//addTorque	=	Vector3.TransformNormal( addTorque, Matrix.Invert(worldTransform) );
			}


			if (up.Y<0) {
				thrust = 0;
			}

			var omega		= Vector3.Cross( up, ( up - oldUp ) / dt );
			var localOmega	= Vector3.TransformNormal( omega, Matrix.Invert(worldTransform) );
			var localOmegaA	= (oldLocalOmega - localOmega) / (dt+0.00001f);
			oldLocalOmega	= localOmega;
			oldUp			= up;

			var targetUp	= Vector3.TransformNormal( aimUpVector, Matrix.Invert(worldTransform) );



			var torque		= (Vector3.Cross( targetUp, Vector3.Up )) * cfg.StabK
							+ (localOmega) * cfg.StabD
							+ addTorque;


			var e1			= 0.15f * ( new Vector3( -1, 0, +1 ).Normalized() );
			var e2			= 0.15f * ( new Vector3( +1, 0, +1 ).Normalized() );

			var f1			= Vector3.Dot( e2, torque ); 
			var f2			= Vector3.Dot( e1, torque ); 

			//var testTorq	= e1 * f1 + e2 * f2 + e3 * f3 + e4 * f4;
			float ctrlRoll	= ( cfg.ControlFactor    * gps.ThumbSticks.Right.X + cfg.TrimRoll  );
			float ctrlPitch	= ( cfg.ControlFactor    * gps.ThumbSticks.Right.Y + cfg.TrimPitch );
			float ctrlYaw	= ( cfg.ControlFactorYaw * gps.ThumbSticks.Left.X  + cfg.TrimYaw   );

			if ( targetPointMode ) {
				float yawVel		=	Vector3.Dot( Vector3.Up, Vector3.Cross( forward, ( forward - oldForward ) / dt ) );

				var targetForward	=	Vector3.TransformNormal( cfg.TargetVector, Matrix.Invert(worldTransform) );

				float yawDist		=	Vector3.Dot( Vector3.Up, Vector3.Cross( targetForward, Vector3.Forward ) );

				ctrlYaw				=	cfg.StabYawK * yawDist + cfg.StabYawD * yawVel;
			}
			oldForward = forward;

			float t1		= MathHelper.Clamp( thrust + f1, 0, 1 ) - ctrlRoll - ctrlPitch + ctrlYaw;
			float t2		= MathHelper.Clamp( thrust + f2, 0, 1 ) - ctrlRoll + ctrlPitch - ctrlYaw;
			float t3		= MathHelper.Clamp( thrust - f1, 0, 1 ) + ctrlRoll + ctrlPitch + ctrlYaw;
			float t4		= MathHelper.Clamp( thrust - f2, 0, 1 ) + ctrlRoll - ctrlPitch - ctrlYaw;//*/


			dr.DrawVector( worldTransform.Translation, Vector3.TransformNormal( torque	 , worldTransform  ), Color.Red );
			dr.DrawVector( worldTransform.Translation, Vector3.TransformNormal( omega	 , worldTransform  ), Color.Yellow );
			dr.DrawVector( worldTransform.Translation, Vector3.TransformNormal( targetUp , worldTransform  ), Color.Blue );
			dr.DrawVector( worldTransform.Translation, Vector3.TransformNormal( up		 , Matrix.Identity ), Color.Green );

			dr.DrawVector( worldTransform.Translation, velocity,		Color.CornflowerBlue );
			dr.DrawVector( worldTransform.Translation, acceleration,	Color.DarkGoldenrod );

			Rotor1	=	(int) ( 50 + 80 * t1 );
			Rotor2	=	(int) ( 50 + 80 * t2 );
			Rotor3	=	(int) ( 50 + 80 * t3 );
			Rotor4	=	(int) ( 50 + 80 * t4 );
			
			bool  engine	= thrust > 0.02f;

			ds.Add(string.Format("Engine: {0,3} {1,3} {2,3} {3,3}",  Rotor1, Rotor2, Rotor3, Rotor4) );

			if (!engine) {
				Rotor1 = Rotor2 = Rotor3 = Rotor4 = 0;
			}


			//
			//	LOG :
			//
			if (logWriter!=null) {
			    if (firstLine) {
			        logWriter.WriteLine("# X Y Z   Yaw Pitch Roll   VYaw VPitch VRoll   Thrust   T1 T2 T3 T4   F1 F2 F3 F4");
			        firstLine = false;	
			    }

			    logWriter.WriteLine("{0} {1} {2}   {3} {4} {5}   {6} {7} {8}   {9}   {10} {11} {12} {13}   {14} {15} {16} {17}",	
			        worldTransform.Translation.X, worldTransform.Translation.Y, worldTransform.Translation.Z, 
			        yaw, pitch, roll,
			        vYaw, vPitch, vRoll,
			        thrust, 
			        t1, t2, t3, t4, 
			        F1, F2, F3, F4 );
			}

			//
			//	BT communication :
			//
			if (port!=null) {
				try {
					if ( GamePad.GetState(0).IsButtonDown( Buttons.B ) ) {
						engineCut = true;
					}
					if ( GamePad.GetState(0).IsButtonDown( Buttons.A ) ) {
						engineCut = false;
					}

					if ( game.GetService<World>().Tracker==null || !TrackObject ) {
						engineCut = true;
					}

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
					string s = string.Format("X{0,2:X2}{1,2:X2}{2,2:X2}{3,2:X2}\n", (byte)Rotor1, (byte)Rotor2, (byte)Rotor3, (byte)Rotor4 );

					port.Write( s );

				} catch ( Exception e ) {
					StopCommunicationProtocol();
				}
			}
		}



		Queue<int> rotor1Delay = new Queue<int>();
		Queue<int> rotor2Delay = new Queue<int>();
		Queue<int> rotor3Delay = new Queue<int>();
		Queue<int> rotor4Delay = new Queue<int>();

		float angle1 = 0, angle2 = 0, angle3 = 0, angle4 = 0;
		float F1 = 0, F2 = 0, F3 = 0, F4 = 0;

		/// <summary>
		/// Updates kinematic state of real or virtual quad-rotor
		/// </summary>
		/// <param name="dt"></param>
		public void UpdateFromTracker (float dt)
		{
			var cfg = game.GetService<Settings>().Configuration;
			var ds	= game.GetService<DebugStrings>();

			if ( TrackObject && world.Tracker!=null ) {

				ds.Add("Tracked : " + Name, Color.Yellow );

				var frame = world.Tracker.GetFrame();
				if (frame==null) {
					return;
				}
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

			if (cfg.UseSimulation) {

				ds.Add("Simulated : " + Name, Color.Yellow );

				if (box==null) {
					box = new Box(Vector3.Zero, 0.4f, 0.07f, 0.4f, 0.580f );
					box.PositionUpdateMode = PositionUpdateMode.Continuous;
					world.Space.Add( box );
					world.Drawer.Add( box );
					box.AngularDamping = 0;
					box.LinearDamping = 0;
				}

				var gps = GamePad.GetState(0);

				if (gps.Buttons.X==ButtonState.Pressed) {
					box.WorldTransform = Matrix.Identity;
					box.LinearVelocity = Vector3.Zero;
					box.AngularVelocity = Vector3.Zero;
				}

				rotor1Delay.Enqueue( Rotor1 );
				rotor2Delay.Enqueue( Rotor2 );
				rotor3Delay.Enqueue( Rotor3 );
				rotor4Delay.Enqueue( Rotor4 );

				int r1,r2,r3,r4;
				r1 = r2 = r3 = r4 = 0;

				while (rotor1Delay.Count>cfg.DelayFrames) { r1 = rotor1Delay.Dequeue();	}
				while (rotor2Delay.Count>cfg.DelayFrames) { r2 = rotor2Delay.Dequeue();	}
				while (rotor3Delay.Count>cfg.DelayFrames) { r3 = rotor3Delay.Dequeue();	}
				while (rotor4Delay.Count>cfg.DelayFrames) { r4 = rotor4Delay.Dequeue();	}

				float	f1	=	MathHelper.Clamp((r1 - 48) / 80.0f * ( 2.5f + 0.05f ), 0, float.MaxValue);
				float	f2	=	MathHelper.Clamp((r2 - 51) / 81.0f * ( 2.5f - 0.02f ), 0, float.MaxValue);
				float	f3	=	MathHelper.Clamp((r3 - 53) / 79.0f * ( 2.5f + 0.04f ), 0, float.MaxValue);
				float	f4	=	MathHelper.Clamp((r4 - 49) / 82.0f * ( 2.5f - 0.03f ), 0, float.MaxValue);

				F1	=	MathHelper.Lerp( F1, f1, cfg.MotorLatency );
				F2	=	MathHelper.Lerp( F2, f2, cfg.MotorLatency );
				F3	=	MathHelper.Lerp( F3, f3, cfg.MotorLatency );
				F4	=	MathHelper.Lerp( F4, f4, cfg.MotorLatency );

				angle1	+=	(float)Math.Sqrt( Math.Max(0, F1) );
				angle2	+=	(float)Math.Sqrt( Math.Max(0, F2) );
				angle3	+=	(float)Math.Sqrt( Math.Max(0, F3) );
				angle4	+=	(float)Math.Sqrt( Math.Max(0, F4) );

				ApplyForceLL( ref box, dt, new Vector3( 1, 0,-1 ) * 0.15f, new Vector3( 0.02f, 1.0f, 0.03f).Normalized() * F1 ); 
				ApplyForceLL( ref box, dt, new Vector3( 1, 0, 1 ) * 0.15f, new Vector3(-0.03f, 1.0f,-0.04f).Normalized() * F2 ); 
				ApplyForceLL( ref box, dt, new Vector3(-1, 0, 1 ) * 0.15f, new Vector3( 0.01f, 1.0f,-0.02f).Normalized() * F3 ); 
				ApplyForceLL( ref box, dt, new Vector3(-1, 0,-1 ) * 0.15f, new Vector3(-0.02f, 1.0f, 0.01f).Normalized() * F4 ); 

				ApplyLocalTorque( ref box, dt,  Vector3.Up * F1 * 0.02f );
				ApplyLocalTorque( ref box, dt, -Vector3.Up * F2 * 0.02f );
				ApplyLocalTorque( ref box, dt,  Vector3.Up * F3 * 0.02f );
				ApplyLocalTorque( ref box, dt, -Vector3.Up * F4 * 0.02f );


				worldTransform	=	
					Matrix.CreateFromYawPitchRoll( 0.033f, 0.021f, -0.029f ) * 
					Matrix.CreateTranslation( 0.01f, 0.004f, -0.009f ) * 
					box.WorldTransform;
			}

		}



		float	rot1 = 3121 % 360;
		float	rot2 = 1135 % 360;
		float	rot3 = 6546 % 360;
		float	rot4 = 1231 % 360;

		public void Draw ( float dt, Matrix view, Matrix proj )
		{
			rot1 += Rotor1;
			rot1 += Rotor2;
			rot1 += Rotor3;
			rot1 += Rotor4;

			Vector3	arm1	=	(float)Math.Sqrt(2)/2 * ( Vector3.Right + Vector3.Forward  );
			Vector3	arm2	=	(float)Math.Sqrt(2)/2 * ( Vector3.Right + Vector3.Backward );
			Vector3	arm3	=	(float)Math.Sqrt(2)/2 * ( Vector3.Left  + Vector3.Backward );
			Vector3	arm4	=	(float)Math.Sqrt(2)/2 * ( Vector3.Left  + Vector3.Forward  );

			SimulatorGame.DrawModel( frame, worldTransform, view, proj );

			SimulatorGame.DrawModel( propellerA, Matrix.CreateRotationY( angle1) * Matrix.CreateTranslation( arm1 * 0.15f ) * worldTransform, view, proj );
			SimulatorGame.DrawModel( propellerB, Matrix.CreateRotationY(-angle2) * Matrix.CreateTranslation( arm2 * 0.15f ) * worldTransform, view, proj );
			SimulatorGame.DrawModel( propellerA, Matrix.CreateRotationY( angle3) * Matrix.CreateTranslation( arm3 * 0.15f ) * worldTransform, view, proj );
			SimulatorGame.DrawModel( propellerB, Matrix.CreateRotationY(-angle4) * Matrix.CreateTranslation( arm4 * 0.15f ) * worldTransform, view, proj );
		}
		



		public void RunCommunicationProtocol ()
		{
			var cfg = game.GetService<Settings>().Configuration;

			Console.WriteLine("Port " + cfg.Port);

			port = new SerialPort( cfg.Port, cfg.BaudRate );
			port.ReadTimeout = SerialPort.InfiniteTimeout;
			port.Open();

			Console.WriteLine("Opened.");
		}



		public void StopCommunicationProtocol ()
		{
			port.Close();
			port.Dispose();
			port = null;
		}


		void ApplyLocalTorque ( ref Box box, float dt, Vector3 torque )
		{
			ApplyForceLL( ref box, dt, Vector3.Right, Vector3.Cross( Vector3.Right, torque ) );
			ApplyForceLL( ref box, dt, Vector3.Left,  Vector3.Cross( Vector3.Left,  torque ) );
		}


		void ApplyForceLL ( ref Box box, float dt, Vector3 point, Vector3 localForce )
		{
		    var m = box.WorldTransform;
		    var p = Vector3.Transform(point, m);
		    var f = Vector3.TransformNormal(localForce * dt, m);

		    float groundEffect = 1;// + 0.4f * (float)Math.Exp( -2*p.Y ) / MathHelper.E;

		    box.ApplyImpulse( p, f * groundEffect );
		}


		void ApplyForceLG ( ref Box box, float dt, Vector3 point, Vector3 globalForce )
		{
		    var m = box.WorldTransform;
		    box.ApplyImpulse( Vector3.Transform(point, m), globalForce * dt );
		}
	}
}
