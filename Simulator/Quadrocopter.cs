using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Audio;
using Microsoft.Xna.Framework.Content;
using Microsoft.Xna.Framework.GamerServices;
using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework.Input;
using Microsoft.Xna.Framework.Media;
using BEPUphysics;
using BEPUphysics.Collidables;
using BEPUphysics.DataStructures;
using BEPUphysics.CollisionShapes.ConvexShapes;
using BEPUphysicsDrawer.Models;
using BEPUphysics.Entities.Prefabs;
using BEPUphysics.EntityStateManagement;
using BEPUphysics.PositionUpdating;
//using ViconDataStreamSDK.DotNET;
using System.ComponentModel;



namespace Simulator {
	public class Quadrocopter {

		[Category("Dynamics")]	public float	MaxRPM				{ set; get; }
		[Category("Dynamics")]	public float	MaxRotorThrust		{ set; get; }
		[Category("Dynamics")]	public float	MaxRotorTorque		{ set; get; }
		[Category("Dynamics")]	public float	Mass				{ set; get; }
		[Category("Dynamics")]	public float	AirResistance		{ set; get; }
		[Category("Dynamics")]	public bool		TrackMotion			{ set; get; }
		[Category("Dynamics")]	public string	TrackingObjectName	{ set; get; }
		[Category("Dynamics")]	public float	ArmLength			{ set; get; }
		[Category("Dynamics")]	public Vector3	LinearSize			{ set; get; }

		Model	frame;
		Model	propellerA;
		Model	propellerB;


		Game	game;
		World	world;

		Box		box;

		float	rpm1 = 0;
		float	rpm2 = 0;
		float	rpm3 = 0;
		float	rpm4 = 0;
		float	trpm1 = 0;
		float	trpm2 = 0;
		float	trpm3 = 0;
		float	trpm4 = 0;

		float	rot1 = 0;
		float	rot2 = 0;
		float	rot3 = 0;
		float	rot4 = 0;

		Vector3	arm1	=	(float)Math.Sqrt(2)/2 * ( Vector3.Right + Vector3.Forward  );
		Vector3	arm2	=	(float)Math.Sqrt(2)/2 * ( Vector3.Right + Vector3.Backward );
		Vector3	arm3	=	(float)Math.Sqrt(2)/2 * ( Vector3.Left  + Vector3.Backward );
		Vector3	arm4	=	(float)Math.Sqrt(2)/2 * ( Vector3.Left  + Vector3.Forward  );

		


		public Vector3 Position { get { return box.WorldTransform.Translation; } }
		public Matrix  Transform { get { return box.WorldTransform; } }
																					   

		public Quadrocopter( Game game, World world )
		{
			MaxRPM			=	20000;	//	rotations per minute
			MaxRotorThrust	=	2.75f;	//	1100 gramms / 4 motors
			MaxRotorTorque	=	1.00f;	//	-- check!
			Mass			=	0.70f;	//	700 gramms
			AirResistance	=	0.10f;	//	
			ArmLength		=	0.15f;
			LinearSize		=	new Vector3( 0.4f, 0.10f, 0.4f );

			TrackingObjectName	=	"Quadrocopter";


			this.game		=	game;
			this.world		=	world;

			frame			=	game.Content.Load<Model>( @"scenes\quadFrame" );
			propellerA		=	game.Content.Load<Model>( @"scenes\propellerA" );
			propellerB		=	game.Content.Load<Model>( @"scenes\propellerB" );

			box	=	new Box( Vector3.Up * LinearSize.Y/2, LinearSize.X, LinearSize.Y, LinearSize.Z, Mass );
			box.AngularDamping = 0.0f;
			box.LinearDamping  = 0.0f;
			world.Space.Add( box );
			world.Drawer.Add( box );

			box.Material.KineticFriction	=	0.7f;
			box.Material.StaticFriction		=	0.7f;
			box.Material.Bounciness			=	0.15f;

			box.PositionUpdateMode = PositionUpdateMode.Continuous;
		}



		public void Update ( float dt )
		{
			float rpm2thrust = MaxRotorThrust / ( MaxRPM * MaxRPM );
			float rpm2torque = MaxRotorTorque / ( MaxRPM * MaxRPM );

			DirectControl();
			UpdateFromTracker();

			trpm1 = MathHelper.Clamp( trpm1, 0, MaxRPM );
			trpm2 = MathHelper.Clamp( trpm2, 0, MaxRPM );
			trpm3 = MathHelper.Clamp( trpm3, 0, MaxRPM );
			trpm4 = MathHelper.Clamp( trpm4, 0, MaxRPM );

			rpm1 = MathHelper.Lerp( rpm1, trpm1, 0.2f );
			rpm2 = MathHelper.Lerp( rpm2, trpm2, 0.2f );
			rpm3 = MathHelper.Lerp( rpm3, trpm3, 0.2f );
			rpm4 = MathHelper.Lerp( rpm4, trpm4, 0.2f );

			ApplyForceLL( ref box, dt, arm1 * ArmLength, Vector3.Up * rpm2thrust * rpm1 * rpm1 );
			ApplyForceLL( ref box, dt, arm2 * ArmLength, Vector3.Up * rpm2thrust * rpm2 * rpm2 );
			ApplyForceLL( ref box, dt, arm3 * ArmLength, Vector3.Up * rpm2thrust * rpm3 * rpm3 );
			ApplyForceLL( ref box, dt, arm4 * ArmLength, Vector3.Up * rpm2thrust * rpm4 * rpm4 );

			ApplyForceLL( ref box, dt, arm1, arm2 * rpm2torque * rpm1 * rpm1 );
			ApplyForceLL( ref box, dt, arm2, arm1 * rpm2torque * rpm2 * rpm2 );
			ApplyForceLL( ref box, dt, arm3, arm4 * rpm2torque * rpm3 * rpm3 );
			ApplyForceLL( ref box, dt, arm4, arm3 * rpm2torque * rpm4 * rpm4 );

			rot1 += rpm1 * dt / 200.0f;
			rot2 += rpm2 * dt / 200.0f;
			rot3 += rpm3 * dt / 200.0f;
			rot4 += rpm4 * dt / 200.0f;

			/*box.AngularMomentum = box.WorldTransform.Up * rpm2torque * rpm1 * rpm1;
			box.AngularMomentum = box.WorldTransform.Up * rpm2torque * rpm2 * rpm2;
			box.AngularMomentum = box.WorldTransform.Up * rpm2torque * rpm3 * rpm3;
			box.AngularMomentum = box.WorldTransform.Up * rpm2torque * rpm4 * rpm4;*/
		}



		public void UpdateFromTracker ()
		{
			if (TrackMotion && world.Tracker!=null) {
				var frame = world.Tracker.GetFrame();
				var subject = frame[ TrackingObjectName ];
				if (subject!=null) {
					var segment = subject[ TrackingObjectName ];

					var xform = segment.Transform;

					xform.Translation = xform.Translation * new Vector3( 0.01f, 0.01f, 0.01f );

					box.WorldTransform = xform;
					
					box.LinearVelocity = Vector3.Zero;
				}

			}
		}


		public void DirectControl ()
		{
			var ks = Keyboard.GetState();
			var gps = GamePad.GetState( 0 );

			if (gps.IsButtonDown( Buttons.B ) || ks.IsKeyDown( Keys.B ) ) {
				box.WorldTransform	=	Matrix.CreateTranslation( Vector3.Up * LinearSize.Y/2 );
			}

			float avgRpm = ( gps.Triggers.Left + world.Mouse3DTranslation.Y / 1.0f ) * MaxRPM;

			if (ks.IsKeyDown(Keys.NumPad8)) avgRpm = MaxRPM;

			trpm1 = avgRpm;
			trpm2 = avgRpm;
			trpm3 = avgRpm;
			trpm4 = avgRpm;

			var m3drot =  world.Mouse3DRotationAxis * world.Mouse3DRotationAngle / 80.0f;
			
			trpm1 -= avgRpm * ( gps.ThumbSticks.Right.X  - m3drot.Z ) / 8;
			trpm2 -= avgRpm * ( gps.ThumbSticks.Right.X  - m3drot.Z ) / 8;
			trpm3 += avgRpm * ( gps.ThumbSticks.Right.X  - m3drot.Z ) / 8;
			trpm4 += avgRpm * ( gps.ThumbSticks.Right.X  - m3drot.Z ) / 8;
							  
			trpm1 -= avgRpm * ( gps.ThumbSticks.Right.Y  - m3drot.X ) / 8;
			trpm2 += avgRpm * ( gps.ThumbSticks.Right.Y  - m3drot.X ) / 8;
			trpm3 += avgRpm * ( gps.ThumbSticks.Right.Y  - m3drot.X ) / 8;
			trpm4 -= avgRpm * ( gps.ThumbSticks.Right.Y  - m3drot.X ) / 8;


			trpm1 += avgRpm * ( gps.ThumbSticks.Left.X   + m3drot.Y )/ 8;
			trpm2 -= avgRpm * ( gps.ThumbSticks.Left.X   + m3drot.Y )/ 8;
			trpm3 += avgRpm * ( gps.ThumbSticks.Left.X   + m3drot.Y )/ 8;
			trpm4 -= avgRpm * ( gps.ThumbSticks.Left.X   + m3drot.Y )/ 8;
		}



		public void Draw ( float dt, Matrix view, Matrix proj )
		{
			SimulatorGame.DrawModel( frame, box.WorldTransform, view, proj );

			SimulatorGame.DrawModel( propellerA, Matrix.CreateRotationY( rot1/3.14f) * Matrix.CreateTranslation( arm1 * ArmLength ) * box.WorldTransform, view, proj );
			SimulatorGame.DrawModel( propellerB, Matrix.CreateRotationY(-rot2/3.14f) * Matrix.CreateTranslation( arm2 * ArmLength ) * box.WorldTransform, view, proj );
			SimulatorGame.DrawModel( propellerA, Matrix.CreateRotationY( rot3/3.14f) * Matrix.CreateTranslation( arm3 * ArmLength ) * box.WorldTransform, view, proj );
			SimulatorGame.DrawModel( propellerB, Matrix.CreateRotationY(-rot4/3.14f) * Matrix.CreateTranslation( arm4 * ArmLength ) * box.WorldTransform, view, proj );
		}
		

		void ApplyForceLL ( ref Box box, float dt, Vector3 point, Vector3 localForce )
		{
			var m = box.WorldTransform;
			var p = Vector3.Transform(point, m);
			var f = Vector3.TransformNormal(localForce * dt, m);

			float groundEffect = 1 + 0.4f * (float)Math.Exp( -2*p.Y ) / MathHelper.E;

			box.ApplyImpulse( p, f * groundEffect );
		}


		void ApplyForceLG ( ref Box box, float dt, Vector3 point, Vector3 globalForce )
		{
			var m = box.WorldTransform;
			box.ApplyImpulse( Vector3.Transform(point, m), globalForce * dt );
		}
	}
}
