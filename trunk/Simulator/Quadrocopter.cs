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



namespace Simulator {
	public class Quadrocopter {

		public float	MaxRPM				{ set; get; }
		public float	MaxRotorThrust		{ set; get; }
		public float	MaxRotorTorque		{ set; get; }
		public float	Mass				{ set; get; }
		public float	AirResistance		{ set; get; }
		public bool		TrackMotion			{ set; get; }
		public string	TrackingObjectName	{ set; get; }
		public float	ArmLength			{ set; get; }
		public Vector3	LinearSize			{ set; get; }

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

		Vector3	arm1	=	(float)Math.Sqrt(2)/2 * ( Vector3.Right + Vector3.Forward  );
		Vector3	arm2	=	(float)Math.Sqrt(2)/2 * ( Vector3.Right + Vector3.Backward );
		Vector3	arm3	=	(float)Math.Sqrt(2)/2 * ( Vector3.Left  + Vector3.Backward );
		Vector3	arm4	=	(float)Math.Sqrt(2)/2 * ( Vector3.Left  + Vector3.Forward  );
																					   

		public Quadrocopter( Game game, World world )
		{
			MaxRPM			=	20000;	//	rotations per minute
			MaxRotorThrust	=	2.75f;	//	1100 gramms / 4 motors
			MaxRotorTorque	=	1.00f;	//	-- check!
			Mass			=	0.70f;	//	700 gramms
			AirResistance	=	0.10f;	//	
			ArmLength		=	0.30f;
			LinearSize		=	new Vector3( 0.6f, 0.06f, 0.6f );


			this.game		=	game;
			this.world		=	world;

			frame			=	game.Content.Load<Model>( @"scenes\quadFrame" );
			propellerA		=	game.Content.Load<Model>( @"scenes\propellerA" );
			propellerB		=	game.Content.Load<Model>( @"scenes\propellerB" );

			box	=	new Box( Vector3.Up * LinearSize.Y/2, LinearSize.X, LinearSize.Y, LinearSize.Z, Mass );
			box.AngularDamping = 0.5f;
			box.LinearDamping  = 0.5f;
			world.Space.Add( box );
			world.Drawer.Add( box );

			box.Material.KineticFriction	=	5.0f;
			box.Material.StaticFriction		=	5.0f;
			box.Material.Bounciness			=	0.2f;

			box.PositionUpdateMode = PositionUpdateMode.Continuous;
		}



		public void Update ( float dt )
		{
			float rpm2thrust = MaxRotorThrust / ( MaxRPM * MaxRPM );
			float rpm2torque = MaxRotorTorque / ( MaxRPM * MaxRPM );

			DirectControl();

			rpm1 = MathHelper.Clamp( rpm1, 0, MaxRPM );
			rpm2 = MathHelper.Clamp( rpm2, 0, MaxRPM );
			rpm3 = MathHelper.Clamp( rpm3, 0, MaxRPM );
			rpm4 = MathHelper.Clamp( rpm4, 0, MaxRPM );

			ApplyForceLL( ref box, dt, arm1 * ArmLength, Vector3.Up * rpm2thrust * rpm1 * rpm1 );
			ApplyForceLL( ref box, dt, arm2 * ArmLength, Vector3.Up * rpm2thrust * rpm2 * rpm2 );
			ApplyForceLL( ref box, dt, arm3 * ArmLength, Vector3.Up * rpm2thrust * rpm3 * rpm3 );
			ApplyForceLL( ref box, dt, arm4 * ArmLength, Vector3.Up * rpm2thrust * rpm4 * rpm4 );

			ApplyForceLL( ref box, dt, arm1, arm2 * rpm2torque * rpm1 * rpm1 );
			ApplyForceLL( ref box, dt, arm2, arm1 * rpm2torque * rpm2 * rpm2 );
			ApplyForceLL( ref box, dt, arm3, arm4 * rpm2torque * rpm3 * rpm3 );
			ApplyForceLL( ref box, dt, arm4, arm3 * rpm2torque * rpm4 * rpm4 );

			/*box.AngularMomentum = box.WorldTransform.Up * rpm2torque * rpm1 * rpm1;
			box.AngularMomentum = box.WorldTransform.Up * rpm2torque * rpm2 * rpm2;
			box.AngularMomentum = box.WorldTransform.Up * rpm2torque * rpm3 * rpm3;
			box.AngularMomentum = box.WorldTransform.Up * rpm2torque * rpm4 * rpm4;*/
		}



		public void DirectControl ()
		{
			var ks = Keyboard.GetState();
			var gps = GamePad.GetState( 0 );

			float avgRpm = gps.Triggers.Left * MaxRPM;

			rpm1 = avgRpm;
			rpm2 = avgRpm;
			rpm3 = avgRpm;
			rpm4 = avgRpm;

			rpm1 -= avgRpm * gps.ThumbSticks.Right.X / 8;
			rpm2 -= avgRpm * gps.ThumbSticks.Right.X / 8;
			rpm3 += avgRpm * gps.ThumbSticks.Right.X / 8;
			rpm4 += avgRpm * gps.ThumbSticks.Right.X / 8;
													 
			rpm1 -= avgRpm * gps.ThumbSticks.Right.Y / 8;
			rpm2 += avgRpm * gps.ThumbSticks.Right.Y / 8;
			rpm3 += avgRpm * gps.ThumbSticks.Right.Y / 8;
			rpm4 -= avgRpm * gps.ThumbSticks.Right.Y / 8;


			rpm1 += avgRpm * gps.ThumbSticks.Left.X / 8;
			rpm2 -= avgRpm * gps.ThumbSticks.Left.X / 8;
			rpm3 += avgRpm * gps.ThumbSticks.Left.X / 8;
			rpm4 -= avgRpm * gps.ThumbSticks.Left.X / 8;
		}



		public void Draw ( float dt, Matrix view, Matrix proj )
		{
			SimulatorGame.DrawModel( frame, box.WorldTransform, view, proj );
		}
		

		void ApplyForceLL ( ref Box box, float dt, Vector3 point, Vector3 localForce )
		{
			var m = box.WorldTransform;
			box.ApplyImpulse( Vector3.Transform(point, m), Vector3.TransformNormal(localForce * dt, m) );
		}


		void ApplyForceLG ( ref Box box, float dt, Vector3 point, Vector3 globalForce )
		{
			var m = box.WorldTransform;
			box.ApplyImpulse( Vector3.Transform(point, m), globalForce * dt );
		}
	}
}
