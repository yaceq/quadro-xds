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

		[Category("Motors")]	public int			Rotor1 { set; get; }
		[Category("Motors")]	public int			Rotor2 { set; get; }
		[Category("Motors")]	public int			Rotor3 { set; get; }
		[Category("Motors")]	public int			Rotor4 { set; get; }
        public string Name { protected set; get; }

		Model	frame;
		Model	propellerA;
		Model	propellerB;

		Game	game;
		World	world;
		Matrix	worldTransform;


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



		public void Update ( float dt )
		{
			var dr = game.GetService<DebugRender>();

			UpdateFromTracker();
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

					worldTransform = xform;
				}
			}
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

					s = string.Format("X{0,2:X2}{1,2:X2}{2,2:X2}{3,2:X2}\n", (byte)Rotor1, (byte)Rotor2, (byte)Rotor3, (byte)Rotor4 );

					port.Write( s );
					/*buf[0] = (byte)'X';
					buf[1] = (byte)Rotor1;
					buf[2] = (byte)Rotor2;
					buf[3] = (byte)Rotor3;
					buf[4] = (byte)Rotor4;
					buf[5] = (byte)'\n';
					port.Write( buf, 0, buf.Length );*/

					//string ack = port.ReadLine();
					//Console.WriteLine(ack);
					Thread.Sleep(50);
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
