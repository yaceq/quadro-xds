using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Audio;
using Microsoft.Xna.Framework.Content;
using Microsoft.Xna.Framework.GamerServices;
using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework.Input;
using Microsoft.Xna.Framework.Media;
using System.Windows;
using System.Windows.Forms;
using BEPUphysics;
using BEPUphysics.Collidables;
using BEPUphysics.DataStructures;
using BEPUphysics.CollisionShapes.ConvexShapes;
using BEPUphysicsDrawer.Models;
using BEPUphysics.Entities.Prefabs;
using BEPUphysics.EntityStateManagement;
using System.Runtime.InteropServices;
using Misc;

namespace Simulator {
	/// <summary>
	/// This is a game component that implements IUpdateable.
	/// </summary>
	public partial class World : Microsoft.Xna.Framework.DrawableGameComponent {

		string		worldModelName;
		Model		worldModel;
		Space		space;
		ModelDrawer	drawer;
		public GameTime	worldTime;

		public Space		Space	{ get { return space;	} }
		public ModelDrawer	Drawer	{ get { return drawer;	} }

		public	Quadrocopter quadrocopter;

		public	List<Quadrocopter> quadrocopters_list;

		public Tracker	Tracker;

		public AudioListener	Listener;


		/// <summary>
		/// Constructor
		/// </summary>
		/// <param name="game"></param>
		/// <param name="worldModel"></param>
		public World ( Game game, string worldModelName ) : base( game )
		{
			this.worldModelName = worldModelName;

		}



		/// <summary>
		/// Allows the game component to perform any initialization it needs to before starting
		/// to run.  This is where it can query for any required services and load content.
		/// </summary>
		public override void Initialize ()
		{
			space		=	new Space();
			space.ForceUpdater.Gravity = Vector3.Down * 9.8f;

			drawer		=	new BruteModelDrawer( this.Game );
			drawer.IsWireframe = true;

			worldModel	=	Game.Content.Load<Model>(worldModelName);


            Vector3[] staticTriangleVertices;
            int[] staticTriangleIndices;

            TriangleMesh.GetVerticesAndIndicesFromModel( worldModel, out staticTriangleVertices, out staticTriangleIndices );
            var staticMesh			= new StaticMesh( staticTriangleVertices, staticTriangleIndices );
            staticMesh.Sidedness	= TriangleSidedness.Counterclockwise;

			space.Add( staticMesh );
			drawer.Add( staticMesh );
				
			Connect3DMouse();

			Listener	=	new AudioListener();

			quadrocopters_list = new List<Quadrocopter>() { new Quadrocopter(Game, this, Vector3.Zero) };

			quadrocopter = quadrocopters_list[0];

			base.Initialize();
		}



		/// <summary>
		/// ConnectTracker
		/// </summary>
		public void ConnectTracker()
		{
			var cfg = Game.GetService<Settings>().Configuration;
			try {
				Tracker = new Tracker( cfg.Host );
			} catch (Exception ex) {
				MessageBox.Show(String.Format("Failed to connect to : {0}\r\n{1}\r\nTracking is disabled", cfg.Host, ex.Message), "Tracker connection error", MessageBoxButtons.OK, System.Windows.Forms.MessageBoxIcon.Warning );
				Tracker = null;
			}
		}



		/// <summary>
		/// DisconnectTracker
		/// </summary>
		public void DisconnectTracker()
		{
			if (Tracker!=null) {	
				Tracker = null;
			}
		}




		private TDx.TDxInput.Sensor		sensor	 = null; 
		private TDx.TDxInput.Keyboard	keyboard = null;
		private TDx.TDxInput.Device		device	 = null;

		/// <summary>
		/// Connects 3D mouse :
		/// </summary>
		public void Connect3DMouse ()
		{
			try {
				this.device		= new TDx.TDxInput.Device();
				this.sensor		= this.device.Sensor;
				this.keyboard	= this.device.Keyboard;

				// Associate a configuration with this device'
				this.device.LoadPreferences("QuadroXDS");

				//Connect everything up
				this.device.Connect();

			} catch (COMException e) {
				Console.WriteLine("{0} Caught exception #1.", e);
				this.device		= null;
				this.sensor		= null;
				this.keyboard	= null;
			}
		}


		/*public Vector3	Get3DxTranslation () 
		{
			if (sensor==null) return Vector3.Zero;

            TDx.TDxInput.Vector3D translation;
            translation = sensor.Translation;
            return new Vector3( (float)translation.X, (float)translation.Y, (float)translation.Z );
		}


		public Vector3	Get3DxRotation () 
		{
			if (sensor==null) return Vector3.Zero;

            TDx.TDxInput.AngleAxis rotation;
            rotation = sensor.Rotation;
            return new Vector3( (float)rotation.X, (float)rotation.Y, (float)rotation.Z );
		} */


		public Vector3 Mouse3DTranslation	{ get; protected set; }
		public Vector3 Mouse3DRotationAxis	{ get; protected set; }
		public float   Mouse3DRotationAngle	{ get; protected set; }


		/// <summary>
		/// Allows the game component to update itself.
		/// </summary>
		/// <param name="gameTime">Provides a snapshot of timing values.</param>
		public override void Update ( GameTime gameTime )
		{
			// TODO: Add your update code here
			var ds = Game.GetService<DebugStrings>();
			float dt = (float)gameTime.ElapsedGameTime.TotalSeconds;

			if (sensor!=null) {
				var translation = sensor.Translation;
				Mouse3DTranslation = new Vector3( (float)translation.X, (float)translation.Y, (float)translation.Z );

				var rotation = sensor.Rotation;

				//sensor.Period
				Mouse3DRotationAxis		= new Vector3( (float)rotation.X, (float)rotation.Y, (float)rotation.Z );
				Mouse3DRotationAngle	= (float)rotation.Angle;

				ds.Add( Mouse3DTranslation.ToString() );
				ds.Add( Mouse3DRotationAxis.ToString() + " : " + Mouse3DRotationAngle.ToString() );
			}


			//quadrocopter.Update( dt );
			foreach (var quadrocop in quadrocopters_list)
			{
				quadrocop.Update(dt);
			}

			space.Update( (float)gameTime.ElapsedGameTime.TotalSeconds );
			drawer.Update();

			worldTime = gameTime;


			UpdateAerodynamicForces( dt );

			base.Update( gameTime );
		}




		Matrix	currentViewMatrix;
		bool	firstFrame = true;

		/// <summary>
		/// Draws stuff
		/// </summary>
		/// <param name="gameTime"></param>
		public override void Draw ( GameTime gameTime )
		{
			var dr = Game.GetService<DebugRender>();

			var dt  = (float)gameTime.ElapsedGameTime.TotalSeconds;
			var cfg	= Game.GetService<Settings>().Configuration;

			Game.GraphicsDevice.ResetDeviceState();

			var proj = Matrix.CreatePerspectiveFieldOfView( MathHelper.ToRadians(cfg.Fov), Game.GraphicsDevice.Viewport.AspectRatio, 0.1f, 5000.0f );
			var view = Matrix.CreateLookAt( 2*(Vector3.Up + Vector3.Backward + Vector3.Right), Vector3.Zero, Vector3.Up );

			if (cfg.CameraMode==Configuration.CameraModes.ViewFromPoint) {
				view = Matrix.CreateLookAt( cfg.PointOfView, quadrocopter.Position, Vector3.Up );
			}

			if (cfg.CameraMode==Configuration.CameraModes.BoundToQuadrocopter) {
				var fw = quadrocopter.Transform.Forward;
				var	up = quadrocopter.Transform.Up;
				var p  = quadrocopter.Position + up * 0.3f - fw;
				if (p.Y < 0.1f ) p.Y = 0.1f;
				view = Matrix.CreateLookAt( p, p + fw, Vector3.Up );
			}

			if (cfg.CameraMode==Configuration.CameraModes.BoundToQuadrocopterHorison) {
				var fw = quadrocopter.Transform.Forward;
				var	up = quadrocopter.Transform.Up;
				var p  = quadrocopter.Position + up * 0.3f - fw;
				if (p.Y < 0.1f ) p.Y = 0.1f;
				view = Matrix.CreateLookAt( p, p + fw, up );
			}

			if (firstFrame) {
				currentViewMatrix = view;
				firstFrame = false;
			} else {
				currentViewMatrix = MathX.LerpMatrix( currentViewMatrix, view, 0.9f );
				//view = currentViewMatrix;
			}

			Listener.Position	=	Matrix.Invert(view).Translation;
			Listener.Forward	=	Matrix.Invert(view).Forward;
			Listener.Up			=	Matrix.Invert(view).Up;


			//quadrocopter.Draw( dt, view, proj );
			foreach (var quadrocop in quadrocopters_list)
			{
				quadrocop.Draw(dt, view, proj);
			}

			SimulatorGame.DrawModel( worldModel, Matrix.Identity, view, proj );


			if ( cfg.ShowBodies ) {
				drawer.Draw( view, proj );
			}

			dr.SetMatrix( view, proj );

			base.Draw( gameTime );

		}
	}
}
