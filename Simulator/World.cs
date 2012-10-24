using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Audio;
using Microsoft.Xna.Framework.Content;
using Microsoft.Xna.Framework.GamerServices;
using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework.Input;
using XnaInput = Microsoft.Xna.Framework.Input;
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
        SpriteBatch spriteBatch;

		public GameTime	worldTime;

		public Space		Space	{ get { return space;	} }
		public ModelDrawer	Drawer	{ get { return drawer;	} }

		public	Quadrocopter quadrocopter;

		public	List<Quadrocopter> quadrocopters;

		public Tracker	Tracker;

		public AudioListener	Listener;

		float yaw	= 45;
		float pitch = 45;
		float dist  = 3;


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

            spriteBatch = new SpriteBatch(Game.GraphicsDevice);


            Vector3[] staticTriangleVertices;
            int[] staticTriangleIndices;

            TriangleMesh.GetVerticesAndIndicesFromModel( worldModel, out staticTriangleVertices, out staticTriangleIndices );
            var staticMesh			= new StaticMesh( staticTriangleVertices, staticTriangleIndices );
            staticMesh.Sidedness	= TriangleSidedness.Counterclockwise;

			space.Add( staticMesh );
			drawer.Add( staticMesh );
				
			Connect3DMouse();

			Listener	=	new AudioListener();

            quadrocopters = new List<Quadrocopter>() { new Quadrocopter(Game, this, Vector3.Zero, "q1") };

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



		public Vector3 Mouse3DTranslation	{ get; protected set; }
		public Vector3 Mouse3DRotationAxis	{ get; protected set; }
		public float   Mouse3DRotationAngle	{ get; protected set; }


		MouseState oldMouseState;

		/// <summary>
		/// Allows the game component to update itself.
		/// </summary>
		/// <param name="gameTime">Provides a snapshot of timing values.</param>
		public override void Update ( GameTime gameTime )
		{
			var ds = Game.GetService<DebugStrings>();
			float dt = (float)gameTime.ElapsedGameTime.TotalSeconds;

			Form form = (Form)Form.FromHandle( Game.Window.Handle );
			MouseState mouseState = Mouse.GetState();

			if ( form.Focused ) { 
				
				if ( mouseState.LeftButton == XnaInput.ButtonState.Pressed ) {
					yaw   += 0.5f * (oldMouseState.X - mouseState.X);
					pitch -= 0.5f * (oldMouseState.Y - mouseState.Y);
				}
				
				if ( mouseState.RightButton == XnaInput.ButtonState.Pressed ) {
					dist *= (float)Math.Pow( 1.007f, oldMouseState.Y - mouseState.Y );
				}
			}

			oldMouseState = mouseState;


			if (Tracker!=null) {
				ds.Add("Tracker enabled", Color.Lime);
			} else {
				ds.Add("Tracker disabled", Color.Red);
			}



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



			foreach (var quadrocop in quadrocopters) {
				quadrocop.Update(dt);
			}


			space.Update( (float)gameTime.ElapsedGameTime.TotalSeconds );
			drawer.Update();

			worldTime = gameTime;

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

			var rot  = Matrix.CreateFromYawPitchRoll( MathHelper.ToRadians( yaw ), MathHelper.ToRadians( pitch ), 0 );
			var proj = Matrix.CreatePerspectiveFieldOfView( MathHelper.ToRadians(cfg.Fov), Game.GraphicsDevice.Viewport.AspectRatio, 0.01f, 500.0f );
			var view = Matrix.CreateLookAt( rot.Forward * dist, Vector3.Zero, Vector3.Up );


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



            Game.GraphicsDevice.SetRenderTarget(null);

			//quadrocopter.Draw( dt, view, proj );
			foreach (var quadrocop in quadrocopters) {
				quadrocop.Draw(dt, view, proj);
			}


			dr.DrawGrid(8);

		

			if ( cfg.ShowBodies ) {
				drawer.Draw( view, proj );
			}

			dr.SetMatrix( view, proj );

			base.Draw( gameTime );

		}
	}
}
