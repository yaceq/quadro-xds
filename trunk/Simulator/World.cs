#define SVT
using System;
using System.Collections.Generic;
using System.Collections;
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
using BEPUphysics.MathExtensions;
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

		//float yaw	= -145;
		//float pitch = 35;
		//float dist  = 3;


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
			space.ForceUpdater.Gravity = Vector3.Down * 9.80665f;

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
			return;

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
			var ds		= Game.GetService<DebugStrings>();
			var cfg 	= Game.GetService<Settings>().Configuration;
			float dt	= (float)gameTime.ElapsedGameTime.TotalSeconds;

			Form form = (Form)Form.FromHandle( Game.Window.Handle );
			MouseState mouseState = Mouse.GetState();

			if ( form.Focused ) { 
				
				if ( mouseState.LeftButton == XnaInput.ButtonState.Pressed ) {
					cfg.Yaw   += 0.5f * (oldMouseState.X - mouseState.X);
					cfg.Pitch -= 0.5f * (oldMouseState.Y - mouseState.Y);
					cfg.Pitch =  MathHelper.Clamp( cfg.Pitch, -89, 89 );
				}
				
				if ( mouseState.RightButton == XnaInput.ButtonState.Pressed ) {
					cfg.Distance *= (float)Math.Pow( 1.007f, oldMouseState.Y - mouseState.Y );
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

#if SVT
                quadrocop.AddGeneralForce(5.687857f, new Vector3(0.0005f, 0.0f, 0.0005f));
                //Нормировочный коэффициет 26.52f по X- разобраться откуда!
                //Нормировочный коэффициет 26.52f по Z- разобраться откуда!
                var back_momentum = new Vector3(-0.0005f, 0, -0.0005f) * 26.52f;
                back_momentum = Vector3.TransformNormal(back_momentum, Matrix.Invert(quadrocop.box.WorldTransform));
                quadrocop.testApplyMomentum(ref back_momentum, dt);
                quadrocop.testApplyCForce(new Vector3(0, 5.687857f, 0), dt);
#endif
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

			var rot  = Matrix.CreateFromYawPitchRoll( MathHelper.ToRadians( cfg.Yaw ), MathHelper.ToRadians( cfg.Pitch ), 0 );
			var proj = Matrix.CreatePerspectiveFieldOfView( MathHelper.ToRadians(cfg.Fov), Game.GraphicsDevice.Viewport.AspectRatio, 0.01f, 500.0f );
			var view = Matrix.CreateLookAt( rot.Forward * cfg.Distance, Vector3.Zero, Vector3.Up );


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


			dr.DrawGrid(3);

			dr.DrawBasis( Matrix.CreateTranslation( Vector3.Up ), 0.5f );

		

			if ( cfg.ShowBodies ) {
				drawer.Draw( view, proj );
			}

			dr.SetMatrix( view, proj );

			base.Draw( gameTime );

		}

	}

        public class QController
    {

        public QController(int history_len)
        {   
            posHistory   = new ArrayList(3);
            velHistory   = new ArrayList(3);
            rotHistory   = new ArrayList(3);
            nHistory     = new ArrayList(3);
            omegaHistory = new ArrayList(3);
            forceHistory = new ArrayList(3);
            stabK = new float[4];
            stabK[0] = -20.0f;
            stabK[1] = -40.0f;
            stabK[2] = -25.01f;
            stabK[3] = -10.01f;
        }

        public void setTarget(Vector3 targetPos)
        { 
            this.targetPos = targetPos;            
        }

        public Vector3 QVMultiply(Quaternion qt, Vector3 vec)
        {
            Quaternion q_res =  Quaternion.Multiply(qt, (new Quaternion(vec, 0.0f)));
            return new Vector3(q_res.X, q_res.Y, q_res.Z);
        }

        public Vector3 QVMultiply(Vector3 vec, Quaternion qt)
        {
            Quaternion q_res = Quaternion.Multiply((new Quaternion(vec, 0.0f)), qt);
            return new Vector3(q_res.X, q_res.Y, q_res.Z);
        }

        public Vector3 QuaternionToVector(Quaternion qt)
        {            
            return new Vector3(qt.X, qt.Y, qt.Z);
        }

        public void QControl(out Vector3 out_force, out Vector3 out_momentum, Matrix state, Matrix3X3 inertiaTensor, float dt)
        {
            //Вычсление относительного положения целевого состояния
            
            Quaternion  rotation = new Quaternion();            
            Vector3     position = new Vector3();
            Vector3     velocity = new Vector3();
            Vector3     omega    = new Vector3();
            Vector3     normal   = new Vector3();
            Vector3     scale;

            if (posHistory.Count >= 3)
                posHistory.RemoveAt(0);

            if (rotHistory.Count >= 3)
                rotHistory.RemoveAt(0);

            if (velHistory.Count >= 3)
                velHistory.RemoveAt(0);

            if (nHistory.Count >= 3)
                nHistory.RemoveAt(0);

            if (omegaHistory.Count >= 3)
                omegaHistory.RemoveAt(0);

            state.Decompose(out scale, out rotation, out position);
            posHistory.Add(position);
            rotHistory.Add(rotation);            
            normal = Vector3.TransformNormal(Vector3.Up,state);
            
            nHistory.Add(normal);

            if (posHistory.Count >= 2)
            {
                velocity = (position - (Vector3)posHistory[posHistory.Count - 2]) / dt;
                velHistory.Add(velocity);
            }
            else
                velHistory.Add(Vector3.Zero);

            if (rotHistory.Count >= 2)
            {
                omega = QuaternionToVector(Quaternion.Multiply((rotation - (Quaternion)rotHistory[rotHistory.Count - 2]), Quaternion.Conjugate(rotation)));
                omega = omega * (2 / dt);
                omegaHistory.Add(omega);
            }
            else
                omegaHistory.Add(Vector3.Zero);


            if (true)
            {
                Vector3     deltaPosition = position - targetPos;
                Vector3      force   = new Vector3();
                Vector3 pred_normal  = new Vector3();
                Vector3 pred_omega   = new Vector3();
                Vector3   momentum   = new Vector3();

                if (forceHistory.Count >= 3)
                {
                    force =
                    (
                          stabK[0] * deltaPosition
                        + stabK[1] * velocity
                        + stabK[2] / dt / dt * ((Vector3)posHistory[0] - 2 * (Vector3)posHistory[1] + (Vector3)posHistory[2])
                        + stabK[3] / dt / dt * ((Vector3)velHistory[0] - 2 * (Vector3)velHistory[1] + (Vector3)velHistory[2])
                    )
                        * dt * dt
                        + 2 * (Vector3)forceHistory[2] - (Vector3)forceHistory[1];


                    forceHistory.Add(force);
                    pred_normal = (force / force.Length());

                    pred_omega =
                       Vector3.Cross(pred_normal + (Vector3)nHistory[nHistory.Count - 1],
                                     pred_normal - (Vector3)nHistory[nHistory.Count - 1])
                                    / 2 / dt;

                    Vector3 localOmegaPred = Vector3.TransformNormal(pred_omega, state);
                    Vector3 localOmegaCurr = Vector3.TransformNormal((Vector3)omegaHistory[omegaHistory.Count - 1], state);

                    Matrix inertiaTensor4X4 = Matrix3X3.ToMatrix4X4(inertiaTensor);
                    momentum = Vector3.TransformNormal((localOmegaPred - localOmegaCurr) / dt, inertiaTensor4X4);          
                    out_momentum = momentum;

                    forceHistory.RemoveAt(0);
                }
                else
                {
                    forceHistory.Add(6.860f * Vector3.Up);
                    out_momentum = Vector3.Zero;
                }               
                
                out_force    = force;                
            }
        }

        private ArrayList posHistory;
        private ArrayList velHistory;
        private ArrayList rotHistory;
        private ArrayList nHistory;
        private ArrayList omegaHistory;
        private ArrayList forceHistory;
           
        private Matrix    targetState;
        private Vector3   targetPos;        

        private float[] stabK;
    }
}

