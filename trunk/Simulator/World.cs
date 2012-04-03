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
using BEPUphysics;
using BEPUphysics.Collidables;
using BEPUphysics.DataStructures;
using BEPUphysics.CollisionShapes.ConvexShapes;
using BEPUphysicsDrawer.Models;
using BEPUphysics.Entities.Prefabs;
using BEPUphysics.EntityStateManagement;
using Misc;

namespace Simulator {
	/// <summary>
	/// This is a game component that implements IUpdateable.
	/// </summary>
	public class World : Microsoft.Xna.Framework.DrawableGameComponent {

		string		worldModelName;
		Model		worldModel;
		Space		space;
		ModelDrawer	drawer;

		public Space		Space	{ get { return space;	} }
		public ModelDrawer	Drawer	{ get { return drawer;	} }

		public	Quadrocopter quadrocopter;


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
				
			quadrocopter =	new Quadrocopter( Game, this );

			base.Initialize();
		}


		/// <summary>
		/// Allows the game component to update itself.
		/// </summary>
		/// <param name="gameTime">Provides a snapshot of timing values.</param>
		public override void Update ( GameTime gameTime )
		{
			// TODO: Add your update code here
			float dt = (float)gameTime.ElapsedGameTime.TotalSeconds;

			quadrocopter.Update( dt );

			space.Update( (float)gameTime.ElapsedGameTime.TotalSeconds );
			drawer.Update();

			base.Update( gameTime );
		}


		/// <summary>
		/// Draws stuff
		/// </summary>
		/// <param name="gameTime"></param>
		public override void Draw ( GameTime gameTime )
		{
			var dt  = (float)gameTime.ElapsedGameTime.TotalSeconds;
			var cfg	= Game.GetService<Settings>().Configuration;

			Game.GraphicsDevice.ResetDeviceState();

			var proj = Matrix.CreatePerspectiveFieldOfView( MathHelper.ToRadians(70), Game.GraphicsDevice.Viewport.AspectRatio, 0.1f, 5000.0f );
			var view = Matrix.CreateLookAt( 2*(Vector3.Up + Vector3.Backward + Vector3.Right), Vector3.Zero, Vector3.Up );

			quadrocopter.Draw( dt, view, proj );

			SimulatorGame.DrawModel( worldModel, Matrix.Identity, view, proj );


			if ( cfg.ShowBodies ) {
				drawer.Draw( view, proj );
			}

			base.Draw( gameTime );

		}
	}
}
