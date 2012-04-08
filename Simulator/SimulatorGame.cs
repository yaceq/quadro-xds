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
using Misc;


namespace Simulator {
	
	public class SimulatorGame : Microsoft.Xna.Framework.Game {
		GraphicsDeviceManager graphics;
		SpriteBatch spriteBatch;

		ControlPanel	controlPanel = null;

		public SimulatorGame ()
		{
			graphics = new GraphicsDeviceManager( this );

			this.AddServiceAndComponent( new Settings( this ) );
			this.AddServiceAndComponent( new World( this, @"scenes\plane" ) );
			this.AddServiceAndComponent( new DebugRender( this ) );
			this.AddServiceAndComponent( new DebugStrings( this, "debugFont", 1 ) );

			this.GetService<Settings>().LoadSettings();

			var cfg = this.GetService<Settings>().Configuration;

			graphics.PreferredBackBufferWidth	=	cfg.PreferredBackBufferWidth;
			graphics.PreferredBackBufferHeight	=	cfg.PreferredBackBufferHeight;
			graphics.PreferMultiSampling		=	cfg.PreferMultiSampling;

			this.IsMouseVisible = true;

			Content.RootDirectory = "Content";
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


		protected override void OnExiting ( object sender, EventArgs args )
		{
			this.GetService<Settings>().SaveSettings();

			base.OnExiting( sender, args );
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
			var ds = this.GetService<DebugStrings>();
			var dr = this.GetService<DebugRender>();
			ds.Clear();
			dr.Clear();

			ds.Add( string.Format("FPS = {0:0.00}", 1 / gameTime.ElapsedGameTime.TotalSeconds ) );

			// Allows the game to exit
			if ( Keyboard.GetState().IsKeyDown( Keys.Escape ) ) {
				this.Exit();
			}

			if (Keyboard.GetState().IsKeyDown( Keys.F2 )) {
				if ( controlPanel==null || controlPanel.IsDisposed ) {
					controlPanel = new ControlPanel(this);
					controlPanel.Show();
					/*controlPanel.propertyGrid1.SelectedObject = settings.UserConfig;
					controlPanel.propertyGrid1.CollapseAllGridItems();*/
				}
			}

			// TODO: Add your update logic here

			base.Update( gameTime );
		}


		/// <summary>
		/// This is called when the game should draw itself.
		/// </summary>
		/// <param name="gameTime">Provides a snapshot of timing values.</param>
		protected override void Draw ( GameTime gameTime )
		{
			GraphicsDevice.Clear( Color.CornflowerBlue );

			// TODO: Add your drawing code here

			var ds = this.GetService<DebugStrings>();

			base.Draw( gameTime );
		}



		/// <summary>
		/// 
		/// </summary>
		/// <param name="model"></param>
		/// <param name="world"></param>
		/// <param name="view"></param>
		/// <param name="proj"></param>
		public static void DrawModel ( Model model, Matrix world, Matrix view, Matrix proj ) 
		{
			foreach ( var modelMesh in model.Meshes ) {
				foreach ( BasicEffect effect in modelMesh.Effects ) {
					effect.DirectionalLight0.Enabled = true;
					effect.DirectionalLight0.Direction = Vector3.One;
					effect.DirectionalLight0.DiffuseColor = Vector3.One;
					effect.DirectionalLight0.SpecularColor = Vector3.One;
					effect.EnableDefaultLighting();
					//effect.DiffuseColor	=	color.ToVector3();

					effect.FogColor		=	(Color.CornflowerBlue).ToVector3();
					effect.FogEnabled	=	true;
					effect.FogStart		=	100;
					effect.FogEnd		=	1500;
				}	
			}

			model.Draw( world, view, proj );
		}

	}
}
