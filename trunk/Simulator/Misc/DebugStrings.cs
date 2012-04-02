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


namespace Misc {

	public class DebugStrings : Microsoft.Xna.Framework.DrawableGameComponent {

		string fontName;
		int    shadowOffset;
		SpriteFont	spriteFont;
		SpriteBatch spriteBatch;
		List<Line> lines = new List<Line>();

		class Line : IComparable {
			public string	text;
			public Color	color;
			public int		priority;

			public int CompareTo (object obj) {
				Line other = obj as Line;
				return other.priority - priority;
			}
		}

		public DebugStrings ( Game game, string fontName, int shadowOffset=1 ) : base( game ) 
		{
			this.fontName = fontName;
			this.shadowOffset = shadowOffset;
		}

		public override void Initialize () 
		{
			base.Initialize();
		}


		public override void Update ( GameTime gameTime ) 
		{
			base.Update( gameTime );
		}
	

		protected override void LoadContent () 
		{
			spriteFont	= Game.Content.Load<SpriteFont>( fontName );
			spriteBatch	= new SpriteBatch( Game.GraphicsDevice );
		}


		protected override void UnloadContent () 
		{
			base.UnloadContent();
		}


		public override void Draw ( GameTime gameTime ) 
		{
			float h = spriteFont.MeasureString("A").Y;
			float x = 0;
			float y = 0;
			float d = (float)shadowOffset;

			//lines.Sort();
			Game.GraphicsDevice.BlendState			= BlendState.Opaque;
			Game.GraphicsDevice.RasterizerState		= RasterizerState.CullCounterClockwise;
			Game.GraphicsDevice.DepthStencilState	= DepthStencilState.Default;

			spriteBatch.Begin();

			//lines.Reverse();

				foreach (var line in lines) {
					spriteBatch.DrawString( spriteFont, line.text, new Vector2(x+d,y+d), Color.Black );
					spriteBatch.DrawString( spriteFont, line.text, new Vector2(x,  y  ), line.color );
					y += h;
				}

			spriteBatch.End();
		}


		public void Add ( string text ) {
			Line line = new Line();
			line.text		= text;
			line.priority	= 0;
			line.color		= Color.White;
			lines.Add( line );
		}

		public void Add ( string text, Color color ) {
			Line line = new Line();
			line.text		= text;
			line.priority	= 0;
			line.color		= color;
			lines.Add( line );
		}

		public void Add ( string text, int priority ) {
			Line line = new Line();
			line.text		= text;
			line.priority	= priority;
			line.color		= Color.White;
			lines.Add( line );
		}

		public void Add ( string text, Color color, int priority ) {
			Line line = new Line();
			line.text		= text;
			line.priority	= priority;
			line.color		= color;
			lines.Add( line );
		}

		public void Clear () {
			lines.Clear();
		}

	}
}
