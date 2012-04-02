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
	/// <summary>
	/// This is a game component that implements IUpdateable.
	/// </summary>

	public class Settings : Microsoft.Xna.Framework.GameComponent
	{
		public Configuration	Configuration { get; protected set; }

		public Settings(Game game) : base(game)
		{
		}

		public override void Initialize()
		{
			LoadSettings();
			base.Initialize();
		}

		public void LoadSettings () 
		{
			Configuration	=	Core.LoadFromXml<Configuration>( "Settings.xml", false );
		}

		public void SaveSettings () 
		{
			Core.SaveToXml<Configuration>( Configuration, "Settings.xml" );
		}

		public override void Update(GameTime gameTime)
		{
			base.Update(gameTime);
		}
	}
}
