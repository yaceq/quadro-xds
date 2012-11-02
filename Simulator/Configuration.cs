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
using System.ComponentModel;


namespace Simulator {
	public class Configuration {

		[Category("Rendering")]	public bool			PreferMultiSampling { set; get; }
		[Category("Rendering")]	public bool			IsFullScreen { set; get; }
		[Category("Rendering")]	public int			PreferredBackBufferWidth { set; get; }
		[Category("Rendering")]	public int			PreferredBackBufferHeight { set; get; }
		[Category("Rendering")]	public bool			UseSecondMonitor { set; get; }

		[Category("View")]		public float		Yaw { set; get; }
		[Category("View")]		public float		Pitch { set; get; }
		[Category("View")]		public float		Distance { set; get; }
		[Category("View")]		public float		Fov			{ set; get; }

		[Category("Tracker")]			public string		Host { set; get; }

		[Category("Communications")]	public string		Port { set; get; }
		[Category("Communications")]	public int			BaudRate		 { set; get; }

		[Category("Physics")]			public bool			ShowBodies { set; get; }

		[Category("Stabiliazation")]	public float		RollK { set; get; }
		[Category("Stabiliazation")]	public float		RollD { set; get; }
		[Category("Stabiliazation")]	public float		PitchK { set; get; }
		[Category("Stabiliazation")]	public float		PitchD { set; get; }
		[Category("Stabiliazation")]	public float		YawK { set; get; }
		[Category("Stabiliazation")]	public float		YawD { set; get; }


		public enum CameraModes {
			BoundToQuadrocopter,
			BoundToQuadrocopterHorison,
			ViewFromPoint,
			ViewAround
		}


		public Configuration()
		{
			PreferredBackBufferWidth	=	1280;
			PreferredBackBufferHeight	=	720;

			Host		=	"192.168.10.1:801";

			Port		=	"COM6";
			BaudRate	=	38400;
			Fov			=	70f;
			Distance	=	3;
		}
	}
}
