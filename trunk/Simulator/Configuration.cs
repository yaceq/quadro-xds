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

		[Category("Tracker")]			public string		Host { set; get; }

		[Category("Communications")]	public string		Port { set; get; }

		[Category("Physics")]			public bool			ShowBodies { set; get; }


		public enum CameraModes {
			BoundToQuadrocopter,
			BoundToQuadrocopterHorison,
			ViewFromPoint,
			ViewAround
		}

		[Category("Camera")]	public CameraModes	CameraMode	{ set; get; }
		[Category("Camera")]	public Vector3		Offset		{ set; get; }
		[Category("Camera")]	public Vector3		PointOfView	{ set; get; }
		[Category("Camera")]	public float		Fov			{ set; get; }


		public Configuration()
		{
			PreferredBackBufferWidth	=	1280;
			PreferredBackBufferHeight	=	720;

			Host	=	"192.168.10.1:801";

			Port	=	"COM6";

			CameraMode	=	CameraModes.ViewFromPoint;
			Offset		=	Vector3.Backward + Vector3.Up * 0.25f;
			PointOfView	=	new Vector3(3, 1.8f, 3);
			Fov			=	70f;
		}
	}
}
