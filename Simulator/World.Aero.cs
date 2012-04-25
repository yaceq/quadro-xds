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
	public partial class World : Microsoft.Xna.Framework.DrawableGameComponent {

		BoundingBox bbox = new BoundingBox( new Vector3(-16, 0, -16), new Vector3(16, 16, 16) );

		void UpdateAerodynamicForces ( float dt )
		{
			var dr = Game.GetService<DebugRender>();

			dr.DrawBox( bbox, Color.White );

		}

	}
}
