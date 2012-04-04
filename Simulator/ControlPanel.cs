using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Audio;
using Microsoft.Xna.Framework.Content;
using Microsoft.Xna.Framework.GamerServices;
using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework.Input;
using Microsoft.Xna.Framework.Media;
using Misc;


namespace Simulator {
	public partial class ControlPanel : Form {

		Game game;

		public ControlPanel ( Game game )
		{
			this.game = game;
			InitializeComponent();

			settingsPropertyGrid.SelectedObject = game.GetService<Settings>().Configuration;
			quadrocopterPropertyGrid.SelectedObject = game.GetService<World>().quadrocopter;

			settingsPropertyGrid.CollapseAllGridItems();
			quadrocopterPropertyGrid.CollapseAllGridItems();
		}

		

		private void exitToolStripMenuItem_Click ( object sender, EventArgs e )
		{
			game.Exit();
		}


		private void timer1_Tick ( object sender, EventArgs e )
		{
			var world = game.GetService<World>();

			LinearPhaseDiagram.Series["Series1"].Points.AddXY( world.quadrocopter.Position.X, -world.quadrocopter.Position.Z ); 
		}


		private void connectToTrackerToolStripMenuItem_Click ( object sender, EventArgs e )
		{
			var world = game.GetService<World>();
			world.ConnectTracker();
		}

		private void disconnectTrackerToolStripMenuItem_Click ( object sender, EventArgs e )
		{
			var world = game.GetService<World>();
			world.ConnectTracker();
		}

	}
}
