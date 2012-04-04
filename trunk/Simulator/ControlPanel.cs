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
		}

		private void treeView1_AfterSelect ( object sender, TreeViewEventArgs e )
		{
			if ( e.Node == this.ObjectTreeView.Nodes["Settings"] ) {
				propertyGrid.SelectedObject = game.GetService<Settings>().Configuration;
			}
			if ( e.Node == this.ObjectTreeView.Nodes["Quadrocopters"] ) {
				propertyGrid.SelectedObject = game.GetService<World>().quadrocopter;
			}
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

	}
}
