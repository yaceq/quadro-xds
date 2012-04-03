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
		}

		private void exitToolStripMenuItem_Click ( object sender, EventArgs e )
		{
			game.Exit();
		}
	}
}
