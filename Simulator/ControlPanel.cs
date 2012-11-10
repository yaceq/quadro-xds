using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.IO;
using System.Timers;
using System.Threading;
using System.Windows.Forms;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Audio;
using Microsoft.Xna.Framework.Content;
using Microsoft.Xna.Framework.GamerServices;
using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework.Input;
using Microsoft.Xna.Framework.Media;
using Misc;
using System.Windows.Forms.DataVisualization.Charting;
using System.IO.Ports;

namespace Simulator {
	public partial class ControlPanel : Form {

		Game game;

		public ControlPanel ( Game game )
		{
			this.game = game;
			InitializeComponent();

			settingsPropertyGrid.SelectedObject = game.GetService<Settings>().Configuration;
			//quadrocopterPropertyGrid.SelectedObject = game.GetService<World>().quadrocopter;
			quadrocopterPropertyGrid.SelectedObject = game.GetService<World>().quadrocopters[0];

            foreach (var qc in game.GetService<World>().quadrocopters) {
                listBox1.Items.Add(qc.Name);
            }

			settingsPropertyGrid.CollapseAllGridItems();
			quadrocopterPropertyGrid.CollapseAllGridItems();
		}

		

		private void exitToolStripMenuItem_Click ( object sender, EventArgs e )
		{
			game.Exit();
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



		void ToAngles(Matrix mat, out float yaw, out float pitch, out float roll)
		{
			//	     |  0  1  2  3 |	| 11 12 13 14 |
			//	M =  |  4  5  6  7 |	| 21 22 23 24 |
			//	     |  8  9 10 11 |	| 31 32 33 34 |
			//	     | 12 13 14 15 |	| 41 42 43 44 |

			float sy = MathHelper.Clamp(mat.M13, -1, 1);
			float angle_x = 0;
			float angle_z = 0;
			float angle_y = -(float)Math.Asin(sy);
			float cy = (float)Math.Cos(angle_y);
			float _trx, _try;

			if (Math.Abs(cy) > 8192.0f * float.Epsilon)
			{
				_trx = mat.M33 / cy;
				_try = -mat.M23 / cy;

				angle_x = (float)Math.Atan2(_try, _trx);

				_trx = mat.M11 / cy;
				_try = -mat.M12 / cy;

				angle_z = (float)Math.Atan2(_try, _trx);

			}
			else
			{
				angle_x = 0;

				_trx = mat.M22;
				_try = mat.M21;

				angle_z = (float)Math.Atan2(_try, _trx);
			}

			pitch = angle_x;
			yaw = angle_y;
			roll = angle_z;
		}

		

        void listBox1_Click(object sender, System.EventArgs e)
        {
            quadrocopterPropertyGrid.SelectedObject = game.GetService<World>().quadrocopters[listBox1.SelectedIndex];
        }



		private void connectToQuadrocopterViaCOMToolStripMenuItem_Click ( object sender, EventArgs e )
		{
			game.GetService<World>().quadrocopters[0].RunCommunicationProtocol();
		}

		private void disconnectToolStripMenuItem_Click ( object sender, EventArgs e )
		{
			game.GetService<World>().quadrocopters[0].StopCommunicationProtocol();
		}

		private void trackBar1_Scroll ( object sender, EventArgs e )
		{
			TrackBar trackBar =  (TrackBar)sender;
			if (trackBar==trackBar1) game.GetService<World>().quadrocopters[0].Rotor1 = trackBar.Value;
			if (trackBar==trackBar2) game.GetService<World>().quadrocopters[0].Rotor2 = trackBar.Value;
			if (trackBar==trackBar3) game.GetService<World>().quadrocopters[0].Rotor3 = trackBar.Value;
			if (trackBar==trackBar4) game.GetService<World>().quadrocopters[0].Rotor4 = trackBar.Value;
		}



		Thread com6Thread;

		void Com6Func ()
		{
			try {
				Console.WriteLine("Connecting to COM6...");
				SerialPort	com = new SerialPort("COM6", 9600);
				com.Open();

				Console.WriteLine("Done.");

				while (true) {
					char ch = (char)com.ReadChar();
					Console.Write(ch);
				}

			} catch ( Exception e ) {
				MessageBox.Show(e.Message);
			}
		}

		private void connectCOM6ToolStripMenuItem_Click ( object sender, EventArgs e )
		{
			com6Thread = new Thread( Com6Func );
			com6Thread.Start();
		}

		private void button1_Click ( object sender, EventArgs e )
		{
			game.GetService<World>().quadrocopters[0].RunTakeoffThrustEstimation();
		}

		private void trackBar5_Scroll ( object sender, EventArgs e )
		{
			label1.Text = "Thrust = " + trackBar5.Value.ToString();
			game.GetService<World>().quadrocopters[0].ThrustTest = trackBar5.Value;
		}

		private void button2_Click ( object sender, EventArgs e )
		{											  
			var q = game.GetService<World>().quadrocopters[0];

			if (q.logWriter==null) {

				string name = string.Format("{0}_{1:yyyyMMdd_HHmmss}.log", q.Name, DateTime.Now);
				q.logWriter = new StreamWriter( name, false );
				q.firstLine	= true;

				button2.Text = string.Format("Log: {0}\r\nStop Logging", name);

			} else {

				button2.Text = "Start Logging";

				q.logWriter.Flush();
				q.logWriter.Close();
				q.logWriter.Dispose();
				q.logWriter = null;

			}
		}

		private void button1_Click_1 ( object sender, EventArgs e )
		{
			var q = game.GetService<World>().quadrocopters[0];
			q.box.WorldTransform	=	Matrix.Identity;
		}

	}
}
