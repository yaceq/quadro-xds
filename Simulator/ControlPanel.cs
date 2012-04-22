using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
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
			quadrocopterPropertyGrid.SelectedObjects = game.GetService<World>().quadrocopters_list.ToArray();


			settingsPropertyGrid.CollapseAllGridItems();
			quadrocopterPropertyGrid.CollapseAllGridItems();


			CustomLabel iLabel = null;
			Axis PitchRollAxisY = PitchRollDiagram.ChartAreas["ChartArea1"].AxisY;
			Axis PitchRollAxisX = PitchRollDiagram.ChartAreas["ChartArea1"].AxisX;
			Axis AltitudeAxisX = AltitudeDiagram.ChartAreas["ChartArea1"].AxisX;
			Axis AltitudeAxisY = AltitudeDiagram.ChartAreas["ChartArea1"].AxisY;
			Axis LinearPhaseAxisX = LinearPhaseDiagram.ChartAreas["ChartArea1"].AxisX;
			Axis LinearPhaseAxisY = LinearPhaseDiagram.ChartAreas["ChartArea1"].AxisY;
			Axis AngularPhaseDiagramAxisX = AngularPhaseDiagram.ChartAreas["ChartArea1"].AxisX;
			Axis AngularPhaseDiagramAxisY = AngularPhaseDiagram.ChartAreas["ChartArea1"].AxisY;

			{
				iLabel = PitchRollAxisY.CustomLabels.Add(2 * Math.PI + Math.PI / 5, 2 * Math.PI - Math.PI / 5, "2pi");
				iLabel.GridTicks = GridTickTypes.All;
				iLabel = PitchRollAxisY.CustomLabels.Add(-2 * Math.PI + Math.PI / 5, -2 * Math.PI - Math.PI / 5, "-2pi");
				iLabel.GridTicks = GridTickTypes.All;

				iLabel = PitchRollAxisY.CustomLabels.Add(Math.PI - Math.PI / 5, Math.PI + Math.PI / 5, "pi");
				iLabel.GridTicks = GridTickTypes.All;
				iLabel = PitchRollAxisY.CustomLabels.Add(-Math.PI + Math.PI / 5, -Math.PI - Math.PI / 5, "-pi");
				iLabel.GridTicks = GridTickTypes.All;

				iLabel = PitchRollAxisY.CustomLabels.Add(Math.PI / 2 + Math.PI / 5, Math.PI / 2 - Math.PI / 5, "pi/2");
				iLabel.GridTicks = GridTickTypes.All;
				iLabel = PitchRollAxisY.CustomLabels.Add(-Math.PI / 2 + Math.PI / 5, -Math.PI / 2 - Math.PI / 5, "-pi/2");
				iLabel.GridTicks = GridTickTypes.All;

				iLabel = PitchRollAxisY.CustomLabels.Add(Math.PI / 5, -Math.PI / 5, "0");
				iLabel.GridTicks = GridTickTypes.All;

				PitchRollAxisY.Minimum = -4;
				PitchRollAxisY.Maximum = 4;
			}
			{
				PitchRollAxisX.Interval = 5;
				PitchRollAxisX.IsMarksNextToAxis = true;
				PitchRollAxisX.LabelStyle.Format = "{0.00}";
				PitchRollAxisX.IsLabelAutoFit = false;
			}
			{
				AltitudeAxisX.Interval = 5;
				AltitudeAxisX.IsMarksNextToAxis = true;
				AltitudeAxisX.LabelStyle.Format = "{0.00}";
				AltitudeAxisX.IsLabelAutoFit = false;

				AltitudeAxisY.IsMarksNextToAxis = true;
			}
			{
				//LinearPhaseAxisX.Minimum = -100;
				//LinearPhaseAxisX.Maximum = 100;
				//LinearPhaseAxisY.Minimum = -100;
				//LinearPhaseAxisY.Maximum = 100;

				LinearPhaseAxisX.LabelStyle.Format = "{0.00}";
				LinearPhaseAxisY.LabelStyle.Format = "{0.00}";

				LinearPhaseAxisX.IsLabelAutoFit = false;
				LinearPhaseAxisY.IsLabelAutoFit = false;
			}
			{
				AngularPhaseDiagramAxisX.Minimum = -Math.PI;
				AngularPhaseDiagramAxisX.Maximum = Math.PI;
				AngularPhaseDiagramAxisY.Minimum = -Math.PI;
				AngularPhaseDiagramAxisY.Maximum = Math.PI;

				AngularPhaseDiagramAxisX.IsMarginVisible = true;
				AngularPhaseDiagramAxisY.IsMarginVisible = true;

				iLabel = AngularPhaseDiagramAxisX.CustomLabels.Add(Math.PI - Math.PI / 5, Math.PI + Math.PI / 5, "pi");
				iLabel.GridTicks = GridTickTypes.All;
				iLabel = AngularPhaseDiagramAxisX.CustomLabels.Add(-Math.PI + Math.PI / 5, -Math.PI - Math.PI / 5, "-pi");
				iLabel.GridTicks = GridTickTypes.All;

				iLabel = AngularPhaseDiagramAxisX.CustomLabels.Add(Math.PI / 2 + Math.PI / 5, Math.PI / 2 - Math.PI / 5, "pi/2");
				iLabel.GridTicks = GridTickTypes.All;
				iLabel = AngularPhaseDiagramAxisX.CustomLabels.Add(-Math.PI / 2 + Math.PI / 5, -Math.PI / 2 - Math.PI / 5, "-pi/2");
				iLabel.GridTicks = GridTickTypes.All;

				iLabel = AngularPhaseDiagramAxisY.CustomLabels.Add(Math.PI - Math.PI / 5, Math.PI + Math.PI / 5, "pi");
				iLabel.GridTicks = GridTickTypes.All;
				iLabel = AngularPhaseDiagramAxisY.CustomLabels.Add(-Math.PI + Math.PI / 5, -Math.PI - Math.PI / 5, "-pi");
				iLabel.GridTicks = GridTickTypes.All;

				iLabel = AngularPhaseDiagramAxisY.CustomLabels.Add(Math.PI / 2 + Math.PI / 5, Math.PI / 2 - Math.PI / 5, "pi/2");
				iLabel.GridTicks = GridTickTypes.All;
				iLabel = AngularPhaseDiagramAxisY.CustomLabels.Add(-Math.PI / 2 + Math.PI / 5, -Math.PI / 2 - Math.PI / 5, "-pi/2");
				iLabel.GridTicks = GridTickTypes.All;

				iLabel = AngularPhaseDiagramAxisX.CustomLabels.Add(Math.PI / 5, -Math.PI / 5, "0");
				iLabel.GridTicks = GridTickTypes.All;
				iLabel = AngularPhaseDiagramAxisY.CustomLabels.Add(Math.PI / 5, -Math.PI / 5, "0");
				iLabel.GridTicks = GridTickTypes.All;
			}
		}

		

		private void exitToolStripMenuItem_Click ( object sender, EventArgs e )
		{
			game.Exit();
		}


		private void timer1_Tick ( object sender, EventArgs e )
		{
			var world = game.GetService<World>();

			float yaw = 0, pitch = 0, roll = 0;

			ToAngles(world.quadrocopter.Transform, out yaw, out pitch, out roll);

			PitchRollDiagram.Series["Yaw"].Points.AddXY(world.worldTime.TotalGameTime.TotalSeconds, yaw);
			PitchRollDiagram.Series["Pitch"].Points.AddXY(world.worldTime.TotalGameTime.TotalSeconds, pitch);
			PitchRollDiagram.Series["Roll"].Points.AddXY(world.worldTime.TotalGameTime.TotalSeconds, roll);

			AngularPhaseDiagram.Series["Series1"].Points.AddXY(pitch, roll);

			LinearPhaseDiagram.Series["Series1"].Points.AddXY(world.quadrocopter.Position.X, -world.quadrocopter.Position.Z);
			AltitudeDiagram.Series["Series1"].Points.AddXY(world.worldTime.TotalGameTime.TotalSeconds, world.quadrocopter.Position.Y);

			if (AltitudeDiagram.Series["Series1"].Points.Count > 500)
			{
				AltitudeDiagram.Series["Series1"].Points.RemoveAt(0);
				LinearPhaseDiagram.Series["Series1"].Points.RemoveAt(0);

				PitchRollDiagram.Series["Yaw"].Points.RemoveAt(0);
				PitchRollDiagram.Series["Pitch"].Points.RemoveAt(0);
				PitchRollDiagram.Series["Roll"].Points.RemoveAt(0);

				AngularPhaseDiagram.Series["Series1"].Points.RemoveAt(0);


				AltitudeDiagram.ResetAutoValues();
				LinearPhaseDiagram.ResetAutoValues();
				PitchRollDiagram.ResetAutoValues();
				AngularPhaseDiagram.ResetAutoValues();
			}


			Axis PitchRollAxisX = PitchRollDiagram.ChartAreas["ChartArea1"].AxisX;
			PitchRollAxisX.Minimum = world.worldTime.TotalGameTime.TotalSeconds - 30.0;
			PitchRollAxisX.Maximum = world.worldTime.TotalGameTime.TotalSeconds + 2.0f;
			PitchRollDiagram.Invalidate();
			Axis AltitudeDiagramAxisX = AltitudeDiagram.ChartAreas["ChartArea1"].AxisX;
			AltitudeDiagramAxisX.Minimum = world.worldTime.TotalGameTime.TotalSeconds - 30.0;
			AltitudeDiagramAxisX.Maximum = world.worldTime.TotalGameTime.TotalSeconds + 2.0f;
			AltitudeDiagram.Invalidate();
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

		SerialPort	comPort;
		Thread		thread;


	}
}
