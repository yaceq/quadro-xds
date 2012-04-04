namespace Simulator {
	partial class ControlPanel {
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.IContainer components = null;

		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		/// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
		protected override void Dispose ( bool disposing )
		{
			if (disposing && (components != null)) {
				components.Dispose();
			}
			base.Dispose( disposing );
		}

		#region Windows Form Designer generated code

		/// <summary>
		/// Required method for Designer support - do not modify
		/// the contents of this method with the code editor.
		/// </summary>
		private void InitializeComponent ()
		{
			this.components = new System.ComponentModel.Container();
			System.Windows.Forms.DataVisualization.Charting.ChartArea chartArea5 = new System.Windows.Forms.DataVisualization.Charting.ChartArea();
			System.Windows.Forms.DataVisualization.Charting.Legend legend5 = new System.Windows.Forms.DataVisualization.Charting.Legend();
			System.Windows.Forms.DataVisualization.Charting.Series series5 = new System.Windows.Forms.DataVisualization.Charting.Series();
			System.Windows.Forms.DataVisualization.Charting.ChartArea chartArea6 = new System.Windows.Forms.DataVisualization.Charting.ChartArea();
			System.Windows.Forms.DataVisualization.Charting.Legend legend6 = new System.Windows.Forms.DataVisualization.Charting.Legend();
			System.Windows.Forms.DataVisualization.Charting.Series series6 = new System.Windows.Forms.DataVisualization.Charting.Series();
			System.Windows.Forms.DataVisualization.Charting.ChartArea chartArea7 = new System.Windows.Forms.DataVisualization.Charting.ChartArea();
			System.Windows.Forms.DataVisualization.Charting.Legend legend7 = new System.Windows.Forms.DataVisualization.Charting.Legend();
			System.Windows.Forms.DataVisualization.Charting.Series series7 = new System.Windows.Forms.DataVisualization.Charting.Series();
			System.Windows.Forms.DataVisualization.Charting.ChartArea chartArea8 = new System.Windows.Forms.DataVisualization.Charting.ChartArea();
			System.Windows.Forms.DataVisualization.Charting.CustomLabel customLabel2 = new System.Windows.Forms.DataVisualization.Charting.CustomLabel();
			System.Windows.Forms.DataVisualization.Charting.Legend legend8 = new System.Windows.Forms.DataVisualization.Charting.Legend();
			System.Windows.Forms.DataVisualization.Charting.Series series8 = new System.Windows.Forms.DataVisualization.Charting.Series();
			this.menuStrip1 = new System.Windows.Forms.MenuStrip();
			this.simulatorToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
			this.addQuadrocopterToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
			this.removeQuadrocopterToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
			this.toolStripSeparator1 = new System.Windows.Forms.ToolStripSeparator();
			this.connectToTrackerToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
			this.toolStripSeparator2 = new System.Windows.Forms.ToolStripSeparator();
			this.exitToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
			this.tableLayoutPanel1 = new System.Windows.Forms.TableLayoutPanel();
			this.PitchRollDiagram = new System.Windows.Forms.DataVisualization.Charting.Chart();
			this.AngularPhaseDiagram = new System.Windows.Forms.DataVisualization.Charting.Chart();
			this.AltitudeDiagram = new System.Windows.Forms.DataVisualization.Charting.Chart();
			this.LinearPhaseDiagram = new System.Windows.Forms.DataVisualization.Charting.Chart();
			this.splitter1 = new System.Windows.Forms.Splitter();
			this.timer1 = new System.Windows.Forms.Timer(this.components);
			this.disconnectTrackerToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
			this.statusStrip1 = new System.Windows.Forms.StatusStrip();
			this.splitContainer1 = new System.Windows.Forms.SplitContainer();
			this.settingsPropertyGrid = new System.Windows.Forms.PropertyGrid();
			this.quadrocopterPropertyGrid = new System.Windows.Forms.PropertyGrid();
			this.menuStrip1.SuspendLayout();
			this.tableLayoutPanel1.SuspendLayout();
			((System.ComponentModel.ISupportInitialize)(this.PitchRollDiagram)).BeginInit();
			((System.ComponentModel.ISupportInitialize)(this.AngularPhaseDiagram)).BeginInit();
			((System.ComponentModel.ISupportInitialize)(this.AltitudeDiagram)).BeginInit();
			((System.ComponentModel.ISupportInitialize)(this.LinearPhaseDiagram)).BeginInit();
			((System.ComponentModel.ISupportInitialize)(this.splitContainer1)).BeginInit();
			this.splitContainer1.Panel1.SuspendLayout();
			this.splitContainer1.Panel2.SuspendLayout();
			this.splitContainer1.SuspendLayout();
			this.SuspendLayout();
			// 
			// menuStrip1
			// 
			this.menuStrip1.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.simulatorToolStripMenuItem});
			this.menuStrip1.Location = new System.Drawing.Point(0, 0);
			this.menuStrip1.Name = "menuStrip1";
			this.menuStrip1.Size = new System.Drawing.Size(1129, 24);
			this.menuStrip1.TabIndex = 0;
			this.menuStrip1.Text = "menuStrip1";
			// 
			// simulatorToolStripMenuItem
			// 
			this.simulatorToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.addQuadrocopterToolStripMenuItem,
            this.removeQuadrocopterToolStripMenuItem,
            this.toolStripSeparator1,
            this.connectToTrackerToolStripMenuItem,
            this.disconnectTrackerToolStripMenuItem,
            this.toolStripSeparator2,
            this.exitToolStripMenuItem});
			this.simulatorToolStripMenuItem.Name = "simulatorToolStripMenuItem";
			this.simulatorToolStripMenuItem.Size = new System.Drawing.Size(70, 20);
			this.simulatorToolStripMenuItem.Text = "Simulator";
			// 
			// addQuadrocopterToolStripMenuItem
			// 
			this.addQuadrocopterToolStripMenuItem.Name = "addQuadrocopterToolStripMenuItem";
			this.addQuadrocopterToolStripMenuItem.Size = new System.Drawing.Size(194, 22);
			this.addQuadrocopterToolStripMenuItem.Text = "Add Quadrocopter";
			// 
			// removeQuadrocopterToolStripMenuItem
			// 
			this.removeQuadrocopterToolStripMenuItem.Name = "removeQuadrocopterToolStripMenuItem";
			this.removeQuadrocopterToolStripMenuItem.Size = new System.Drawing.Size(194, 22);
			this.removeQuadrocopterToolStripMenuItem.Text = "Remove Quadrocopter";
			// 
			// toolStripSeparator1
			// 
			this.toolStripSeparator1.Name = "toolStripSeparator1";
			this.toolStripSeparator1.Size = new System.Drawing.Size(191, 6);
			// 
			// connectToTrackerToolStripMenuItem
			// 
			this.connectToTrackerToolStripMenuItem.Name = "connectToTrackerToolStripMenuItem";
			this.connectToTrackerToolStripMenuItem.Size = new System.Drawing.Size(194, 22);
			this.connectToTrackerToolStripMenuItem.Text = "Connect to Tracker";
			this.connectToTrackerToolStripMenuItem.Click += new System.EventHandler(this.connectToTrackerToolStripMenuItem_Click);
			// 
			// toolStripSeparator2
			// 
			this.toolStripSeparator2.Name = "toolStripSeparator2";
			this.toolStripSeparator2.Size = new System.Drawing.Size(191, 6);
			// 
			// exitToolStripMenuItem
			// 
			this.exitToolStripMenuItem.Name = "exitToolStripMenuItem";
			this.exitToolStripMenuItem.Size = new System.Drawing.Size(194, 22);
			this.exitToolStripMenuItem.Text = "Exit";
			this.exitToolStripMenuItem.Click += new System.EventHandler(this.exitToolStripMenuItem_Click);
			// 
			// tableLayoutPanel1
			// 
			this.tableLayoutPanel1.ColumnCount = 2;
			this.tableLayoutPanel1.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 50F));
			this.tableLayoutPanel1.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 50F));
			this.tableLayoutPanel1.Controls.Add(this.PitchRollDiagram, 1, 1);
			this.tableLayoutPanel1.Controls.Add(this.AngularPhaseDiagram, 0, 1);
			this.tableLayoutPanel1.Controls.Add(this.AltitudeDiagram, 1, 0);
			this.tableLayoutPanel1.Controls.Add(this.LinearPhaseDiagram, 0, 0);
			this.tableLayoutPanel1.Dock = System.Windows.Forms.DockStyle.Fill;
			this.tableLayoutPanel1.Location = new System.Drawing.Point(300, 24);
			this.tableLayoutPanel1.Name = "tableLayoutPanel1";
			this.tableLayoutPanel1.RowCount = 2;
			this.tableLayoutPanel1.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 50F));
			this.tableLayoutPanel1.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 50F));
			this.tableLayoutPanel1.Size = new System.Drawing.Size(829, 603);
			this.tableLayoutPanel1.TabIndex = 3;
			// 
			// PitchRollDiagram
			// 
			chartArea5.Name = "ChartArea1";
			this.PitchRollDiagram.ChartAreas.Add(chartArea5);
			this.PitchRollDiagram.Dock = System.Windows.Forms.DockStyle.Fill;
			legend5.Name = "Legend1";
			this.PitchRollDiagram.Legends.Add(legend5);
			this.PitchRollDiagram.Location = new System.Drawing.Point(417, 304);
			this.PitchRollDiagram.Name = "PitchRollDiagram";
			series5.ChartArea = "ChartArea1";
			series5.Legend = "Legend1";
			series5.Name = "Series1";
			this.PitchRollDiagram.Series.Add(series5);
			this.PitchRollDiagram.Size = new System.Drawing.Size(409, 296);
			this.PitchRollDiagram.TabIndex = 3;
			this.PitchRollDiagram.Text = "chart4";
			// 
			// AngularPhaseDiagram
			// 
			chartArea6.Name = "ChartArea1";
			this.AngularPhaseDiagram.ChartAreas.Add(chartArea6);
			this.AngularPhaseDiagram.Dock = System.Windows.Forms.DockStyle.Fill;
			legend6.Name = "Legend1";
			this.AngularPhaseDiagram.Legends.Add(legend6);
			this.AngularPhaseDiagram.Location = new System.Drawing.Point(3, 304);
			this.AngularPhaseDiagram.Name = "AngularPhaseDiagram";
			series6.ChartArea = "ChartArea1";
			series6.Legend = "Legend1";
			series6.Name = "Series1";
			this.AngularPhaseDiagram.Series.Add(series6);
			this.AngularPhaseDiagram.Size = new System.Drawing.Size(408, 296);
			this.AngularPhaseDiagram.TabIndex = 2;
			this.AngularPhaseDiagram.Text = "chart3";
			// 
			// AltitudeDiagram
			// 
			chartArea7.Name = "ChartArea1";
			this.AltitudeDiagram.ChartAreas.Add(chartArea7);
			this.AltitudeDiagram.Dock = System.Windows.Forms.DockStyle.Fill;
			legend7.Name = "Legend1";
			this.AltitudeDiagram.Legends.Add(legend7);
			this.AltitudeDiagram.Location = new System.Drawing.Point(417, 3);
			this.AltitudeDiagram.Name = "AltitudeDiagram";
			series7.ChartArea = "ChartArea1";
			series7.Legend = "Legend1";
			series7.Name = "Series1";
			this.AltitudeDiagram.Series.Add(series7);
			this.AltitudeDiagram.Size = new System.Drawing.Size(409, 295);
			this.AltitudeDiagram.TabIndex = 1;
			this.AltitudeDiagram.Text = "chart2";
			// 
			// LinearPhaseDiagram
			// 
			chartArea8.AxisX.CustomLabels.Add(customLabel2);
			chartArea8.Name = "ChartArea1";
			this.LinearPhaseDiagram.ChartAreas.Add(chartArea8);
			this.LinearPhaseDiagram.Dock = System.Windows.Forms.DockStyle.Fill;
			legend8.Name = "Legend1";
			this.LinearPhaseDiagram.Legends.Add(legend8);
			this.LinearPhaseDiagram.Location = new System.Drawing.Point(3, 3);
			this.LinearPhaseDiagram.Name = "LinearPhaseDiagram";
			series8.ChartArea = "ChartArea1";
			series8.ChartType = System.Windows.Forms.DataVisualization.Charting.SeriesChartType.Line;
			series8.IsVisibleInLegend = false;
			series8.Legend = "Legend1";
			series8.Name = "Series1";
			this.LinearPhaseDiagram.Series.Add(series8);
			this.LinearPhaseDiagram.Size = new System.Drawing.Size(408, 295);
			this.LinearPhaseDiagram.TabIndex = 0;
			this.LinearPhaseDiagram.Text = "chart1";
			// 
			// splitter1
			// 
			this.splitter1.Location = new System.Drawing.Point(300, 24);
			this.splitter1.Name = "splitter1";
			this.splitter1.Size = new System.Drawing.Size(4, 603);
			this.splitter1.TabIndex = 4;
			this.splitter1.TabStop = false;
			// 
			// timer1
			// 
			this.timer1.Enabled = true;
			this.timer1.Interval = 30;
			this.timer1.Tick += new System.EventHandler(this.timer1_Tick);
			// 
			// disconnectTrackerToolStripMenuItem
			// 
			this.disconnectTrackerToolStripMenuItem.Name = "disconnectTrackerToolStripMenuItem";
			this.disconnectTrackerToolStripMenuItem.Size = new System.Drawing.Size(194, 22);
			this.disconnectTrackerToolStripMenuItem.Text = "Disconnect Tracker";
			this.disconnectTrackerToolStripMenuItem.Click += new System.EventHandler(this.disconnectTrackerToolStripMenuItem_Click);
			// 
			// statusStrip1
			// 
			this.statusStrip1.Location = new System.Drawing.Point(0, 627);
			this.statusStrip1.Name = "statusStrip1";
			this.statusStrip1.Size = new System.Drawing.Size(1129, 22);
			this.statusStrip1.TabIndex = 7;
			this.statusStrip1.Text = "statusStrip1";
			// 
			// splitContainer1
			// 
			this.splitContainer1.Dock = System.Windows.Forms.DockStyle.Left;
			this.splitContainer1.Location = new System.Drawing.Point(0, 24);
			this.splitContainer1.Name = "splitContainer1";
			this.splitContainer1.Orientation = System.Windows.Forms.Orientation.Horizontal;
			// 
			// splitContainer1.Panel1
			// 
			this.splitContainer1.Panel1.Controls.Add(this.settingsPropertyGrid);
			// 
			// splitContainer1.Panel2
			// 
			this.splitContainer1.Panel2.Controls.Add(this.quadrocopterPropertyGrid);
			this.splitContainer1.Size = new System.Drawing.Size(300, 603);
			this.splitContainer1.SplitterDistance = 280;
			this.splitContainer1.TabIndex = 8;
			// 
			// settingsPropertyGrid
			// 
			this.settingsPropertyGrid.Dock = System.Windows.Forms.DockStyle.Fill;
			this.settingsPropertyGrid.Location = new System.Drawing.Point(0, 0);
			this.settingsPropertyGrid.Name = "settingsPropertyGrid";
			this.settingsPropertyGrid.Size = new System.Drawing.Size(300, 280);
			this.settingsPropertyGrid.TabIndex = 2;
			// 
			// quadrocopterPropertyGrid
			// 
			this.quadrocopterPropertyGrid.Dock = System.Windows.Forms.DockStyle.Fill;
			this.quadrocopterPropertyGrid.Location = new System.Drawing.Point(0, 0);
			this.quadrocopterPropertyGrid.Name = "quadrocopterPropertyGrid";
			this.quadrocopterPropertyGrid.Size = new System.Drawing.Size(300, 319);
			this.quadrocopterPropertyGrid.TabIndex = 7;
			// 
			// ControlPanel
			// 
			this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
			this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
			this.ClientSize = new System.Drawing.Size(1129, 649);
			this.Controls.Add(this.splitter1);
			this.Controls.Add(this.tableLayoutPanel1);
			this.Controls.Add(this.splitContainer1);
			this.Controls.Add(this.menuStrip1);
			this.Controls.Add(this.statusStrip1);
			this.MainMenuStrip = this.menuStrip1;
			this.Name = "ControlPanel";
			this.Text = "ControlPanel";
			this.menuStrip1.ResumeLayout(false);
			this.menuStrip1.PerformLayout();
			this.tableLayoutPanel1.ResumeLayout(false);
			((System.ComponentModel.ISupportInitialize)(this.PitchRollDiagram)).EndInit();
			((System.ComponentModel.ISupportInitialize)(this.AngularPhaseDiagram)).EndInit();
			((System.ComponentModel.ISupportInitialize)(this.AltitudeDiagram)).EndInit();
			((System.ComponentModel.ISupportInitialize)(this.LinearPhaseDiagram)).EndInit();
			this.splitContainer1.Panel1.ResumeLayout(false);
			this.splitContainer1.Panel2.ResumeLayout(false);
			((System.ComponentModel.ISupportInitialize)(this.splitContainer1)).EndInit();
			this.splitContainer1.ResumeLayout(false);
			this.ResumeLayout(false);
			this.PerformLayout();

		}

		#endregion

		private System.Windows.Forms.MenuStrip menuStrip1;
		private System.Windows.Forms.ToolStripMenuItem simulatorToolStripMenuItem;
		private System.Windows.Forms.ToolStripMenuItem addQuadrocopterToolStripMenuItem;
		private System.Windows.Forms.ToolStripMenuItem removeQuadrocopterToolStripMenuItem;
		private System.Windows.Forms.ToolStripSeparator toolStripSeparator1;
		private System.Windows.Forms.ToolStripMenuItem connectToTrackerToolStripMenuItem;
		private System.Windows.Forms.ToolStripSeparator toolStripSeparator2;
		private System.Windows.Forms.ToolStripMenuItem exitToolStripMenuItem;
		private System.Windows.Forms.TableLayoutPanel tableLayoutPanel1;
		public  System.Windows.Forms.DataVisualization.Charting.Chart PitchRollDiagram;
		public  System.Windows.Forms.DataVisualization.Charting.Chart AngularPhaseDiagram;
		public  System.Windows.Forms.DataVisualization.Charting.Chart AltitudeDiagram;
		public  System.Windows.Forms.DataVisualization.Charting.Chart LinearPhaseDiagram;
		private System.Windows.Forms.Splitter splitter1;
		private System.Windows.Forms.Timer timer1;
		private System.Windows.Forms.ToolStripMenuItem disconnectTrackerToolStripMenuItem;
		private System.Windows.Forms.StatusStrip statusStrip1;
		private System.Windows.Forms.SplitContainer splitContainer1;
		private System.Windows.Forms.PropertyGrid settingsPropertyGrid;
		private System.Windows.Forms.PropertyGrid quadrocopterPropertyGrid;

	}
}