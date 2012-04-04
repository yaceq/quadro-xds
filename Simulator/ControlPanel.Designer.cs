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
			System.Windows.Forms.TreeNode treeNode1 = new System.Windows.Forms.TreeNode("Settings");
			System.Windows.Forms.TreeNode treeNode2 = new System.Windows.Forms.TreeNode("Quadrocopters");
			System.Windows.Forms.DataVisualization.Charting.ChartArea chartArea1 = new System.Windows.Forms.DataVisualization.Charting.ChartArea();
			System.Windows.Forms.DataVisualization.Charting.Legend legend1 = new System.Windows.Forms.DataVisualization.Charting.Legend();
			System.Windows.Forms.DataVisualization.Charting.Series series1 = new System.Windows.Forms.DataVisualization.Charting.Series();
			System.Windows.Forms.DataVisualization.Charting.ChartArea chartArea2 = new System.Windows.Forms.DataVisualization.Charting.ChartArea();
			System.Windows.Forms.DataVisualization.Charting.Legend legend2 = new System.Windows.Forms.DataVisualization.Charting.Legend();
			System.Windows.Forms.DataVisualization.Charting.Series series2 = new System.Windows.Forms.DataVisualization.Charting.Series();
			System.Windows.Forms.DataVisualization.Charting.ChartArea chartArea3 = new System.Windows.Forms.DataVisualization.Charting.ChartArea();
			System.Windows.Forms.DataVisualization.Charting.Legend legend3 = new System.Windows.Forms.DataVisualization.Charting.Legend();
			System.Windows.Forms.DataVisualization.Charting.Series series3 = new System.Windows.Forms.DataVisualization.Charting.Series();
			System.Windows.Forms.DataVisualization.Charting.ChartArea chartArea4 = new System.Windows.Forms.DataVisualization.Charting.ChartArea();
			System.Windows.Forms.DataVisualization.Charting.CustomLabel customLabel1 = new System.Windows.Forms.DataVisualization.Charting.CustomLabel();
			System.Windows.Forms.DataVisualization.Charting.Legend legend4 = new System.Windows.Forms.DataVisualization.Charting.Legend();
			System.Windows.Forms.DataVisualization.Charting.Series series4 = new System.Windows.Forms.DataVisualization.Charting.Series();
			this.menuStrip1 = new System.Windows.Forms.MenuStrip();
			this.simulatorToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
			this.addQuadrocopterToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
			this.removeQuadrocopterToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
			this.toolStripSeparator1 = new System.Windows.Forms.ToolStripSeparator();
			this.connectToTrackerToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
			this.toolStripSeparator2 = new System.Windows.Forms.ToolStripSeparator();
			this.exitToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
			this.propertyGrid = new System.Windows.Forms.PropertyGrid();
			this.ObjectTreeView = new System.Windows.Forms.TreeView();
			this.tableLayoutPanel1 = new System.Windows.Forms.TableLayoutPanel();
			this.PitchRollDiagram = new System.Windows.Forms.DataVisualization.Charting.Chart();
			this.AngularPhaseDiagram = new System.Windows.Forms.DataVisualization.Charting.Chart();
			this.AltitudeDiagram = new System.Windows.Forms.DataVisualization.Charting.Chart();
			this.LinearPhaseDiagram = new System.Windows.Forms.DataVisualization.Charting.Chart();
			this.splitter1 = new System.Windows.Forms.Splitter();
			this.splitter2 = new System.Windows.Forms.Splitter();
			this.timer1 = new System.Windows.Forms.Timer(this.components);
			this.menuStrip1.SuspendLayout();
			this.tableLayoutPanel1.SuspendLayout();
			((System.ComponentModel.ISupportInitialize)(this.PitchRollDiagram)).BeginInit();
			((System.ComponentModel.ISupportInitialize)(this.AngularPhaseDiagram)).BeginInit();
			((System.ComponentModel.ISupportInitialize)(this.AltitudeDiagram)).BeginInit();
			((System.ComponentModel.ISupportInitialize)(this.LinearPhaseDiagram)).BeginInit();
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
			// propertyGrid
			// 
			this.propertyGrid.Dock = System.Windows.Forms.DockStyle.Left;
			this.propertyGrid.Location = new System.Drawing.Point(146, 24);
			this.propertyGrid.Name = "propertyGrid";
			this.propertyGrid.Size = new System.Drawing.Size(283, 625);
			this.propertyGrid.TabIndex = 1;
			// 
			// ObjectTreeView
			// 
			this.ObjectTreeView.Dock = System.Windows.Forms.DockStyle.Left;
			this.ObjectTreeView.Location = new System.Drawing.Point(0, 24);
			this.ObjectTreeView.Name = "ObjectTreeView";
			treeNode1.Name = "Settings";
			treeNode1.Text = "Settings";
			treeNode2.Name = "Quadrocopters";
			treeNode2.Text = "Quadrocopters";
			this.ObjectTreeView.Nodes.AddRange(new System.Windows.Forms.TreeNode[] {
            treeNode1,
            treeNode2});
			this.ObjectTreeView.Size = new System.Drawing.Size(142, 625);
			this.ObjectTreeView.TabIndex = 2;
			this.ObjectTreeView.AfterSelect += new System.Windows.Forms.TreeViewEventHandler(this.treeView1_AfterSelect);
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
			this.tableLayoutPanel1.Location = new System.Drawing.Point(429, 24);
			this.tableLayoutPanel1.Name = "tableLayoutPanel1";
			this.tableLayoutPanel1.RowCount = 2;
			this.tableLayoutPanel1.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 50F));
			this.tableLayoutPanel1.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 50F));
			this.tableLayoutPanel1.Size = new System.Drawing.Size(700, 625);
			this.tableLayoutPanel1.TabIndex = 3;
			// 
			// PitchRollDiagram
			// 
			chartArea1.Name = "ChartArea1";
			this.PitchRollDiagram.ChartAreas.Add(chartArea1);
			this.PitchRollDiagram.Dock = System.Windows.Forms.DockStyle.Fill;
			legend1.Name = "Legend1";
			this.PitchRollDiagram.Legends.Add(legend1);
			this.PitchRollDiagram.Location = new System.Drawing.Point(353, 315);
			this.PitchRollDiagram.Name = "PitchRollDiagram";
			series1.ChartArea = "ChartArea1";
			series1.Legend = "Legend1";
			series1.Name = "Series1";
			this.PitchRollDiagram.Series.Add(series1);
			this.PitchRollDiagram.Size = new System.Drawing.Size(344, 307);
			this.PitchRollDiagram.TabIndex = 3;
			this.PitchRollDiagram.Text = "chart4";
			// 
			// AngularPhaseDiagram
			// 
			chartArea2.Name = "ChartArea1";
			this.AngularPhaseDiagram.ChartAreas.Add(chartArea2);
			this.AngularPhaseDiagram.Dock = System.Windows.Forms.DockStyle.Fill;
			legend2.Name = "Legend1";
			this.AngularPhaseDiagram.Legends.Add(legend2);
			this.AngularPhaseDiagram.Location = new System.Drawing.Point(3, 315);
			this.AngularPhaseDiagram.Name = "AngularPhaseDiagram";
			series2.ChartArea = "ChartArea1";
			series2.Legend = "Legend1";
			series2.Name = "Series1";
			this.AngularPhaseDiagram.Series.Add(series2);
			this.AngularPhaseDiagram.Size = new System.Drawing.Size(344, 307);
			this.AngularPhaseDiagram.TabIndex = 2;
			this.AngularPhaseDiagram.Text = "chart3";
			// 
			// AltitudeDiagram
			// 
			chartArea3.Name = "ChartArea1";
			this.AltitudeDiagram.ChartAreas.Add(chartArea3);
			this.AltitudeDiagram.Dock = System.Windows.Forms.DockStyle.Fill;
			legend3.Name = "Legend1";
			this.AltitudeDiagram.Legends.Add(legend3);
			this.AltitudeDiagram.Location = new System.Drawing.Point(353, 3);
			this.AltitudeDiagram.Name = "AltitudeDiagram";
			series3.ChartArea = "ChartArea1";
			series3.Legend = "Legend1";
			series3.Name = "Series1";
			this.AltitudeDiagram.Series.Add(series3);
			this.AltitudeDiagram.Size = new System.Drawing.Size(344, 306);
			this.AltitudeDiagram.TabIndex = 1;
			this.AltitudeDiagram.Text = "chart2";
			// 
			// LinearPhaseDiagram
			// 
			chartArea4.AxisX.CustomLabels.Add(customLabel1);
			chartArea4.Name = "ChartArea1";
			this.LinearPhaseDiagram.ChartAreas.Add(chartArea4);
			this.LinearPhaseDiagram.Dock = System.Windows.Forms.DockStyle.Fill;
			legend4.Name = "Legend1";
			this.LinearPhaseDiagram.Legends.Add(legend4);
			this.LinearPhaseDiagram.Location = new System.Drawing.Point(3, 3);
			this.LinearPhaseDiagram.Name = "LinearPhaseDiagram";
			series4.ChartArea = "ChartArea1";
			series4.ChartType = System.Windows.Forms.DataVisualization.Charting.SeriesChartType.Line;
			series4.IsVisibleInLegend = false;
			series4.Legend = "Legend1";
			series4.Name = "Series1";
			this.LinearPhaseDiagram.Series.Add(series4);
			this.LinearPhaseDiagram.Size = new System.Drawing.Size(344, 306);
			this.LinearPhaseDiagram.TabIndex = 0;
			this.LinearPhaseDiagram.Text = "chart1";
			// 
			// splitter1
			// 
			this.splitter1.Location = new System.Drawing.Point(429, 24);
			this.splitter1.Name = "splitter1";
			this.splitter1.Size = new System.Drawing.Size(4, 625);
			this.splitter1.TabIndex = 4;
			this.splitter1.TabStop = false;
			// 
			// splitter2
			// 
			this.splitter2.Location = new System.Drawing.Point(142, 24);
			this.splitter2.Name = "splitter2";
			this.splitter2.Size = new System.Drawing.Size(4, 625);
			this.splitter2.TabIndex = 5;
			this.splitter2.TabStop = false;
			// 
			// timer1
			// 
			this.timer1.Enabled = true;
			this.timer1.Interval = 30;
			this.timer1.Tick += new System.EventHandler(this.timer1_Tick);
			// 
			// ControlPanel
			// 
			this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
			this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
			this.ClientSize = new System.Drawing.Size(1129, 649);
			this.Controls.Add(this.splitter1);
			this.Controls.Add(this.tableLayoutPanel1);
			this.Controls.Add(this.propertyGrid);
			this.Controls.Add(this.splitter2);
			this.Controls.Add(this.ObjectTreeView);
			this.Controls.Add(this.menuStrip1);
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
		private System.Windows.Forms.PropertyGrid propertyGrid;
		private System.Windows.Forms.TreeView ObjectTreeView;
		private System.Windows.Forms.TableLayoutPanel tableLayoutPanel1;
		public  System.Windows.Forms.DataVisualization.Charting.Chart PitchRollDiagram;
		public  System.Windows.Forms.DataVisualization.Charting.Chart AngularPhaseDiagram;
		public  System.Windows.Forms.DataVisualization.Charting.Chart AltitudeDiagram;
		public  System.Windows.Forms.DataVisualization.Charting.Chart LinearPhaseDiagram;
		private System.Windows.Forms.Splitter splitter1;
		private System.Windows.Forms.Splitter splitter2;
		private System.Windows.Forms.Timer timer1;

	}
}