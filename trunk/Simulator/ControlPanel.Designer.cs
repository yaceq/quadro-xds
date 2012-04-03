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
			System.Windows.Forms.TreeNode treeNode3 = new System.Windows.Forms.TreeNode("Settings");
			System.Windows.Forms.TreeNode treeNode4 = new System.Windows.Forms.TreeNode("Quadrocopters");
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
			System.Windows.Forms.DataVisualization.Charting.Legend legend8 = new System.Windows.Forms.DataVisualization.Charting.Legend();
			System.Windows.Forms.DataVisualization.Charting.Series series8 = new System.Windows.Forms.DataVisualization.Charting.Series();
			this.menuStrip1 = new System.Windows.Forms.MenuStrip();
			this.simulatorToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
			this.addQuadrocopterToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
			this.removeQuadrocopterToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
			this.toolStripSeparator1 = new System.Windows.Forms.ToolStripSeparator();
			this.exitToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
			this.connectToTrackerToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
			this.toolStripSeparator2 = new System.Windows.Forms.ToolStripSeparator();
			this.propertyGrid = new System.Windows.Forms.PropertyGrid();
			this.ObjectTreeView = new System.Windows.Forms.TreeView();
			this.tableLayoutPanel1 = new System.Windows.Forms.TableLayoutPanel();
			this.chart1 = new System.Windows.Forms.DataVisualization.Charting.Chart();
			this.chart2 = new System.Windows.Forms.DataVisualization.Charting.Chart();
			this.chart3 = new System.Windows.Forms.DataVisualization.Charting.Chart();
			this.chart4 = new System.Windows.Forms.DataVisualization.Charting.Chart();
			this.splitter1 = new System.Windows.Forms.Splitter();
			this.splitter2 = new System.Windows.Forms.Splitter();
			this.menuStrip1.SuspendLayout();
			this.tableLayoutPanel1.SuspendLayout();
			((System.ComponentModel.ISupportInitialize)(this.chart1)).BeginInit();
			((System.ComponentModel.ISupportInitialize)(this.chart2)).BeginInit();
			((System.ComponentModel.ISupportInitialize)(this.chart3)).BeginInit();
			((System.ComponentModel.ISupportInitialize)(this.chart4)).BeginInit();
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
			// exitToolStripMenuItem
			// 
			this.exitToolStripMenuItem.Name = "exitToolStripMenuItem";
			this.exitToolStripMenuItem.Size = new System.Drawing.Size(194, 22);
			this.exitToolStripMenuItem.Text = "Exit";
			this.exitToolStripMenuItem.Click += new System.EventHandler(this.exitToolStripMenuItem_Click);
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
			treeNode3.Name = "Settings";
			treeNode3.Text = "Settings";
			treeNode4.Name = "Quadrocopters";
			treeNode4.Text = "Quadrocopters";
			this.ObjectTreeView.Nodes.AddRange(new System.Windows.Forms.TreeNode[] {
            treeNode3,
            treeNode4});
			this.ObjectTreeView.Size = new System.Drawing.Size(142, 625);
			this.ObjectTreeView.TabIndex = 2;
			this.ObjectTreeView.AfterSelect += new System.Windows.Forms.TreeViewEventHandler(this.treeView1_AfterSelect);
			// 
			// tableLayoutPanel1
			// 
			this.tableLayoutPanel1.ColumnCount = 2;
			this.tableLayoutPanel1.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 50F));
			this.tableLayoutPanel1.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 50F));
			this.tableLayoutPanel1.Controls.Add(this.chart4, 1, 1);
			this.tableLayoutPanel1.Controls.Add(this.chart3, 0, 1);
			this.tableLayoutPanel1.Controls.Add(this.chart2, 1, 0);
			this.tableLayoutPanel1.Controls.Add(this.chart1, 0, 0);
			this.tableLayoutPanel1.Dock = System.Windows.Forms.DockStyle.Fill;
			this.tableLayoutPanel1.Location = new System.Drawing.Point(429, 24);
			this.tableLayoutPanel1.Name = "tableLayoutPanel1";
			this.tableLayoutPanel1.RowCount = 2;
			this.tableLayoutPanel1.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 50F));
			this.tableLayoutPanel1.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 50F));
			this.tableLayoutPanel1.Size = new System.Drawing.Size(700, 625);
			this.tableLayoutPanel1.TabIndex = 3;
			// 
			// chart1
			// 
			chartArea5.Name = "ChartArea1";
			this.chart1.ChartAreas.Add(chartArea5);
			this.chart1.Dock = System.Windows.Forms.DockStyle.Fill;
			legend5.Name = "Legend1";
			this.chart1.Legends.Add(legend5);
			this.chart1.Location = new System.Drawing.Point(3, 3);
			this.chart1.Name = "chart1";
			series5.ChartArea = "ChartArea1";
			series5.Legend = "Legend1";
			series5.Name = "Series1";
			this.chart1.Series.Add(series5);
			this.chart1.Size = new System.Drawing.Size(344, 306);
			this.chart1.TabIndex = 0;
			this.chart1.Text = "chart1";
			// 
			// chart2
			// 
			chartArea6.Name = "ChartArea1";
			this.chart2.ChartAreas.Add(chartArea6);
			this.chart2.Dock = System.Windows.Forms.DockStyle.Fill;
			legend6.Name = "Legend1";
			this.chart2.Legends.Add(legend6);
			this.chart2.Location = new System.Drawing.Point(353, 3);
			this.chart2.Name = "chart2";
			series6.ChartArea = "ChartArea1";
			series6.Legend = "Legend1";
			series6.Name = "Series1";
			this.chart2.Series.Add(series6);
			this.chart2.Size = new System.Drawing.Size(344, 306);
			this.chart2.TabIndex = 1;
			this.chart2.Text = "chart2";
			// 
			// chart3
			// 
			chartArea7.Name = "ChartArea1";
			this.chart3.ChartAreas.Add(chartArea7);
			this.chart3.Dock = System.Windows.Forms.DockStyle.Fill;
			legend7.Name = "Legend1";
			this.chart3.Legends.Add(legend7);
			this.chart3.Location = new System.Drawing.Point(3, 315);
			this.chart3.Name = "chart3";
			series7.ChartArea = "ChartArea1";
			series7.Legend = "Legend1";
			series7.Name = "Series1";
			this.chart3.Series.Add(series7);
			this.chart3.Size = new System.Drawing.Size(344, 307);
			this.chart3.TabIndex = 2;
			this.chart3.Text = "chart3";
			// 
			// chart4
			// 
			chartArea8.Name = "ChartArea1";
			this.chart4.ChartAreas.Add(chartArea8);
			this.chart4.Dock = System.Windows.Forms.DockStyle.Fill;
			legend8.Name = "Legend1";
			this.chart4.Legends.Add(legend8);
			this.chart4.Location = new System.Drawing.Point(353, 315);
			this.chart4.Name = "chart4";
			series8.ChartArea = "ChartArea1";
			series8.Legend = "Legend1";
			series8.Name = "Series1";
			this.chart4.Series.Add(series8);
			this.chart4.Size = new System.Drawing.Size(344, 307);
			this.chart4.TabIndex = 3;
			this.chart4.Text = "chart4";
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
			((System.ComponentModel.ISupportInitialize)(this.chart1)).EndInit();
			((System.ComponentModel.ISupportInitialize)(this.chart2)).EndInit();
			((System.ComponentModel.ISupportInitialize)(this.chart3)).EndInit();
			((System.ComponentModel.ISupportInitialize)(this.chart4)).EndInit();
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
		private System.Windows.Forms.DataVisualization.Charting.Chart chart4;
		private System.Windows.Forms.DataVisualization.Charting.Chart chart3;
		private System.Windows.Forms.DataVisualization.Charting.Chart chart2;
		private System.Windows.Forms.DataVisualization.Charting.Chart chart1;
		private System.Windows.Forms.Splitter splitter1;
		private System.Windows.Forms.Splitter splitter2;

	}
}