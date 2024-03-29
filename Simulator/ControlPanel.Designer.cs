﻿namespace Simulator {
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
			this.menuStrip1 = new System.Windows.Forms.MenuStrip();
			this.simulatorToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
			this.addQuadrocopterToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
			this.removeQuadrocopterToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
			this.toolStripSeparator1 = new System.Windows.Forms.ToolStripSeparator();
			this.connectToTrackerToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
			this.disconnectTrackerToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
			this.toolStripSeparator2 = new System.Windows.Forms.ToolStripSeparator();
			this.exitToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
			this.quadrocopterToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
			this.connectToQuadrocopterViaCOMToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
			this.disconnectToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
			this.connectCOM6ToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
			this.splitter1 = new System.Windows.Forms.Splitter();
			this.timer1 = new System.Windows.Forms.Timer( this.components );
			this.statusStrip1 = new System.Windows.Forms.StatusStrip();
			this.splitContainer1 = new System.Windows.Forms.SplitContainer();
			this.settingsPropertyGrid = new System.Windows.Forms.PropertyGrid();
			this.quadrocopterPropertyGrid = new System.Windows.Forms.PropertyGrid();
			this.listBox1 = new System.Windows.Forms.ListBox();
			this.trackBar1 = new System.Windows.Forms.TrackBar();
			this.trackBar2 = new System.Windows.Forms.TrackBar();
			this.trackBar3 = new System.Windows.Forms.TrackBar();
			this.trackBar4 = new System.Windows.Forms.TrackBar();
			this.trackBar5 = new System.Windows.Forms.TrackBar();
			this.label1 = new System.Windows.Forms.Label();
			this.button2 = new System.Windows.Forms.Button();
			this.menuStrip1.SuspendLayout();
			((System.ComponentModel.ISupportInitialize)(this.splitContainer1)).BeginInit();
			this.splitContainer1.Panel1.SuspendLayout();
			this.splitContainer1.Panel2.SuspendLayout();
			this.splitContainer1.SuspendLayout();
			((System.ComponentModel.ISupportInitialize)(this.trackBar1)).BeginInit();
			((System.ComponentModel.ISupportInitialize)(this.trackBar2)).BeginInit();
			((System.ComponentModel.ISupportInitialize)(this.trackBar3)).BeginInit();
			((System.ComponentModel.ISupportInitialize)(this.trackBar4)).BeginInit();
			((System.ComponentModel.ISupportInitialize)(this.trackBar5)).BeginInit();
			this.SuspendLayout();
			// 
			// menuStrip1
			// 
			this.menuStrip1.Items.AddRange( new System.Windows.Forms.ToolStripItem[] {
            this.simulatorToolStripMenuItem,
            this.quadrocopterToolStripMenuItem} );
			this.menuStrip1.Location = new System.Drawing.Point( 0, 0 );
			this.menuStrip1.Name = "menuStrip1";
			this.menuStrip1.Size = new System.Drawing.Size( 511, 24 );
			this.menuStrip1.TabIndex = 0;
			this.menuStrip1.Text = "menuStrip1";
			// 
			// simulatorToolStripMenuItem
			// 
			this.simulatorToolStripMenuItem.DropDownItems.AddRange( new System.Windows.Forms.ToolStripItem[] {
            this.addQuadrocopterToolStripMenuItem,
            this.removeQuadrocopterToolStripMenuItem,
            this.toolStripSeparator1,
            this.connectToTrackerToolStripMenuItem,
            this.disconnectTrackerToolStripMenuItem,
            this.toolStripSeparator2,
            this.exitToolStripMenuItem} );
			this.simulatorToolStripMenuItem.Name = "simulatorToolStripMenuItem";
			this.simulatorToolStripMenuItem.Size = new System.Drawing.Size( 70, 20 );
			this.simulatorToolStripMenuItem.Text = "Simulator";
			// 
			// addQuadrocopterToolStripMenuItem
			// 
			this.addQuadrocopterToolStripMenuItem.Name = "addQuadrocopterToolStripMenuItem";
			this.addQuadrocopterToolStripMenuItem.Size = new System.Drawing.Size( 194, 22 );
			this.addQuadrocopterToolStripMenuItem.Text = "Add Quadrocopter";
			// 
			// removeQuadrocopterToolStripMenuItem
			// 
			this.removeQuadrocopterToolStripMenuItem.Name = "removeQuadrocopterToolStripMenuItem";
			this.removeQuadrocopterToolStripMenuItem.Size = new System.Drawing.Size( 194, 22 );
			this.removeQuadrocopterToolStripMenuItem.Text = "Remove Quadrocopter";
			// 
			// toolStripSeparator1
			// 
			this.toolStripSeparator1.Name = "toolStripSeparator1";
			this.toolStripSeparator1.Size = new System.Drawing.Size( 191, 6 );
			// 
			// connectToTrackerToolStripMenuItem
			// 
			this.connectToTrackerToolStripMenuItem.Name = "connectToTrackerToolStripMenuItem";
			this.connectToTrackerToolStripMenuItem.Size = new System.Drawing.Size( 194, 22 );
			this.connectToTrackerToolStripMenuItem.Text = "Connect to Tracker";
			this.connectToTrackerToolStripMenuItem.Click += new System.EventHandler( this.connectToTrackerToolStripMenuItem_Click );
			// 
			// disconnectTrackerToolStripMenuItem
			// 
			this.disconnectTrackerToolStripMenuItem.Name = "disconnectTrackerToolStripMenuItem";
			this.disconnectTrackerToolStripMenuItem.Size = new System.Drawing.Size( 194, 22 );
			this.disconnectTrackerToolStripMenuItem.Text = "Disconnect Tracker";
			this.disconnectTrackerToolStripMenuItem.Click += new System.EventHandler( this.disconnectTrackerToolStripMenuItem_Click );
			// 
			// toolStripSeparator2
			// 
			this.toolStripSeparator2.Name = "toolStripSeparator2";
			this.toolStripSeparator2.Size = new System.Drawing.Size( 191, 6 );
			// 
			// exitToolStripMenuItem
			// 
			this.exitToolStripMenuItem.Name = "exitToolStripMenuItem";
			this.exitToolStripMenuItem.Size = new System.Drawing.Size( 194, 22 );
			this.exitToolStripMenuItem.Text = "Exit";
			this.exitToolStripMenuItem.Click += new System.EventHandler( this.exitToolStripMenuItem_Click );
			// 
			// quadrocopterToolStripMenuItem
			// 
			this.quadrocopterToolStripMenuItem.DropDownItems.AddRange( new System.Windows.Forms.ToolStripItem[] {
            this.connectToQuadrocopterViaCOMToolStripMenuItem,
            this.disconnectToolStripMenuItem,
            this.connectCOM6ToolStripMenuItem} );
			this.quadrocopterToolStripMenuItem.Name = "quadrocopterToolStripMenuItem";
			this.quadrocopterToolStripMenuItem.Size = new System.Drawing.Size( 93, 20 );
			this.quadrocopterToolStripMenuItem.Text = "Quadrocopter";
			// 
			// connectToQuadrocopterViaCOMToolStripMenuItem
			// 
			this.connectToQuadrocopterViaCOMToolStripMenuItem.Name = "connectToQuadrocopterViaCOMToolStripMenuItem";
			this.connectToQuadrocopterViaCOMToolStripMenuItem.Size = new System.Drawing.Size( 259, 22 );
			this.connectToQuadrocopterViaCOMToolStripMenuItem.Text = "Connect to Quadrocopter via COM";
			this.connectToQuadrocopterViaCOMToolStripMenuItem.Click += new System.EventHandler( this.connectToQuadrocopterViaCOMToolStripMenuItem_Click );
			// 
			// disconnectToolStripMenuItem
			// 
			this.disconnectToolStripMenuItem.Name = "disconnectToolStripMenuItem";
			this.disconnectToolStripMenuItem.Size = new System.Drawing.Size( 259, 22 );
			this.disconnectToolStripMenuItem.Text = "Disconnect";
			this.disconnectToolStripMenuItem.Click += new System.EventHandler( this.disconnectToolStripMenuItem_Click );
			// 
			// connectCOM6ToolStripMenuItem
			// 
			this.connectCOM6ToolStripMenuItem.Name = "connectCOM6ToolStripMenuItem";
			this.connectCOM6ToolStripMenuItem.Size = new System.Drawing.Size( 259, 22 );
			this.connectCOM6ToolStripMenuItem.Text = "Connect COM6";
			this.connectCOM6ToolStripMenuItem.Click += new System.EventHandler( this.connectCOM6ToolStripMenuItem_Click );
			// 
			// splitter1
			// 
			this.splitter1.Location = new System.Drawing.Point( 317, 24 );
			this.splitter1.Name = "splitter1";
			this.splitter1.Size = new System.Drawing.Size( 10, 672 );
			this.splitter1.TabIndex = 4;
			this.splitter1.TabStop = false;
			// 
			// statusStrip1
			// 
			this.statusStrip1.Location = new System.Drawing.Point( 0, 696 );
			this.statusStrip1.Name = "statusStrip1";
			this.statusStrip1.Size = new System.Drawing.Size( 511, 22 );
			this.statusStrip1.TabIndex = 7;
			this.statusStrip1.Text = "statusStrip1";
			// 
			// splitContainer1
			// 
			this.splitContainer1.Dock = System.Windows.Forms.DockStyle.Left;
			this.splitContainer1.Location = new System.Drawing.Point( 0, 24 );
			this.splitContainer1.Name = "splitContainer1";
			this.splitContainer1.Orientation = System.Windows.Forms.Orientation.Horizontal;
			// 
			// splitContainer1.Panel1
			// 
			this.splitContainer1.Panel1.Controls.Add( this.settingsPropertyGrid );
			// 
			// splitContainer1.Panel2
			// 
			this.splitContainer1.Panel2.Controls.Add( this.quadrocopterPropertyGrid );
			this.splitContainer1.Panel2.Controls.Add( this.listBox1 );
			this.splitContainer1.Size = new System.Drawing.Size( 317, 672 );
			this.splitContainer1.SplitterDistance = 301;
			this.splitContainer1.TabIndex = 8;
			// 
			// settingsPropertyGrid
			// 
			this.settingsPropertyGrid.Dock = System.Windows.Forms.DockStyle.Fill;
			this.settingsPropertyGrid.HelpVisible = false;
			this.settingsPropertyGrid.Location = new System.Drawing.Point( 0, 0 );
			this.settingsPropertyGrid.Name = "settingsPropertyGrid";
			this.settingsPropertyGrid.Size = new System.Drawing.Size( 317, 301 );
			this.settingsPropertyGrid.TabIndex = 2;
			// 
			// quadrocopterPropertyGrid
			// 
			this.quadrocopterPropertyGrid.Dock = System.Windows.Forms.DockStyle.Fill;
			this.quadrocopterPropertyGrid.HelpVisible = false;
			this.quadrocopterPropertyGrid.Location = new System.Drawing.Point( 0, 0 );
			this.quadrocopterPropertyGrid.Name = "quadrocopterPropertyGrid";
			this.quadrocopterPropertyGrid.Size = new System.Drawing.Size( 317, 285 );
			this.quadrocopterPropertyGrid.TabIndex = 7;
			// 
			// listBox1
			// 
			this.listBox1.Dock = System.Windows.Forms.DockStyle.Bottom;
			this.listBox1.FormattingEnabled = true;
			this.listBox1.Location = new System.Drawing.Point( 0, 285 );
			this.listBox1.Name = "listBox1";
			this.listBox1.Size = new System.Drawing.Size( 317, 82 );
			this.listBox1.TabIndex = 4;
			this.listBox1.Click += new System.EventHandler( this.listBox1_Click );
			// 
			// trackBar1
			// 
			this.trackBar1.Location = new System.Drawing.Point( 333, 27 );
			this.trackBar1.Maximum = 180;
			this.trackBar1.Minimum = 10;
			this.trackBar1.Name = "trackBar1";
			this.trackBar1.Size = new System.Drawing.Size( 166, 45 );
			this.trackBar1.TabIndex = 9;
			this.trackBar1.TickFrequency = 10;
			this.trackBar1.Value = 10;
			this.trackBar1.Scroll += new System.EventHandler( this.trackBar1_Scroll );
			// 
			// trackBar2
			// 
			this.trackBar2.Location = new System.Drawing.Point( 333, 78 );
			this.trackBar2.Maximum = 180;
			this.trackBar2.Minimum = 10;
			this.trackBar2.Name = "trackBar2";
			this.trackBar2.Size = new System.Drawing.Size( 166, 45 );
			this.trackBar2.TabIndex = 10;
			this.trackBar2.TickFrequency = 10;
			this.trackBar2.Value = 10;
			this.trackBar2.Scroll += new System.EventHandler( this.trackBar1_Scroll );
			// 
			// trackBar3
			// 
			this.trackBar3.Location = new System.Drawing.Point( 333, 129 );
			this.trackBar3.Maximum = 180;
			this.trackBar3.Minimum = 10;
			this.trackBar3.Name = "trackBar3";
			this.trackBar3.Size = new System.Drawing.Size( 166, 45 );
			this.trackBar3.TabIndex = 11;
			this.trackBar3.TickFrequency = 10;
			this.trackBar3.Value = 10;
			this.trackBar3.Scroll += new System.EventHandler( this.trackBar1_Scroll );
			// 
			// trackBar4
			// 
			this.trackBar4.Location = new System.Drawing.Point( 333, 180 );
			this.trackBar4.Maximum = 180;
			this.trackBar4.Minimum = 10;
			this.trackBar4.Name = "trackBar4";
			this.trackBar4.Size = new System.Drawing.Size( 166, 45 );
			this.trackBar4.TabIndex = 12;
			this.trackBar4.TickFrequency = 10;
			this.trackBar4.Value = 10;
			this.trackBar4.Scroll += new System.EventHandler( this.trackBar1_Scroll );
			// 
			// trackBar5
			// 
			this.trackBar5.LargeChange = 10;
			this.trackBar5.Location = new System.Drawing.Point( 333, 355 );
			this.trackBar5.Maximum = 180;
			this.trackBar5.Name = "trackBar5";
			this.trackBar5.Size = new System.Drawing.Size( 166, 45 );
			this.trackBar5.SmallChange = 10;
			this.trackBar5.TabIndex = 14;
			this.trackBar5.TickFrequency = 10;
			this.trackBar5.Scroll += new System.EventHandler( this.trackBar5_Scroll );
			// 
			// label1
			// 
			this.label1.AutoSize = true;
			this.label1.Location = new System.Drawing.Point( 334, 406 );
			this.label1.Name = "label1";
			this.label1.Size = new System.Drawing.Size( 37, 13 );
			this.label1.TabIndex = 15;
			this.label1.Text = "Thrust";
			// 
			// button2
			// 
			this.button2.Location = new System.Drawing.Point( 333, 249 );
			this.button2.Name = "button2";
			this.button2.Size = new System.Drawing.Size( 166, 57 );
			this.button2.TabIndex = 16;
			this.button2.Text = "Start Logging";
			this.button2.UseVisualStyleBackColor = true;
			this.button2.Click += new System.EventHandler( this.button2_Click );
			// 
			// ControlPanel
			// 
			this.AutoScaleDimensions = new System.Drawing.SizeF( 6F, 13F );
			this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
			this.ClientSize = new System.Drawing.Size( 511, 718 );
			this.Controls.Add( this.button2 );
			this.Controls.Add( this.label1 );
			this.Controls.Add( this.trackBar5 );
			this.Controls.Add( this.trackBar4 );
			this.Controls.Add( this.trackBar3 );
			this.Controls.Add( this.trackBar2 );
			this.Controls.Add( this.trackBar1 );
			this.Controls.Add( this.splitter1 );
			this.Controls.Add( this.splitContainer1 );
			this.Controls.Add( this.menuStrip1 );
			this.Controls.Add( this.statusStrip1 );
			this.MainMenuStrip = this.menuStrip1;
			this.Name = "ControlPanel";
			this.Text = "ControlPanel";
			this.menuStrip1.ResumeLayout( false );
			this.menuStrip1.PerformLayout();
			this.splitContainer1.Panel1.ResumeLayout( false );
			this.splitContainer1.Panel2.ResumeLayout( false );
			((System.ComponentModel.ISupportInitialize)(this.splitContainer1)).EndInit();
			this.splitContainer1.ResumeLayout( false );
			((System.ComponentModel.ISupportInitialize)(this.trackBar1)).EndInit();
			((System.ComponentModel.ISupportInitialize)(this.trackBar2)).EndInit();
			((System.ComponentModel.ISupportInitialize)(this.trackBar3)).EndInit();
			((System.ComponentModel.ISupportInitialize)(this.trackBar4)).EndInit();
			((System.ComponentModel.ISupportInitialize)(this.trackBar5)).EndInit();
			this.ResumeLayout( false );
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
		private System.Windows.Forms.Splitter splitter1;
		private System.Windows.Forms.Timer timer1;
		private System.Windows.Forms.ToolStripMenuItem disconnectTrackerToolStripMenuItem;
		private System.Windows.Forms.StatusStrip statusStrip1;
		private System.Windows.Forms.SplitContainer splitContainer1;
		private System.Windows.Forms.PropertyGrid settingsPropertyGrid;
		private System.Windows.Forms.PropertyGrid quadrocopterPropertyGrid;
		private System.Windows.Forms.ListBox listBox1;
		private System.Windows.Forms.ToolStripMenuItem quadrocopterToolStripMenuItem;
		private System.Windows.Forms.ToolStripMenuItem connectToQuadrocopterViaCOMToolStripMenuItem;
		private System.Windows.Forms.ToolStripMenuItem disconnectToolStripMenuItem;
		public System.Windows.Forms.TrackBar trackBar1;
		public System.Windows.Forms.TrackBar trackBar2;
		public System.Windows.Forms.TrackBar trackBar3;
		public System.Windows.Forms.TrackBar trackBar4;
		private System.Windows.Forms.ToolStripMenuItem connectCOM6ToolStripMenuItem;
		public System.Windows.Forms.TrackBar trackBar5;
		private System.Windows.Forms.Label label1;
		private System.Windows.Forms.Button button2;

	}
}