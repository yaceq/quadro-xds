using System;
using Misc;
using System.Windows.Forms;

namespace Simulator {
#if WINDOWS || XBOX
    static class Program
    {
        /// <summary>
        /// The main entry point for the application.
        /// </summary>
        static void Main(string[] args)
        {
			Core.InitEnvironment();

			Application.EnableVisualStyles();
			Application.SetCompatibleTextRenderingDefault( false );

            using (SimulatorGame game = new SimulatorGame())
            {
                game.Run();
            }
        }
    }
#endif
}

