using System;
using Misc;

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

            using (SimulatorGame game = new SimulatorGame())
            {
                game.Run();
            }
        }
    }
#endif
}

