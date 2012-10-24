using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Audio;
using Microsoft.Xna.Framework.Content;
using Microsoft.Xna.Framework.GamerServices;
using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework.Input;
using Microsoft.Xna.Framework.Media;

namespace Simulator
{
    /*public class QuadOnboardCamera
    {
        Vector3 Position { set; get; }
        Vector3 Direction { set; get; }
        float FOV { set; get; }
        public bool isDraw = true;

        public RenderTarget2D cVPort;

        public QuadOnboardCamera(GraphicsDevice dev)
        {
            cVPort = new RenderTarget2D(dev, 400, 400, true, SurfaceFormat.Color, DepthFormat.Depth24, 0, RenderTargetUsage.DiscardContents);

            Position = new Vector3(0.0f, 0.0f, 0.0f);
            FOV = 90;
            Direction = Vector3.Forward;
        }

        public void GetPV ( Matrix QuadTransform, out Matrix proj, out Matrix camView )
        {
            var WorldPos = Vector3.Transform(Position, QuadTransform);
            var WorldLookAt = Vector3.Transform(Position + Direction, QuadTransform);
            
            QuadTransform.Translation = Vector3.Zero;
            var WorldUp = Vector3.Transform(Vector3.Up, QuadTransform);

            proj = Matrix.CreatePerspectiveFieldOfView(MathHelper.ToRadians(FOV), 1.0f, 0.01f, 5000.0f);
            camView = Matrix.CreateLookAt(WorldPos, WorldLookAt, WorldUp);
        }
    } */
}
