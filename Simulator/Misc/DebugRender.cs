using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Audio;
using Microsoft.Xna.Framework.Content;
using Microsoft.Xna.Framework.GamerServices;
using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework.Input;
using Microsoft.Xna.Framework.Media;


namespace Misc {

	public class DebugRender : Microsoft.Xna.Framework.DrawableGameComponent {

		List<VertexPositionColor>	lines	=	new List<VertexPositionColor>();
		BasicEffect	effect;

		public Matrix	ViewMatrix;
		public Matrix	ProjMatrix;
		public bool		ClearZBuffer = false;


		public DebugRender ( Game game ) : base( game )
		{
		}


		public override void Initialize () 
		{
			base.Initialize();
		}


		public override void Update ( GameTime gameTime ) 
		{
			base.Update( gameTime );
		}
	

		protected override void LoadContent () 
		{
			effect	=	new BasicEffect( this.GraphicsDevice );
			effect.VertexColorEnabled = true;
			base.LoadContent();
		}


		protected override void UnloadContent () 
		{
			base.UnloadContent();
		}


		public override void Draw ( GameTime gameTime ) 
		{
			if (ClearZBuffer) {
				Game.GraphicsDevice.Clear( ClearOptions.DepthBuffer, Vector4.Zero, 1, 0 );
			}

			Game.GraphicsDevice.BlendState			= BlendState.NonPremultiplied;
			Game.GraphicsDevice.RasterizerState		= RasterizerState.CullCounterClockwise;
			Game.GraphicsDevice.DepthStencilState	= DepthStencilState.Default;

			foreach (EffectPass pass in effect.CurrentTechnique.Passes ) {
				pass.Apply();
				DrawAllLines();
			}

			base.Draw( gameTime );
		}


		public void Clear () {
			lines.Clear();
		}


		public virtual void SetMatrix ( Matrix view, Matrix projection ) 
		{
			effect.View			=	view;
			effect.Projection	=	projection;
			effect.World		=	Matrix.Identity;

			ViewMatrix			=	view;
			ProjMatrix			=	projection;
		}


		public void DrawLine ( Vector3 p0, Vector3 p1, Color color, Matrix xform )
		{
			lines.Add( new VertexPositionColor( Vector3.Transform(p0, xform), color ) );
			lines.Add( new VertexPositionColor( Vector3.Transform(p1, xform), color ) );
		}


		public void DrawLine ( Vector3 p0, Vector3 p1, Color color )
		{
			DrawLine( p0, p1, color, Matrix.Identity );
		}


		void DrawAllLines ()
		{
			if (lines.Count>=2) {
				Game.GraphicsDevice.DrawUserPrimitives<VertexPositionColor>( PrimitiveType.LineList, lines.ToArray(), 0, lines.Count/2 );
			}
		}


		public void DrawGrid ( int wireCount ) 
		{
			int gridsz = wireCount;
			for (int x=-gridsz; x<=gridsz; x+=1 ) {
				float dim = 0.7f;
				if (x==0) dim = 1.0f;
				DrawLine( new Vector3(x, 0, gridsz), new Vector3(x, 0, -gridsz), Color.DarkGray * dim );
				DrawLine( new Vector3(gridsz, 0, x), new Vector3(-gridsz, 0, x), Color.DarkGray * dim);
			}
		}


		public void DrawBasis ( Matrix basis, float scale ) 
		{
			Vector3	pos		= Vector3.Transform( Vector3.Zero, basis );
			Vector3 xaxis	= Vector3.TransformNormal( Vector3.UnitX * scale, basis );
			Vector3 yaxis	= Vector3.TransformNormal( Vector3.UnitY * scale, basis );
			Vector3 zaxis	= Vector3.TransformNormal( Vector3.UnitZ * scale, basis );
			DrawLine( pos, pos + xaxis, Color.Red );
			DrawLine( pos, pos + yaxis, Color.Lime );
			DrawLine( pos, pos + zaxis, Color.Blue );
		}


		public void DrawVector ( Vector3 origin, Vector3 dir, Color color, float scale=1.0f )
		{
			DrawLine( origin, origin + dir * scale, color, Matrix.Identity );
		}


		public void DrawPoint ( Vector3 p, float size, Color color ) 
		{
			float h = size/2;	// half size
			DrawLine( p + Vector3.UnitX*h, p - Vector3.UnitX*h, color );
			DrawLine( p + Vector3.UnitY*h, p - Vector3.UnitY*h, color );
			DrawLine( p + Vector3.UnitZ*h, p - Vector3.UnitZ*h, color );
		}

		
		public void DrawRing ( Vector3 origin, float radius, Color color )
		{
			const int N = 32;
			Vector3[] points = new Vector3[N+1];

			for (int i=0; i<=N; i++) {
				points[i].X = origin.X + radius * (float)Math.Cos( Math.PI * 2 * i / N );
				points[i].Y = origin.Y;
				points[i].Z = origin.Z + radius * (float)Math.Sin( Math.PI * 2 * i / N );
			}

			for (int i=0; i<N; i++) {
				DrawLine( points[i], points[i+1], color );
			}
		}


		public void DrawRing ( Matrix basis, float radius, Color color, float stretch=1 )
		{
			const int N = 32;
			Vector3[] points = new Vector3[N+1];
			Vector3 origin = basis.Translation;

			for (int i=0; i<=N; i++) {
				points[i] = origin  + radius * basis.Forward * (float)Math.Cos( Math.PI * 2 * i / N ) * stretch
									+ radius * basis.Left    * (float)Math.Sin( Math.PI * 2 * i / N );
			}

			for (int i=0; i<N; i++) {
				DrawLine( points[i], points[i+1], color );
			}
		}


		public void DrawBox ( BoundingBox box, Color color )
		{
			var corners = box.GetCorners();
			
			foreach (var p in corners) {
				DrawPoint( p, 0.1f, color );
			}

			DrawLine( corners[0], corners[1], color );
			DrawLine( corners[1], corners[2], color );
			DrawLine( corners[2], corners[3], color );
			DrawLine( corners[3], corners[0], color );

			DrawLine( corners[4], corners[5], color );
			DrawLine( corners[5], corners[6], color );
			DrawLine( corners[6], corners[7], color );
			DrawLine( corners[7], corners[4], color );

			DrawLine( corners[4], corners[0], color );
			DrawLine( corners[5], corners[1], color );
			DrawLine( corners[6], corners[2], color );
			DrawLine( corners[7], corners[3], color );
		}
	}
}
