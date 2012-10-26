using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Diagnostics;
using Microsoft.Xna;
using Microsoft.Xna.Framework;

namespace Misc {
	public static partial class MathX {
		/// <summary>
		/// Decomposes matrix to yaw, pitch, roll angles. Need check: ROLL DIRECTION!!!
		/// </summary>
		/// <param name="matrix"></param>
		/// <param name="yaw"></param>
		/// <param name="pitch"></param>
		/// <param name="roll"></param>
		public static void ToAngles ( this Matrix mat, out float yaw, out float pitch, out float roll )
		{
			pitch		=	0;
			roll		=	0;
			yaw			=	(float)Math.Atan2( mat.Backward.X, mat.Backward.Z );	
			
			pitch		=	(float)Math.Asin( mat.Forward.Y );

			var	xp		=	Vector3.Cross( mat.Forward, Vector3.UnitY );
			var	xp2		=	Vector3.Cross( mat.Forward, Vector3.UnitY );
			xp.Normalize();

			var dotY	=	MathHelper.Clamp( Vector3.Dot( mat.Up, xp ), -1, 1 );
			var dotX	=	MathHelper.Clamp( Vector3.Dot( mat.Right, xp ), -1, 1 );

			roll		=	- (float)Math.Atan2( dotY, dotX );

			/*Debug.Assert( !float.IsNaN(yaw) );
			Debug.Assert( !float.IsNaN(pitch) );
			Debug.Assert( !float.IsNaN(roll) );*/

			return;
		}	
	}
}
