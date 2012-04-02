using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.Xna;
using Microsoft.Xna.Framework;

namespace Misc {
	public static partial class MathX {
		/// <summary>
		/// Decomposes matrix to yaw, pitch, roll angles. Need check!!!
		/// </summary>
		/// <param name="matrix"></param>
		/// <param name="yaw"></param>
		/// <param name="pitch"></param>
		/// <param name="roll"></param>
		public static void ToAngles ( this Matrix mat, out float yaw, out float pitch, out float roll )
		{
			//	     |  0  1  2  3 |	| 11 12 13 14 |
			//	M =  |  4  5  6  7 |	| 21 22 23 24 |
			//	     |  8  9 10 11 |	| 31 32 33 34 |
			//	     | 12 13 14 15 |	| 41 42 43 44 |

			float sy		=	MathHelper.Clamp( mat.M13, -1, 1 );
			float angle_x	=	0;
			float angle_z	=	0;
			float angle_y	=	-(float)Math.Asin( sy );
			float cy        =	 (float)Math.Cos( angle_y );
			float _trx, _try;

			if ( Math.Abs( cy ) > 8192.0f * float.Epsilon ) {
				_trx	=	 mat.M33 / cy;          
				_try	=	-mat.M23 / cy;

				angle_x	=	(float)Math.Atan2( _try, _trx );

				_trx	=	 mat.M11 / cy;          
				_try	=	-mat.M12 / cy;

				angle_z	=	(float)Math.Atan2( _try, _trx );

			} else {
				angle_x	=	0;                     

				_trx	=	mat.M22;                
				_try	=	mat.M21;

				angle_z	=	(float)Math.Atan2( _try, _trx );
			}

			pitch	= angle_x;
			yaw		= angle_y;
			roll	= angle_z;			
		}	

			//double	theta;
			//double	cy;
			//float	sy;
			//Matrix	data = matrix;

			//sy = data.M13;

			//// cap off our sin value so that we don't get any NANs
			//sy = MathHelper.Clamp( sy, -1, 1 );

			//theta = -Math.Asin( sy );
			//cy = Math.Cos( theta );

			//if ( cy > 8192.0f * float.Epsilon ) {
			//    pitch   =   (float)Math.Atan2( data.M12, data.M11 );
			//    yaw     =   (float)theta;

			//    roll    =   (float)Math.Atan2( data.M23, data.M33 );
			//} else {
			//    pitch   =   (float)theta;
			//    yaw     = - (float)Math.Atan2( data.M21, data.M22 );
			//    roll    =   0;
			//}
	}
}
