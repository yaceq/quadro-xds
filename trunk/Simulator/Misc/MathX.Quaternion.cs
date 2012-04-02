using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.Xna;
using Microsoft.Xna.Framework;

namespace Misc {
	public static partial class MathX {
		public static Quaternion Logarithm ( this Quaternion self )
		{
			return self.Log();
			/*self.Normalize();
			float a		= (float)Math.Acos(self.W);
			float sina	= (float)Math.Sin(a);
			Quaternion ret;

			ret.W = 0;
			if (sina > 0) {
				ret.X = a*self.X/sina;
				ret.Y = a*self.Y/sina;
				ret.Z = a*self.Z/sina;
			} else {
				ret.X =	0;
				ret.Y =	0;
				ret.Z =	0;
			}
			return ret;		*/
		}

		public static Quaternion Exponent ( this Quaternion self )
		{
			return self.Exp();

			/*self.Normalize();
			Quaternion q;
			double theta =  self.Length();
			double sin_theta = Math.Sin(theta);

			q.W = (float)Math.Cos(theta);
			if ( theta > float.Epsilon ) {
				q.X = (float) ( self.X * sin_theta / theta );
				q.Y = (float) ( self.Y * sin_theta / theta );
				q.Z = (float) ( self.Z * sin_theta / theta );
			} else {
				q.X = 0;
				q.Y = 0;
				q.Z = 0;
			}

			q = q * (float)Math.Exp( self.W );

			return q;*/
		}

		public static Quaternion Log ( this Quaternion self )
		{
		    var s	 = self.W;
		    var v	 = new Vector3( self.X, self.Y, self.Z );
			var qabs = self.Length();
		    var vabs = v.Length();

			Quaternion q;
			if ( vabs > float.Epsilon ) {
				q.W = (float)Math.Log( qabs );
				q.X = v.X * (float)Math.Acos( s / qabs ) / vabs;
				q.Y = v.Y * (float)Math.Acos( s / qabs ) / vabs;
				q.Z = v.Z * (float)Math.Acos( s / qabs ) / vabs;
			} else {
				q.W = (float)Math.Log( qabs );
				q.X = 0;
				q.Y = 0;
				q.Z = 0;
			}
			return q;
		}

		public static Quaternion Exp ( this Quaternion self )
		{
		    Quaternion q;

		    var s	 = self.W;
		    var v	 = new Vector3( self.X, self.Y, self.Z );
		    var vabs = v.Length();

			if (vabs > float.Epsilon) {
				q.W	= (float)Math.Cos( vabs );
				q.X	= v.X * (float)Math.Sin( vabs ) / vabs;
				q.Y	= v.Y * (float)Math.Sin( vabs ) / vabs;
				q.Z	= v.Z * (float)Math.Sin( vabs ) / vabs;
			} else {  
				q.W	= 1;
				q.X	= 0;
				q.Y	= 0;
				q.Z	= 0;
			}

			q = q * (float)Math.Exp( s );
					  
		    return q;
		}
	}
}
