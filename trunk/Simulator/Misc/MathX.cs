using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.Xna;
using Microsoft.Xna.Framework;

namespace Misc {
	public static partial class MathX {

		
		/// <summary>
		/// Matrix linear interpolation using Vector3.Lerp and Quaternion.Slerp
		/// </summary>
		/// <param name="t0"></param>
		/// <param name="t1"></param>
		/// <param name="factor"></param>
		/// <returns></returns>
		static public Matrix LerpMatrix ( Matrix t0, Matrix t1, float factor )
		{
			var q0	=	Quaternion.CreateFromRotationMatrix( t0 );
			var q1	=	Quaternion.CreateFromRotationMatrix( t1 );
			var p0	=	t0.Translation;
			var p1	=	t1.Translation;
			var q	=	Quaternion.Slerp( q0, q1, factor );
			var p	=	Vector3.Lerp( p0, p1, factor );
			var	t	=	Matrix.CreateFromQuaternion( q );
			t.Translation = p;
			return t;
		}


		/// <summary>
		/// Squad interpolation from a to b, p and q are control rotations
		/// </summary>
		/// <param name="p"></param>
		/// <param name="a"></param>
		/// <param name="b"></param>
		/// <param name="q"></param>
		/// <param name="t"></param>
		/// <returns></returns>
		static public Quaternion Squad ( Quaternion q0, Quaternion a, Quaternion b, Quaternion q1, float t )
		{
			return Quaternion.Slerp( Quaternion.Slerp(q0,q1,t), Quaternion.Slerp(a,b,t), 2*t*(1-t) );
		}


		/// <summary>
		/// Computes quadrangle points fot q1.
		/// </summary>
		/// <param name="q0"></param>
		/// <param name="q1"></param>
		/// <param name="q2"></param>
		/// <returns></returns>
		static public Quaternion QuadranglePoint ( Quaternion q0, Quaternion q1, Quaternion q2 )
		{
			q0.Normalize();
			q1.Normalize();
			q2.Normalize();
			
			var q1inv = Quaternion.Inverse(q1);

			Quaternion q = q1 * ( (( q1inv * q2 ).Logarithm() + ( q1inv * q0 ).Logarithm()) * -0.25f ).Exponent();
			//q.Normalize();
			return q;
		}



		/// <summary>
		/// Matrix cubic interpolator
		/// </summary>
		public class MatrixInterpolator4 {
			Vector3		p0, p1, p2, p3;
			Quaternion	q0, q1, q2, q3;
			Quaternion	q01a, q01b;
			Quaternion	q12a, q12b;
			Quaternion	q23a, q23b;

			public void Setup ( Matrix m0, Matrix m1, Matrix m2, Matrix m3 )
			{
				Vector3 dummy;
				m0.Decompose( out dummy, out q0, out p0 );
				m1.Decompose( out dummy, out q1, out p1 );
				m2.Decompose( out dummy, out q2, out p2 );
				m3.Decompose( out dummy, out q3, out p3 );

				q01a	=	MathX.QuadranglePoint( q0, q0, q1 );
				q01b	=	MathX.QuadranglePoint( q0, q1, q2 );

				q12a	=	MathX.QuadranglePoint( q0, q1, q2 );
				q12b	=	MathX.QuadranglePoint( q1, q2, q3 );

				q23a	=	MathX.QuadranglePoint( q1, q2, q3 );
				q23b	=	MathX.QuadranglePoint( q2, q3, q3 );
			}

			public Matrix Interpolate ( float factor )
			{
				factor *= 3;
				
				Vector3		p;
				Quaternion	q;

				if (factor<0) {
					p = p0;
					q = q0;
				} else
				if (factor<1) {
					p = Vector3.CatmullRom	( p0, p0, p1, p2, factor );
					q = MathX.Squad			( q0, q01a, q01b, q1, factor );
					q = Quaternion.Slerp	( q0, q1, factor );
				} else 
				if (factor<2) {
					p = Vector3.CatmullRom	( p0, p1, p2, p3, factor-1 );
					q = MathX.Squad			( q1, q12a, q12b, q2, factor-1 );
					q = Quaternion.Slerp	( q1, q2, factor-1 );
				} else 
				if (factor<=3) {
					p = Vector3.CatmullRom	( p1, p2, p3, p3, factor-2 );
					q = MathX.Squad			( q2, q23a, q23b, q3, factor-2 );
					q = Quaternion.Slerp	( q2, q3, factor-2 );
				} else {
					p = p3;
					q = q3;
				}

				Matrix m = Matrix.CreateFromQuaternion( q );
				m.Translation = p;

				return m;
			}
		}


	}
}
