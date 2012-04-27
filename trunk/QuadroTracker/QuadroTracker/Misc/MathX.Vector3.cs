using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.Xna;
using Microsoft.Xna.Framework;

namespace Misc {
	public static partial class MathX {
		public static Vector3 Flattern ( this Vector3 self, Vector3 normal )
		{
			return 0.5f * (self + Vector3.Reflect( self, normal ));
		}


		public static Vector3 Normalized ( this Vector3 self )
		{
			Vector3 result = self;
			result.Normalize();
			return result;
		}


		public static Vector3 Expand ( this Vector3 self, float addition )
		{
			float len = self.Length();
			return self.Resize( len + addition );
		}


		public static Vector3 Resize ( this Vector3 self, float newLength )
		{
			Vector3 result = self;
			float selfLength = self.Length();
			result /= selfLength;
			result *= newLength;
			return result;
		}


		public static Vector3 ClampLength ( this Vector3 self, float length )
		{
			Vector3 result = self;
			float selfLength = self.Length();
			if ( selfLength > length ) {
				result /= selfLength;
				result *= length;
			}
			return result;
		}

	}
}
