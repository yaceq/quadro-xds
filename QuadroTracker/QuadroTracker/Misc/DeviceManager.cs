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
	public static class DeviceManagerExt {

		public static void ResetDeviceState ( this GraphicsDevice device ) {
			device.BlendState			= BlendState.Opaque;
			device.RasterizerState		= RasterizerState.CullCounterClockwise;
			device.DepthStencilState	= DepthStencilState.Default;

			//SamplerState	samplerState = new SamplerState();
			//samplerState.MaxAnisotropy	=	64;
			//samplerState.AddressU		=	TextureAddressMode.Wrap;
			//samplerState.AddressV		=	TextureAddressMode.Wrap;
			//samplerState.Filter			=	TextureFilter.Anisotropic;
			
			for (int i=0; i<8; i++) {
				device.SamplerStates[i] = SamplerState.AnisotropicWrap;
			}
		}
	}
}
