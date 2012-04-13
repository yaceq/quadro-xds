using System;
using Microsoft.SPOT;

namespace Firmware {
	public class ExpanderPlug : I2CPlug
	{
		private const int ExpanderPlugAddress = 0x20;
 
		public enum Registers
		{
			IODIR,
			IPOL,
			GPINTEN,
			DEFVAL,
			INTCON,
			IOCON,
			GPPU,
			INTF,
			INTCAP,
			GPIO,
			OLAT
		};
 
		public ExpanderPlug()
			: base(ExpanderPlugAddress)
		{
		}
		public ExpanderPlug(byte directions)
			: base(ExpanderPlugAddress)
		{
			SetDirections(directions);
		}
 
		public void SetDirections(byte directions)
		{
			this.WriteToRegister((byte)Registers.IODIR, directions);
		}
 
		public void Write(byte values)
		{
			this.WriteToRegister((byte)Registers.GPIO, values);
		}
 
		public byte Read()
		{
			byte[] values = new byte[1] { 0};
 
			this.ReadFromRegister((byte)Registers.GPIO, values);
 
			return values[0];
		}
	}
}
