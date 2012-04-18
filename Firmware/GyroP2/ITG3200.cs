using System;
using System.Threading;
using Microsoft.SPOT;
using Microsoft.SPOT.Hardware;


namespace Firmware {
	public class ITG3200 {

		#region CONSTANTS
		public const byte  ITG3200_ADDR_AD0_HIGH = 0x69;   //AD0=1 0x69 I2C address when AD0 is connected to HIGH ( VCC ) - default for sparkfun breakout
		public const byte  ITG3200_ADDR_AD0_LOW  = 0x68;   //AD0=0 0x68 I2C address when AD0 is connected to LOW ( GND )
		// "The LSB bit of the 7 bit address is determined by the logic level on pin 9. 
		// This allows two ITG-3200 devices to be connected to the same I2C bus.
		// One device should have pin9 ( or bit0 ) LOW and the other should be HIGH." source: ITG3200 datasheet
		// Note that pin9 ( AD0 - I2C Slave Address LSB ) may not be available on some breakout boards so check 
		// the schematics of your breakout board for the correct address to use.

		public const byte  GYROSTART_UP_DELAY = 70;    // 50ms from gyro startup + 20ms register r/w startup

		/* ---- Registers ---- */
		public const byte  WHO_AM_I          = 0x00;  // RW   SETUP: I2C address   
		public const byte  SMPLRT_DIV        = 0x15;  // RW   SETUP: Sample Rate Divider       
		public const byte  DLPF_FS           = 0x16;  // RW   SETUP: Digital Low Pass Filter/ Full Scale range
		public const byte  INT_CFG           = 0x17;  // RW   Interrupt: Configuration
		public const byte  INT_STATUS        = 0x1A;  // R	Interrupt: Status
		public const byte  TEMP_OUT          = 0x1B;  // R	SENSOR: Temperature 2bytes
		public const byte  GYRO_XOUT         = 0x1D;  // R	SENSOR: Gyro X 2bytes  
		public const byte  GYRO_YOUT         = 0x1F;  // R	SENSOR: Gyro Y 2bytes
		public const byte  GYRO_ZOUT         = 0x21;  // R	SENSOR: Gyro Z 2bytes
		public const byte  PWR_MGM           = 0x3E;  // RW	Power Management

		/* ---- bit maps ---- */
		public const byte  DLPFFS_FS_SEL            = 0x18;  // 00011000
		public const byte  DLPFFS_DLPF_CFG          = 0x07;  // 00000111
		public const byte  INTCFG_ACTL              = 0x80;  // 10000000
		public const byte  INTCFG_OPEN              = 0x40;  // 01000000
		public const byte  INTCFG_LATCH_INT_EN      = 0x20;  // 00100000
		public const byte  INTCFG_INT_ANYRD_2CLEAR  = 0x10;  // 00010000
		public const byte  INTCFG_ITG_RDY_EN        = 0x04;  // 00000100
		public const byte  INTCFG_RAW_RDY_EN        = 0x01;  // 00000001
		public const byte  INTSTATUS_ITG_RDY        = 0x04;  // 00000100
		public const byte  INTSTATUS_RAW_DATA_RDY   = 0x01;  // 00000001
		public const byte  PWRMGM_HRESET            = 0x80;  // 10000000
		public const byte  PWRMGM_SLEEP             = 0x40;  // 01000000
		public const byte  PWRMGM_STBY_XG           = 0x20;  // 00100000
		public const byte  PWRMGM_STBY_YG           = 0x10;  // 00010000
		public const byte  PWRMGM_STBY_ZG           = 0x08;  // 00001000
		public const byte  PWRMGM_CLK_SEL           = 0x07;  // 00000111

		/************************************/
		/*    REGISTERS PARAMETERS    */
		/************************************/
		// Sample Rate Divider
		public const byte  NOSRDIVIDER        = 0; // default    FsampleHz=SampleRateHz/( divider+1 )
		// Gyro Full Scale Range
		public const byte  RANGE2000          = 3;   // default
		// Digital Low Pass Filter BandWidth and SampleRate
		public const byte  BW256_SR8          = 0;   // default    256Khz BW and 8Khz SR
		public const byte  BW188_SR1          = 1;
		public const byte  BW098_SR1          = 2;
		public const byte  BW042_SR1          = 3;
		public const byte  BW020_SR1          = 4;
		public const byte  BW010_SR1          = 5;
		public const byte  BW005_SR1          = 6;
		// Interrupt Active logic lvl
		public const byte  ACTIVE_ONHIGH      = 0; // default
		public const byte  ACTIVE_ONLOW       = 1;
		// Interrupt drive type
		public const byte  PUSH_PULL          = 0; // default
		public const byte  OPEN_DRAIN         = 1;
		// Interrupt Latch mode
		public const byte  PULSE_50US         = 0; // default
		public const byte  UNTIL_INT_CLEARED  = 1;
		// Interrupt Latch clear method
		public const byte  READ_STATUSREG     = 0; // default
		public const byte  READ_ANYREG        = 1;
		// Power management
		public const byte  NORMAL             = 0; // default
		public const byte  STANDBY            = 1;
		// Clock Source - user parameters
		public const byte  INTERNALOSC        = 0;   // default
		public const byte  PLL_XGYRO_REF      = 1;
		public const byte  PLL_YGYRO_REF      = 2;
		public const byte  PLL_ZGYRO_REF      = 3;
		public const byte  PLL_EXTERNAL32     = 4;   // 32.768 kHz
		public const byte  PLL_EXTERNAL19     = 5;   // 19.2 Mhz
	#endregion
	

		byte	_dev_address;
		float[]	scalefactor	= new float[3];    // Scale Factor for gain and polarity
		int[]	offsets		= new int[3];
		byte[]	_buff		= new byte[6];  
		
		I2CDevice.Configuration	config;  
		I2CBus					bus;  

		public ITG3200() 
		{
			config	=	new I2CDevice.Configuration( ITG3200_ADDR_AD0_LOW, 400 );
			bus		=	I2CBus.GetInstance();

			setOffsets( 0,0,0 );
			setScaleFactor( 1.0f, 1.0f, 1.0f, false );  // true to change readGyro output to radians
			//Wire.begin();     //Normally this code is called from setup() at user code
								//but some people reported that joining I2C bus earlier
								//apparently solved problems with master/slave conditions.
								//Uncomment if needed.
		}

		public void init( byte address ) {
		  // Uncomment or change your default ITG3200 initialization
  
		  // fast sample rate - divisor = 0 filter = 0 clocksrc = 0, 1, 2, or 3  ( raw values )
		  init( address, NOSRDIVIDER, RANGE2000, BW256_SR8, PLL_XGYRO_REF, true, true );
  
		  // slow sample rate - divisor = 0  filter = 1,2,3,4,5, or 6  clocksrc = 0, 1, 2, or 3  ( raw values )
		  //init( NOSRDIVIDER, RANGE2000, BW010_SR1, INTERNALOSC, true, true );
  
		  // fast sample rate 32Khz external clock - divisor = 0  filter = 0  clocksrc = 4  ( raw values )
		  //init( NOSRDIVIDER, RANGE2000, BW256_SR8, PLL_EXTERNAL32, true, true );
  
		  // slow sample rate 32Khz external clock - divisor = 0  filter = 1,2,3,4,5, or 6  clocksrc = 4  ( raw values )
		  //init( address, NOSRDIVIDER, RANGE2000, BW010_SR1, PLL_EXTERNAL32, true, true );
		}

		public void init( byte address, byte _SRateDiv, byte _Range, byte _filterBW, byte _ClockSrc, bool _ITGReady, bool _INTRawDataReady ) 
		{
			_dev_address = address;

			setSampleRateDiv	( _SRateDiv );
			setFSRange			( _Range );
			setFilterBW			( _filterBW );
			setClockSource		( _ClockSrc );
			setITGReady			( _ITGReady );
			setRawDataReady		( _INTRawDataReady );  
			Thread.Sleep		( GYROSTART_UP_DELAY );  // startup 
		}


		byte tempBuf;

		public byte getDevAddr() {
			//readmem( WHO_AM_I, out tempBuf ); 
			//return _buff[0];  
			return _dev_address;
		}

		public void setDevAddr( byte  _addr ) {
			writemem( WHO_AM_I, _addr ); 
			_dev_address = _addr;
		}

		public byte getSampleRateDiv() {
			return readmem( SMPLRT_DIV );
		}

		public void setSampleRateDiv( byte _SampleRate ) {
		  writemem( SMPLRT_DIV, _SampleRate );
		}

		public byte getFSRange() {
		  readmem( DLPF_FS, out tempBuf );
		  return (byte)( ( tempBuf & DLPFFS_FS_SEL ) >> 3 );
		}

		public void setFSRange( byte _Range ) {
		  readmem( DLPF_FS, out tempBuf );   
		  writemem( DLPF_FS, (byte)( ( tempBuf & ~DLPFFS_FS_SEL ) | ( _Range << 3 ) )  ); 
		}

		public byte getFilterBW() {  
		  readmem( DLPF_FS, out tempBuf );
		  return (byte)( tempBuf & DLPFFS_DLPF_CFG ); 
		}

		public void setFilterBW( byte _BW ) {   
		  readmem( DLPF_FS, out tempBuf );
		  writemem( DLPF_FS, (byte)( ( tempBuf & ~DLPFFS_DLPF_CFG ) | _BW ) ); 
		}

		public bool isINTActiveOnLow() {  
		  readmem( INT_CFG, out tempBuf );
		  return ( ( tempBuf & INTCFG_ACTL ) >> 7 )==1;
		}

		public void setINTLogiclvl( bool _State ) {
		  readmem( INT_CFG, out tempBuf );
		  writemem( INT_CFG, (byte)( ( tempBuf & ~INTCFG_ACTL ) | ( _State ? 1:0 << 7 ) ) ); 
		}

		public bool isINTOpenDrain() {  
		  readmem( INT_CFG, out tempBuf );
		  return ( ( tempBuf & INTCFG_OPEN ) >> 6 )==1;
		}

		public void setINTDriveType( bool _State ) {
		  readmem( INT_CFG, out tempBuf );
		  writemem( INT_CFG, (byte)( ( tempBuf & ~INTCFG_OPEN ) | (_State? 1:0) << 6 ) ); 
		}

		public bool isLatchUntilCleared() {    
		  readmem( INT_CFG, out tempBuf );
		  return ( ( tempBuf & INTCFG_LATCH_INT_EN ) >> 5 )==1;
		}

		public void setLatchMode( bool _State ) {
		  readmem( INT_CFG, out tempBuf );
		  writemem( INT_CFG, (byte)( ( tempBuf & ~INTCFG_LATCH_INT_EN ) | (_State? 1:0) << 5 ) ); 
		}

		public bool isAnyRegClrMode() {    
		  readmem( INT_CFG, out tempBuf );
		  return ( ( tempBuf & INTCFG_INT_ANYRD_2CLEAR ) >> 4 )==1;
		}

		public void setLatchClearMode( bool _State ) {
		  readmem( INT_CFG, out tempBuf );
		  writemem( INT_CFG, (byte)( ( tempBuf & ~INTCFG_INT_ANYRD_2CLEAR ) | (_State? 1:0) << 4 ) ); 
		}

		public bool isITGReadyOn() {   
		  readmem( INT_CFG, out tempBuf );
		  return ( ( tempBuf & INTCFG_ITG_RDY_EN ) >> 2 )==1;
		}

		public void setITGReady( bool _State ) {
		  readmem( INT_CFG, out tempBuf );
		  writemem( INT_CFG, (byte)( ( tempBuf & ~INTCFG_ITG_RDY_EN ) | (_State? 1:0) << 2 ) ); 
		}

		public bool isRawDataReadyOn() {
		  readmem( INT_CFG, out tempBuf );
		  return ( tempBuf & INTCFG_RAW_RDY_EN )==1;
		}

		public void setRawDataReady( bool _State ) {
		  readmem( INT_CFG, out tempBuf );
		  writemem( INT_CFG, (byte)( ( tempBuf & ~INTCFG_RAW_RDY_EN ) | (_State? 1:0) ) ); 
		}

		public bool isITGReady() {
		  readmem( INT_STATUS, out tempBuf );
		  return ( ( tempBuf & INTSTATUS_ITG_RDY ) >> 2 )==1;
		}

		public bool isRawDataReady() {
		  readmem( INT_STATUS, out tempBuf );
		  return ( tempBuf & INTSTATUS_RAW_DATA_RDY )==1;
		}

		public float readTemp() {
			var b1 = readmem( TEMP_OUT );
			var b2 = readmem( TEMP_OUT+1 );
			return 35 + ( ( b1 << 8 | b2 ) + 13200 ) / 280.0f;    // F=C*9/5+32
		}

		public void readGyroRaw( out int _GyroX, out int _GyroY, out int _GyroZ ) 
		{
			var _buff = new byte[6];
			bus.ReadRegister( config, GYRO_XOUT, _buff, 1000 );
			_GyroX = _buff[0] << 8 | _buff[1];
			_GyroY = _buff[2] << 8 | _buff[3]; 
			_GyroZ = _buff[4] << 8 | _buff[5];
		}

		/*public void readGyroRaw(  int *_GyroXYZ ){
		  readGyroRaw( _GyroXYZ, _GyroXYZ+1, _GyroXYZ+2 );
		}*/

		public void setScaleFactor( float _Xcoeff, float _Ycoeff, float _Zcoeff, bool _Radians ) 
		{ 
			scalefactor[0] = 14.375f * _Xcoeff;   
			scalefactor[1] = 14.375f * _Ycoeff;
			scalefactor[2] = 14.375f * _Zcoeff;    
    
			if ( _Radians ) {
				scalefactor[0] /= 0.0174532925f;//0.0174532925 = PI/180
				scalefactor[1] /= 0.0174532925f;
				scalefactor[2] /= 0.0174532925f;
			}
		}

		public void setOffsets( int _Xoffset, int _Yoffset, int _Zoffset ) 
		{
			offsets[0] = _Xoffset;
			offsets[1] = _Yoffset;
			offsets[2] = _Zoffset;
		}

		public void zeroCalibrate( uint totSamples, int sampleDelayMS ) {
		  float tmpX=0, tmpY=0, tmpZ=0;
		  int x, y, z;

		  for ( int i = 0;i < totSamples;i++ ){
			Thread.Sleep( sampleDelayMS );
			readGyroRaw( out x, out y, out z );
			tmpX += x;
			tmpY += y;
			tmpZ += z;
		  }
		  setOffsets( (int)(-tmpX / totSamples + 0.5f), (int)(-tmpY / totSamples + 0.5f), (int)(-tmpZ / totSamples + 0.5f) );
		}

		public void readGyroRawCal( out int _GyroX, out int _GyroY, out int _GyroZ ) { 
			readGyroRaw( out _GyroX, out _GyroY, out _GyroZ ); 
			_GyroX += offsets[0]; 
			_GyroY += offsets[1]; 
			_GyroZ += offsets[2]; 
		} 

		//public void readGyroRawCal( int *_GyroXYZ ) { 
		//  readGyroRawCal( _GyroXYZ, _GyroXYZ+1, _GyroXYZ+2 ); 
		//} 

		public void readGyro( out float _GyroX, out float _GyroY, out float _GyroZ ){
		  int x, y, z; 
		  readGyroRawCal( out x, out y, out z ); // x,y,z will contain calibrated integer values from the sensor 
		  _GyroX =  x / scalefactor[0]; 
		  _GyroY =  y / scalefactor[1]; 
		  _GyroZ =  z / scalefactor[2];     
		} 

		//public void readGyro( float *_GyroXYZ ){
		//  readGyro( _GyroXYZ, _GyroXYZ+1, _GyroXYZ+2 );
		//}

		public void reset() {     
		  writemem( PWR_MGM, PWRMGM_HRESET ); 
		  Thread.Sleep( GYROSTART_UP_DELAY ); //gyro startup 
		}

		public bool isLowPower() {   
		  readmem( PWR_MGM, out tempBuf );
		  return (( tempBuf & PWRMGM_SLEEP ) >> 6 ) == 1;
		}
  
		public void setPowerMode( bool _State ) {
		  readmem( PWR_MGM, out tempBuf );
		  writemem( PWR_MGM, (byte)( ( tempBuf & ~PWRMGM_SLEEP ) | (_State? 1:0) << 6 ) );  
		}

		public bool isXgyroStandby() {
		  readmem( PWR_MGM, out tempBuf );
		  return (( tempBuf & PWRMGM_STBY_XG ) >> 5 ) == 1;
		}

		public bool isYgyroStandby() {
		  readmem( PWR_MGM, out tempBuf );
		  return (( tempBuf & PWRMGM_STBY_YG ) >> 4 ) == 1;
		}

		public bool isZgyroStandby() {
		  readmem( PWR_MGM, out tempBuf );
		  return (( tempBuf & PWRMGM_STBY_ZG ) >> 3 ) == 1;
		}

		public void setXgyroStandby( bool _Status ) {
		  readmem( PWR_MGM, out tempBuf );
		  writemem( PWR_MGM, (byte)( ( tempBuf & PWRMGM_STBY_XG ) | (_Status? 1:0) << 5 ) );
		}

		public void setYgyroStandby( bool _Status ) {
		  readmem( PWR_MGM, out tempBuf );
		  writemem( PWR_MGM, (byte)( ( tempBuf & PWRMGM_STBY_YG ) | (_Status? 1:0) << 4 ) );
		}

		public void setZgyroStandby( bool _Status ) {
		  readmem( PWR_MGM, out tempBuf );
		  writemem( PWR_MGM, (byte)( ( tempBuf & PWRMGM_STBY_ZG ) | (_Status? 1:0) << 3 ) );
		}

		public byte getClockSource() {  
		  readmem( PWR_MGM, out tempBuf );
		  return (byte)( tempBuf & PWRMGM_CLK_SEL );
		}

		public void setClockSource( byte _CLKsource ) {   
		  readmem( PWR_MGM, out tempBuf );
		  writemem( PWR_MGM, (byte)( ( tempBuf & ~PWRMGM_CLK_SEL ) | _CLKsource ) ); 
		}

		public void writemem( byte _addr, byte _val ) {
			bus.WriteRegister( config, _addr, _val, 1000 );
		}

		public byte readmem( byte _addr ) {
			return bus.ReadRegister( config, _addr, 1000 );
		}

		public void readmem( byte _addr, out byte val ) {
			val = bus.ReadRegister( config, _addr, 1000 );
		}



	}
}
