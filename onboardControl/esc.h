class Esc {
  protected: 
  int _pin;

  public:	
  void attach ( int pin ) {
    _pin = pin;
  }
	
  void setThrust ( int thrust ) {
    thrust = thrust & 0xFF;
    analogWrite(_pin, thrust);
  }
};
