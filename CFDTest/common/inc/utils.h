#pragma once

class noncopyable
{
protected:
  noncopyable() {}
  ~noncopyable() {}
private:  // emphasize the following members are private
  noncopyable( const noncopyable& );
  const noncopyable& operator=( const noncopyable& );
};

inline int iDivUp(int a, int b){
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

