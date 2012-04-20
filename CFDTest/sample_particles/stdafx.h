#pragma once

#include <iostream>
#include <vector>
#include <sstream>

#include "GL/glew.h"
#include "GL/freeglut.h"
#include "cuda_runtime.h"
#include "cuda_gl_interop.h"

#include <cutil.h>
#include <cutil_math.h>

#include "utils.h"
#include "stopwatch.h"


#define CHECK_OPENGL_ERROR \
{ GLenum error; \
  while ( (error = glGetError()) != GL_NO_ERROR) { \
    printf( "OpenGL ERROR: %s\nCHECK POINT: %s (line %d)\n", \
      gluErrorString(error), __FILE__, __LINE__ ); \
  } \
}