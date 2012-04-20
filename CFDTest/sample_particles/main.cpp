#include "stdafx.h"
#include "particles.h"


/*-----------------------------------------------------------------------------
	main
-----------------------------------------------------------------------------*/

using namespace std;

const int		PARTICLE_N = 10*1024;
ParticleSystem *g_system = NULL;
int		g_buttonState = 0;
int		g_mouse_x = 0, g_mouse_y = 0;
float	g_view_phi = 10, g_view_theta = 10;
float	g_view_dist = 150.0f;
int		g_w, g_h;
bool	g_stereo;

void shutdown();


void init()
{
    g_system = new ParticleSystem(PARTICLE_N);
	atexit( shutdown );
}


void shutdown()
{
	delete g_system;
	g_system  = NULL;
}



void drawbox(float xsz, float ysz, float zsz)
{
	glBegin(GL_LINES);

		glColor4f (1,0,0,1);
		glVertex3f(0,0,0);
		glVertex3f(1,0,0);

		glColor4f (0,1,0,1);
		glVertex3f(0,0,0);
		glVertex3f(0,1,0);

		glColor4f (0,0,1,1);
		glVertex3f(0,0,0);
		glVertex3f(0,0,1);

		glColor4f (1,1,1,1);
		
		glVertex3f(  xsz/2,  ysz/2, -zsz/2 );	 	glVertex3f( -xsz/2,  ysz/2, -zsz/2 );
		glVertex3f( -xsz/2,  ysz/2, -zsz/2 );	 	glVertex3f( -xsz/2, -ysz/2, -zsz/2 );
		glVertex3f( -xsz/2, -ysz/2, -zsz/2 );	 	glVertex3f(  xsz/2, -ysz/2, -zsz/2 );
		glVertex3f(  xsz/2, -ysz/2, -zsz/2 );	 	glVertex3f(  xsz/2,  ysz/2, -zsz/2 );
												 	
		glVertex3f(  xsz/2,  ysz/2,  zsz/2 );	 	glVertex3f( -xsz/2,  ysz/2,  zsz/2 );
		glVertex3f( -xsz/2,  ysz/2,  zsz/2 );	 	glVertex3f( -xsz/2, -ysz/2,  zsz/2 );
		glVertex3f( -xsz/2, -ysz/2,  zsz/2 );	 	glVertex3f(  xsz/2, -ysz/2,  zsz/2 );
		glVertex3f(  xsz/2, -ysz/2,  zsz/2 );	 	glVertex3f(  xsz/2,  ysz/2,  zsz/2 );
												 	
		glVertex3f(  xsz/2,  ysz/2, -zsz/2 );	 	glVertex3f(  xsz/2,  ysz/2,  zsz/2 );
		glVertex3f( -xsz/2,  ysz/2, -zsz/2 );	 	glVertex3f( -xsz/2,  ysz/2,  zsz/2 );
		glVertex3f( -xsz/2, -ysz/2, -zsz/2 );	 	glVertex3f( -xsz/2, -ysz/2,  zsz/2 );
		glVertex3f(  xsz/2, -ysz/2, -zsz/2 );	 	glVertex3f(  xsz/2, -ysz/2,  zsz/2 );
												 	
	glEnd();
}


void display()
{
    g_system->Update(0.005f, g_view_dist);

    float calc_time = g_system->getLastIterTime();

	if (!g_stereo) {
	    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		float aspect = (float) g_h / (float) g_w;
		//gluPerspective(60.0, (float) g_w / (float) g_h, 0.1, 10000.0);
		glFrustum(-0.1, 0.1, -0.1*aspect, 0.1*aspect, 0.1, 10000.0f );
		glMatrixMode(GL_MODELVIEW);

		glEnable(GL_BLEND);
		glBlendFunc(GL_ONE, GL_ONE);
		//glPointSize(3);
		//glEnable(GL_POINT_SMOOTH);

		glLoadIdentity();
		glTranslatef(0, 0, -g_view_dist);
		glRotatef(g_view_theta, 1, 0, 0);
		glRotatef(g_view_phi, 0, 1, 0);


		glBindBuffer(GL_ARRAY_BUFFER, g_system->GetParticleVBO());

			glEnableClientState(GL_VERTEX_ARRAY);
			glVertexPointer(3, GL_FLOAT, sizeof(Vertex), NULL);

			glEnableClientState(GL_COLOR_ARRAY);
			glColorPointer(4, GL_FLOAT, sizeof(Vertex), (void*)(offsetof(Vertex, color)));

			glDrawArrays(GL_POINTS, 0,  g_system->GetParticleNum());

		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glDisableClientState(GL_VERTEX_ARRAY);
		glDisableClientState(GL_COLOR_ARRAY);

		drawbox( 32,32,32 );


        // print stats
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        glRasterPos2f(-0.9, 0.9);

        std::ostringstream ss;
        ss << "calc time: " << calc_time << "ms\nMode: " << mode_names[g_system->getRunMode()];
        glutBitmapString(GLUT_BITMAP_9_BY_15, (const unsigned char*)ss.str().c_str());

		glutSwapBuffers();
	} 
}



void mouse(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN) {
        g_buttonState |= 1<<button;
	} else if (state == GLUT_UP) {
        g_buttonState = 0;
	}
    
    g_mouse_x = x;
    g_mouse_y = y;
    glutPostRedisplay();
}



void motion(int x, int y)
{
    int dx = x - g_mouse_x;
    int dy = y - g_mouse_y;
    
    if (g_buttonState & 1)
    {
        g_view_phi += 0.5f * dx;
        g_view_theta += 0.5f * dy;
        g_view_theta = clamp(g_view_theta, -90.0, 90.0);
    }
    if (g_buttonState & 4)
    {
		g_view_dist *= pow(1.007f, -dy);
    }

    g_mouse_x = x;
    g_mouse_y = y;
    glutPostRedisplay();
}


void keyboard(unsigned char key, int x, int y) 
{
	if (key==VK_ESCAPE) {
		exit(0);
	}
    if (key == 'm')
    {
        int mode = g_system->getRunMode();
        g_system->setRunMode((mode + 1) % MODE_NUM);

    }
}


void reshape(int w, int h)
{
	g_w = w;
	g_h = h;
    glViewport(0, 0, w, h);
}



void idle(void)
{
    glutPostRedisplay();
}



int main(int argc, char **argv)
{
    glutInit(&argc, argv);
	if (argc>=2 && (strcmp(argv[1], "-stereo")==0) ) {
		g_stereo = true;
	}
	glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE | (g_stereo ? GLUT_STEREO : 0));
    //glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE );
    glutInitWindowSize(800, 600);

    glutCreateWindow("CUDA n-body system");

    CUDA_SAFE_CALL( cudaGLSetGLDevice( 0 ) );

    GLenum err = glewInit();
    if (GLEW_OK != err)
    {
      fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
      return 1;
    }

    init();

    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutMotionFunc(motion);
    glutMouseFunc(mouse);
	glutKeyboardFunc(keyboard);
    glutIdleFunc(idle);

    glutMainLoop();

    return 0;
}