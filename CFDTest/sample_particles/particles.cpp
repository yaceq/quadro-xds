#include "stdafx.h"
#include "particles.h"


/*-----------------------------------------------------------------------------
	Particle system :
-----------------------------------------------------------------------------*/

inline float GetRandomFloat()
{
	return ((float)rand()) / RAND_MAX;
}


inline float3 GetRandomFloat3()
{
	float3 p;
	p.x = 2*((float)rand()) / RAND_MAX - 1.0f;
	p.y = 2*((float)rand()) / RAND_MAX - 1.0f;
	p.z = 2*((float)rand()) / RAND_MAX - 1.0f;
	return p;
}


float3 GetRandomDirection()
{
	float3 p;
	float pn;
	do {
		p = GetRandomFloat3();
		pn = length(p);
	} while (pn>1);

	if (abs(pn)<0.00001f) return make_float3(1,0,0);
	return p/pn;
}



//
//	ParticleSystem::ParticleSystem
//
ParticleSystem::ParticleSystem(int N) 
: m_particle_num(N)
, m_run_mode(MODE_GPU_NAIVE)
, m_pos(N), m_vel(N), m_force(N), m_verts(N)
, m_host_valid(true)
{
    for (int i = 0; i < N; ++i) {
		float3 p = GetRandomDirection();
        m_pos[i] = p*10;
		m_vel[i] = p*25 + GetRandomDirection();
    }

	//for (int i = 0; i < N; ++i) {
	//	float a = GetRandomFloat() * 6.28;
	//	float x = 10*cos(a)*GetRandomFloat();
	//	float y = 1*GetRandomFloat() * GetRandomFloat();
	//	float z = 10*sin(a)*GetRandomFloat();
	//	float vx = 12*cos(a+3.14/2);
	//	float vy = 0;
	//	float vz = 12*sin(a+3.14/2);

	//	vel[i] = make_float3(vx,vy,vz);
	//	pos[i] = make_float3(x,y,z);
	//}

	int buf_size = sizeof(float3) * N;
	int vbo_size = sizeof(Vertex) * N;

    glGenBuffers(1, &m_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glBufferData(GL_ARRAY_BUFFER, vbo_size, NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    CHECK_OPENGL_ERROR;
    CUDA_SAFE_CALL( cudaGraphicsGLRegisterBuffer(&m_vbo_cuda, m_vbo, cudaGraphicsMapFlagsNone) );

	CUDA_SAFE_CALL( cudaMalloc(&d_pos, buf_size) );
	CUDA_SAFE_CALL( cudaMalloc(&d_vel, buf_size) );
	CUDA_SAFE_CALL( cudaMalloc(&d_force, buf_size) );
	CUDA_SAFE_CALL( cudaMemcpy( d_pos, &m_pos[0], buf_size, cudaMemcpyHostToDevice ) );
	CUDA_SAFE_CALL( cudaMemcpy( d_vel, &m_vel[0], buf_size, cudaMemcpyHostToDevice ) );
	CUDA_SAFE_CALL( cudaMemset(d_force, 0, buf_size) );
}


//
//	ParticleSystem::~ParticleSystem
//
ParticleSystem::~ParticleSystem()
{
	CUDA_SAFE_CALL( cudaGraphicsUnregisterResource(m_vbo_cuda) );
    glDeleteBuffers(1, &m_vbo);
	CUDA_SAFE_CALL( cudaFree(d_vel) );
	CUDA_SAFE_CALL( cudaFree(d_force) );
}


//
//	ParticleSystem::Update
//
void ParticleSystem::Update(float dt, float view_dist)
{
    if (m_run_mode == MODE_CPU)
    {
        m_cuda_timer.reset();
        m_cuda_timer.start();
        LaunchCPU(dt, view_dist);
        m_cuda_timer.stop();
    }
    else
    {
        m_cuda_timer.reset();
        m_cuda_timer.start();
        LaunchGPU( dt, view_dist );			  
        m_cuda_timer.stop();
    }
}


__forceinline float RSqrt( float x ) {

 long i;
 float y, r;

 y = x * 0.5f;
 i = *reinterpret_cast<long *>( &x );
 i = 0x5f3759df - ( i >> 1 );
 r = *reinterpret_cast<float *>( &i );
 r = r * ( 1.5f - r * r * y );
 return r;
}
//
//	ParticleSystem::LaunchCPU
//
void ParticleSystem::LaunchCPU( float dt, float view_dist )
{
    float3 * p_pos = &m_pos[0];
    float3 * p_vel = &m_vel[0];
    float3 * p_force = &m_force[0];
    

    if (!m_host_valid)
    {
        int buf_size = sizeof(float3) * m_particle_num;
    	CUDA_SAFE_CALL( cudaMemcpy( &m_pos[0], d_pos, buf_size, cudaMemcpyDeviceToHost ) );
	    CUDA_SAFE_CALL( cudaMemcpy( &m_vel[0], d_vel, buf_size, cudaMemcpyDeviceToHost ) );
        m_host_valid = true;
    }

    const float SofteningSqr = 0.01f;

#pragma omp parallel for
    for (int i = 0; i < m_particle_num; ++i)
    {
        const float3 cur_pos = p_pos[i];
        float3 force = make_float3(0);
        for (int j = 0; j < m_particle_num; ++j)
        {
            float3 r = cur_pos - p_pos[j];
            float invDist = rsqrtf( dot(r, r) + SofteningSqr );
            force += r * (invDist * invDist * invDist);
            
            //float invDist = RSqrt( dot(r, r) + SofteningSqr );
            //force += calcForce(cur_pos, p_pos[j]);
        }
        p_force[i] = force;
    }
    for (int i = 0; i < m_particle_num; ++i)
    {
	    float3 v = p_vel[i] + p_force[i] * dt;
	    float3 p = p_pos[i] + v*dt;
	    p_pos[i] = p;
	    p_vel[i] = v;
    }
    for (int i = 0; i < m_particle_num; ++i)
    {
        m_verts[i] = calcVisualProps(p_pos[i], view_dist);
    }
    
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(Vertex) * m_verts.size(), &m_verts[0], GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}


//
//	ParticleSystem::getLastIterTime
//
float ParticleSystem::getLastIterTime() const
{
    return m_cuda_timer.getTime();
}

