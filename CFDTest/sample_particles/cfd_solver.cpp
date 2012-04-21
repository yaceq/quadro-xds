#include "stdafx.h"
#include "cfd_solver.h"


const uint NUM_PARTICLES = 1024 * 16;


float randf() {
	return (float)rand()/(float)RAND_MAX;
}

float randf_signed() {
	return randf() * 2 - 1;
}


cfd_solver::cfd_solver( float xsz, float ysz, float zsz )
{
	m_particle_num		=	NUM_PARTICLES;

	size_x		= xsz;
	size_y		= ysz;
	size_z		= zsz;

	nx			=	32;
	ny			=	32;
	nz			=	32;

	uint total_grid_size	=	nx * ny * nz;
	uint vbo_size			=	sizeof(float3) * m_particle_num;

	//	create vbo for particles :
	m_position.resize( m_particle_num );
	for (uint i=0; i<m_position.size(); i++) {
		m_position[i].x = randf_signed() * xsz/2;
		m_position[i].y = randf_signed() * ysz/2;
		m_position[i].z = randf_signed() * zsz/2;
	}

    glGenBuffers(1, &m_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glBufferData(GL_ARRAY_BUFFER, vbo_size, &m_position[0], GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    CHECK_OPENGL_ERROR;
    CUDA_SAFE_CALL( cudaGraphicsGLRegisterBuffer(&m_vbo_cuda, m_vbo, cudaGraphicsMapFlagsNone) );

	//	alloc CUDA buffers :
	CUDA_SAFE_CALL( cudaMalloc( &d_position, total_grid_size * sizeof(float3) ) );
	CUDA_SAFE_CALL( cudaMalloc( &d_velocity[0], total_grid_size * sizeof(float3) ) );
	CUDA_SAFE_CALL( cudaMalloc( &d_pressure[0], total_grid_size * sizeof(float) ) );

	CUDA_SAFE_CALL( cudaMalloc( &d_velocity[1], total_grid_size * sizeof(float3) ) );
	CUDA_SAFE_CALL( cudaMalloc( &d_pressure[1], total_grid_size * sizeof(float) ) );


	//	fill buffers with zero :
	CUDA_SAFE_CALL( cudaMemset( d_position, 0, total_grid_size * sizeof(float3) ) );
	CUDA_SAFE_CALL( cudaMemset( d_velocity[0], 0, total_grid_size * sizeof(float3) ) );
	CUDA_SAFE_CALL( cudaMemset( d_pressure[0], 0, total_grid_size * sizeof(float) ) );

	CUDA_SAFE_CALL( cudaMemset( d_velocity[1], 0, total_grid_size * sizeof(float3) ) );
	CUDA_SAFE_CALL( cudaMemset( d_pressure[1], 0, total_grid_size * sizeof(float) ) );

	//	set initial particle positions :
	//CUDA_SAFE_CALL( cudaMemcpy( d_position[0], &m_position[0], total_grid_size * sizeof(float3), cudaMemcpyHostToDevice ) );
}



cfd_solver::~cfd_solver(void)
{
}



void cfd_solver::solve( float dt )
{
	launch_gpu( dt );
}
