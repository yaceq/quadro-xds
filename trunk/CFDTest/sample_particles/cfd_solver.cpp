#include "stdafx.h"
#include "cfd_solver.h"


const uint NUM_PARTICLES = 1024 * 16 * 16;


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
	uint vbo_size			=	sizeof(FVertex) * m_particle_num;

	//	create vbo for particles :
	m_position.resize( m_particle_num );
	for (uint i=0; i<m_position.size(); i++) {
		m_position[i].pos.x = randf_signed() * xsz/2;
		m_position[i].pos.y = randf_signed() * ysz/2;
		m_position[i].pos.z = randf_signed() * zsz/2;
		m_position[i].pos.w = 1;
		m_position[i].color.x = 1;
		m_position[i].color.y = 1;
		m_position[i].color.z = 1;
		m_position[i].color.w = 1;
	}

    glGenBuffers(1, &m_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glBufferData(GL_ARRAY_BUFFER, vbo_size, &m_position[0], GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    CHECK_OPENGL_ERROR;
    CUDA_SAFE_CALL( cudaGraphicsGLRegisterBuffer(&m_vbo_cuda, m_vbo, cudaGraphicsMapFlagsNone) );

	//	alloc CUDA buffers :
	volume_extent		=	make_cudaExtent( nx, ny, nz );
	volume_chan_desc	=	cudaCreateChannelDesc<float>();
	volume_size			=	sizeof(float) * nx * ny * nz;

	std::vector<float> zero_array;
	zero_array.resize( nx*ny*nz, 0 );
	//for (int i=0; i<zero_array.size(); i++) { zero_array[i] = randf_signed(); }
	
	CUDA_SAFE_CALL( cudaMalloc3DArray( &d_divergence_array, &volume_chan_desc, volume_extent ) );
	CUDA_SAFE_CALL( cudaMalloc3DArray( &d_velocity_x_array, &volume_chan_desc, volume_extent ) );
	CUDA_SAFE_CALL( cudaMalloc3DArray( &d_velocity_y_array, &volume_chan_desc, volume_extent ) );
	CUDA_SAFE_CALL( cudaMalloc3DArray( &d_velocity_z_array, &volume_chan_desc, volume_extent ) );
	CUDA_SAFE_CALL( cudaMalloc3DArray( &d_pressure_array,   &volume_chan_desc, volume_extent ) );

	CUDA_SAFE_CALL( cudaMalloc( &d_divergence, volume_size ) );
	CUDA_SAFE_CALL( cudaMalloc( &d_velocity_x, volume_size ) );
	CUDA_SAFE_CALL( cudaMalloc( &d_velocity_y, volume_size ) );
	CUDA_SAFE_CALL( cudaMalloc( &d_velocity_z, volume_size ) );
	CUDA_SAFE_CALL( cudaMalloc( &d_pressure,   volume_size ) );

	CUDA_SAFE_CALL( cudaMemcpy( d_divergence, &zero_array[0], volume_size, cudaMemcpyHostToDevice ) );
	CUDA_SAFE_CALL( cudaMemcpy( d_velocity_x, &zero_array[0], volume_size, cudaMemcpyHostToDevice ) );
	CUDA_SAFE_CALL( cudaMemcpy( d_velocity_y, &zero_array[0], volume_size, cudaMemcpyHostToDevice ) );
	CUDA_SAFE_CALL( cudaMemcpy( d_velocity_z, &zero_array[0], volume_size, cudaMemcpyHostToDevice ) );
	CUDA_SAFE_CALL( cudaMemcpy( d_pressure,   &zero_array[0], volume_size, cudaMemcpyHostToDevice ) );

	//	fill array with zero :
	copy_back( d_divergence_array, d_divergence );
	copy_back( d_velocity_x_array, d_velocity_x );
	copy_back( d_velocity_y_array, d_velocity_y );
	copy_back( d_velocity_z_array, d_velocity_z );
	copy_back( d_pressure_array,   d_pressure );

	init_gpu();
}



void cfd_solver::copy_back( cudaArray *dst, float *src )
{
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr   = make_cudaPitchedPtr( src, nx*sizeof(float), ny, nz );
    copyParams.extent   = volume_extent;
    copyParams.kind     = cudaMemcpyDeviceToDevice;
    copyParams.dstArray = dst;
	CUDA_SAFE_CALL( cudaMemcpy3D(&copyParams) )
}


cfd_solver::~cfd_solver(void)
{
}



void cfd_solver::solve( float dt )
{
	/*std::vector<float> zero_array;
	zero_array.resize( nx*ny*nz, 0 );
	for (int i=0; i<zero_array.size(); i++) { zero_array[i] = randf_signed(); }

	CUDA_SAFE_CALL( cudaMemcpy( d_velocity_x, &zero_array[0], volume_size, cudaMemcpyHostToDevice ) );
	CUDA_SAFE_CALL( cudaMemcpy( d_velocity_y, &zero_array[0], volume_size, cudaMemcpyHostToDevice ) );
	CUDA_SAFE_CALL( cudaMemcpy( d_velocity_z, &zero_array[0], volume_size, cudaMemcpyHostToDevice ) );
	CUDA_SAFE_CALL( cudaMemcpy( d_pressure,   &zero_array[0], volume_size, cudaMemcpyHostToDevice ) );

	//	fill array with zero :
	copy_back( d_velocity_x_array, d_velocity_x );
	copy_back( d_velocity_y_array, d_velocity_y );
	copy_back( d_velocity_z_array, d_velocity_z );
	copy_back( d_pressure_array,   d_pressure );*/


	launch_gpu( dt );
}
