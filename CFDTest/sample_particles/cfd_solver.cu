#include "stdafx.h"
#include "cfd_solver.h"


/*-----------------------------------------------------------------------------
	Kernels :
-----------------------------------------------------------------------------*/

__global__ void update_particles ( int particle_num, float3 *pos, float dt )
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= particle_num) {
        return;
	}
	pos[tid].z = pos[tid].z;
}



/*-----------------------------------------------------------------------------
	Do GPU stuff :
-----------------------------------------------------------------------------*/

void cfd_solver::launch_gpu( float dt )
{


    const int block_size1d = 128;
    const int grid_size1d = iDivUp(m_particle_num, block_size1d);


	CUDA_SAFE_CALL( cudaGraphicsMapResources(1, &m_vbo_cuda) );

	float3 * d_vbo = NULL;
	size_t vbo_size = 0;
	CUDA_SAFE_CALL( cudaGraphicsResourceGetMappedPointer( (void**)&d_vbo, &vbo_size, m_vbo_cuda) );

	update_particles<<<grid_size1d, block_size1d>>>( m_particle_num, d_vbo, dt );

	CUDA_SAFE_CALL( cudaGraphicsUnmapResources(1, &m_vbo_cuda) );

}
