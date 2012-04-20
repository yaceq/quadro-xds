#include "stdafx.h"
#include "particles.h"

const int BLOCK_SIZE = 256;

__shared__ float3 shared_pos[BLOCK_SIZE];

__global__ void calcForcesKernel( int particle_num, const float3 * pos, float3 * forces);
__global__ void calcForcesTiledKernel( int particle_num, const float3 * pos, float3 * forces);
__global__ void integrateForcesKernel( int particle_num, float3 * pos, float3 * vel, const float3 * forces, float dt);
__global__ void calcVisualProperties( int particle_num, float3 *pos, float3 *vel, Vertex *d_vbo, float dist );

/*-----------------------------------------------------------------------------
	DEVICE code :
-----------------------------------------------------------------------------*/

//
//	ParticleSystem::LaunchGPU()
//
void ParticleSystem::LaunchGPU( float dt, float view_dist )
{
    if (m_host_valid)
    {
        int buf_size = sizeof(float3) * m_particle_num;
	    CUDA_SAFE_CALL( cudaMemcpy( d_pos, &m_pos[0], buf_size, cudaMemcpyHostToDevice ) );
	    CUDA_SAFE_CALL( cudaMemcpy( d_vel, &m_vel[0], buf_size, cudaMemcpyHostToDevice ) );
        m_host_valid = false;
    }


	//  solve N-body :
    const int block_size = 256;
    const int grid_size = iDivUp(m_particle_num, block_size);
    
    if (m_run_mode == MODE_GPU_NAIVE)
        calcForcesKernel<<<grid_size, block_size>>>(m_particle_num, d_pos, d_force);
    else
        calcForcesTiledKernel<<<grid_size, block_size>>>(m_particle_num, d_pos, d_force);

	integrateForcesKernel<<<grid_size, block_size>>>(m_particle_num, d_pos, d_vel, d_force, dt);

	//	update VBO :
	CUDA_SAFE_CALL( cudaGraphicsMapResources(1, &m_vbo_cuda) );

	Vertex * d_vbo = NULL;
	size_t vbo_size = 0;
	CUDA_SAFE_CALL( cudaGraphicsResourceGetMappedPointer( (void**)&d_vbo, &vbo_size, m_vbo_cuda) );

	calcVisualProperties<<<grid_size, block_size>>>(m_particle_num, d_pos, d_vel, d_vbo, view_dist);

	CUDA_SAFE_CALL( cudaGraphicsUnmapResources(1, &m_vbo_cuda) );
}


//
//	calcForcesKernel()
//
__global__ void calcForcesKernel( int particle_num, const float3 * pos, float3 * forces)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= particle_num)
        return;

	float3 cur_pos = pos[tid];
	float3 force = make_float3(0);
    for (int i = 0; i < particle_num; ++i)
    {
        force += calcForce(cur_pos, pos[i]);
    }
    forces[tid] = force;
}


//
//	calcForcesKernelTiled()
//

__global__ void calcForcesTiledKernel( int particle_num, const float3 * pos, float3 * forces)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= particle_num)
        return;
    
	float3 cur_pos = pos[tid];
	float3 force = make_float3(0);
    for (int tile = 0; tile < particle_num; tile += BLOCK_SIZE)
    {
        shared_pos[threadIdx.x] = pos[tile + threadIdx.x];
        __syncthreads();
        
        for (int i = 0; i < BLOCK_SIZE; ++i)
            force += calcForce(cur_pos, shared_pos[i]);
        __syncthreads();
    }
    forces[tid] = force;
}


//
//	integrateForcesKernel()
//
__global__ void integrateForcesKernel( int particle_num, float3 * pos, float3 * vel, const float3 * forces, float dt)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= particle_num)
        return;

	float3 v = vel[tid] + forces[tid] * dt;
	float3 p = pos[tid] + v*dt;
	
	pos[tid] = p;
	vel[tid] = v;
}


//
//	calcVisualProperties
//
__global__ void calcVisualProperties( int particle_num, float3 *pos, float3 *vel, Vertex *d_vbo, float dist )
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= particle_num)
        return;

	d_vbo[tid] = calcVisualProps(pos[tid], dist);
}