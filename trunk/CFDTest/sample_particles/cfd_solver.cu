#include "stdafx.h"
#include "cfd_solver.h"


/*-----------------------------------------------------------------------------
	Kernels :
-----------------------------------------------------------------------------*/

__global__ void updateParticles ( int particle_num, float3 *pos, float dt )
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= particle_num) {
        return;
	}
	pos[tid] = pos[tid] + 0.0001f * dt;
}



/*-----------------------------------------------------------------------------
	Do GPU stuff :
-----------------------------------------------------------------------------*/

void cfd_solver::launch_gpu( float dt )
{

}
