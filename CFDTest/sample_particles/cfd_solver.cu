#include "stdafx.h"
#include "cfd_solver.h"


/*-----------------------------------------------------------------------------
	Device functions :
-----------------------------------------------------------------------------*/

texture<float, 3, cudaReadModeElementType> tex_velocity_x;
texture<float, 3, cudaReadModeElementType> tex_velocity_y;
texture<float, 3, cudaReadModeElementType> tex_velocity_z;
texture<float, 3, cudaReadModeElementType> tex_pressure;


void __device__ write_array_3d_float3 ( cudaPitchedPtr ptr, int x, int y, int z, float3 value, int nx, int ny, int nz )
{
	int i = ( z * ptr.ysize + y ) * ptr.pitch/sizeof(float3) + x;
	float3 *p = (float3*)ptr.ptr;
	p[i] = value;
}


float3 __device__ read_array_3d_float3 ( cudaPitchedPtr ptr, int x, int y, int z, int nx, int ny, int nz )
{
	int i = ( z * ptr.ysize + y ) * ptr.pitch/sizeof(float3) + x;
	float3 *p = (float3*)ptr.ptr;
	return p[i];
}

/*float3 __device__ read_array_3d_float3 ( cudaPitchedPtr ptr, float x, float y, float z, int nx, int ny, int nz )
{
	int x000 = floor(x+0);	int y000 = floor(y+0);	int z000 = floor(y+0);	
	int x001 = floor(x+0);	int y000 = floor(y+0);	int z000 = floor(y+1);	
	int x010 = floor(x+0);	int y000 = floor(y+1);	int z000 = floor(y+0);	
	int x011 = floor(x+0);	int y000 = floor(y+1);	int z000 = floor(y+1);	
	int x100 = floor(x+1);	int y000 = floor(y+0);	int z000 = floor(y+0);	
	int x101 = floor(x+1);	int y000 = floor(y+0);	int z000 = floor(y+1);	
	int x110 = floor(x+1);	int y000 = floor(y+1);	int z000 = floor(y+0);	
	int x111 = floor(x+1);	int y000 = floor(y+1);	int z000 = floor(y+1);	
	float fx = fmod(x,1);
	float fy = fmod(x,1);
	float fz = fmod(x,1);

	return read_array_3d_float3( ptr, x000, y000, z000 );
} */


/*-----------------------------------------------------------------------------
	Kernels :
-----------------------------------------------------------------------------*/

__global__ void update_particles ( int particle_num, float3 *pos, float dt, int nx, int ny, int nz )
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= particle_num) {
        return;
	}
	//pos[tid].z = pos[tid].z - dt;

	float3 uvw = pos[tid] / make_float3(nx,ny,nz) + make_float3(0.5f, 0.5f, 0.5f);
	pos[tid].x = pos[tid].x + tex3D( tex_velocity_x, uvw.x, uvw.y, uvw.z );
	pos[tid].y = pos[tid].y + tex3D( tex_velocity_y, uvw.x, uvw.y, uvw.z );
	pos[tid].z = pos[tid].z + tex3D( tex_velocity_z, uvw.x, uvw.y, uvw.z );
	if ( pos[tid].x >  nx/2 ) pos[tid].x = -nx/2; 
	if ( pos[tid].x < -nx/2 ) pos[tid].x =  nx/2; 
	if ( pos[tid].y >  nx/2 ) pos[tid].y = -ny/2; 
	if ( pos[tid].y < -nx/2 ) pos[tid].y =  ny/2; 
	if ( pos[tid].z >  nx/2 ) pos[tid].z = -nz/2; 
	if ( pos[tid].z < -nx/2 ) pos[tid].z =  nz/2; 
}



__global__ void propeller ( cudaPitchedPtr velocity, int nx, int ny, int nz )
{
	//	cmpute voxel indices
	int const x = blockIdx.x * blockDim.x + threadIdx.x;
	int const y = blockIdx.y * blockDim.y + threadIdx.y;
	int const z = blockIdx.z * blockDim.z + threadIdx.z;
  
	float3 p = make_float3( x,y,z ) - make_float3( nx/2, ny/2, nz/3 );
	float3 v = make_float3( length( p ), 0, 0 );

	write_array_3d_float3( velocity, x,y,z, v, nx,ny,nz );
}

/*-----------------------------------------------------------------------------
	Do GPU stuff :
-----------------------------------------------------------------------------*/

void bind_array_to_texture ( cudaArray *array, texture<float, 3, cudaReadModeElementType> &tex, cudaChannelFormatDesc desc )
{
    tex.normalized		=	 true;                  // access with normalized texture coordinates
    tex.filterMode		=	 cudaFilterModeLinear;  // linear interpolation
    tex.addressMode[0]	=	 cudaAddressModeWrap;   // wrap texture coordinates
    tex.addressMode[1]	=	 cudaAddressModeWrap;
    tex.addressMode[2]	=	 cudaAddressModeWrap;

    CUDA_SAFE_CALL( cudaBindTextureToArray( tex, array, desc ) );
}



void cfd_solver::init_gpu()
{
	bind_array_to_texture( d_velocity_x_array, tex_velocity_x, volume_chan_desc );
	bind_array_to_texture( d_velocity_y_array, tex_velocity_y, volume_chan_desc );
	bind_array_to_texture( d_velocity_z_array, tex_velocity_z, volume_chan_desc );
	bind_array_to_texture( d_pressure_array,   tex_pressure,   volume_chan_desc );
}



void cfd_solver::launch_gpu( float dt )
{
    const int block_size_1d  = 128;		// num thread per block
    const int grid_size_1d   = iDivUp( m_particle_num, block_size_1d );	//	num grids

/*	dim3 const block_size_3d ( 8, 8, 8 ); 
	dim3 const grid_size_3d  ( iDivUp( nx, 8 ), iDivUp( ny, 8 ), iDivUp( nz, 8 ) );

	int pitch_1d = d_pressure[0].pitch / sizeof( float );
	int pitch_3d = d_velocity[0].pitch / sizeof( float3 );



	//	make propeller turbulence :

	propeller<<<grid_size_3d, block_size_3d>>> ( d_velocity[0], nx, ny, nz );*/


	//	update particles :

	CUDA_SAFE_CALL( cudaGraphicsMapResources(1, &m_vbo_cuda) );

	float3 * d_vbo = NULL;
	size_t vbo_size = 0;
	CUDA_SAFE_CALL( cudaGraphicsResourceGetMappedPointer( (void**)&d_vbo, &vbo_size, m_vbo_cuda) );

	update_particles<<<grid_size_1d, block_size_1d>>>( m_particle_num, d_vbo, dt, nx, ny, nz );

	CUDA_SAFE_CALL( cudaGraphicsUnmapResources(1, &m_vbo_cuda) );

}
