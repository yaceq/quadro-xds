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




__global__ void advect( float *velx, float *vely, float *velz, float dt, int nx, int ny, int nz, float dx, float dy, float dz )
{
	int const x = blockIdx.x * blockDim.x + threadIdx.x;
	int const y = blockIdx.y * blockDim.y + threadIdx.y;
	int const z = blockIdx.z * blockDim.z + threadIdx.z;
	int const i = x + y*nx + z*nx*ny;

	float tx = ((float)x + 0.5f) / (float)nx;
	float ty = ((float)y + 0.5f) / (float)ny;
	float tz = ((float)z + 0.5f) / (float)nz;

	float vx = tex3D( tex_velocity_x, tx, ty, tz );
	float vy = tex3D( tex_velocity_y, tx, ty, tz );
	float vz = tex3D( tex_velocity_z, tx, ty, tz );

	velx[i]  = tex3D( tex_velocity_x,  tx - vx * dx * dt,  ty - vy * dy * dt,  tz - vz * dz * dt );
	vely[i]  = tex3D( tex_velocity_y,  tx - vx * dx * dt,  ty - vy * dy * dt,  tz - vz * dz * dt );
	velz[i]  = tex3D( tex_velocity_z,  tx - vx * dx * dt,  ty - vy * dy * dt,  tz - vz * dz * dt );
}




__global__ void propeller ( float *velx, float *vely, float *velz, int nx, int ny, int nz )
{
	//	cmpute voxel indices
	int const x = blockIdx.x * blockDim.x + threadIdx.x;
	int const y = blockIdx.y * blockDim.y + threadIdx.y;
	int const z = blockIdx.z * blockDim.z + threadIdx.z;
	int const i = x + y*nx + z*nx*ny;
  
	float3 v = make_float3(0,0,0);

	if ( x >= 12 && x < 20 )
	if ( y >= 12 && y < 20 )
	if ( z >= 12 && z < 20 )
		v = make_float3(0,1,0);

	velx[i] += v.x;
	vely[i] += v.y;
	velz[i] += v.z;
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
	float dx = size_x / nx;
	float dy = size_y / ny;
	float dz = size_z / nz;

    const int block_size_1d  = 128;		// num thread per block
    const int grid_size_1d   = iDivUp( m_particle_num, block_size_1d );	//	num grids

	dim3 const block_size_3d ( 8, 8, 8 ); 
	dim3 const grid_size_3d  ( iDivUp( nx, 8 ), iDivUp( ny, 8 ), iDivUp( nz, 8 ) );


	//	make propeller turbulence :
	advect<<<grid_size_3d, block_size_3d>>>( d_velocity_x, d_velocity_y, d_velocity_z, dt/8, nx, ny, nz, dx, dy, dz );
	advect<<<grid_size_3d, block_size_3d>>>( d_velocity_x, d_velocity_y, d_velocity_z, dt/8, nx, ny, nz, dx, dy, dz );
	advect<<<grid_size_3d, block_size_3d>>>( d_velocity_x, d_velocity_y, d_velocity_z, dt/8, nx, ny, nz, dx, dy, dz );

	propeller<<<grid_size_3d, block_size_3d>>> ( d_velocity_x, d_velocity_y, d_velocity_z, nx, ny, nz );

	copy_back( d_velocity_x_array, d_velocity_x );
	copy_back( d_velocity_y_array, d_velocity_y );
	copy_back( d_velocity_z_array, d_velocity_z );


	//	update particles :

	CUDA_SAFE_CALL( cudaGraphicsMapResources(1, &m_vbo_cuda) );

	float3 * d_vbo = NULL;
	size_t vbo_size = 0;
	CUDA_SAFE_CALL( cudaGraphicsResourceGetMappedPointer( (void**)&d_vbo, &vbo_size, m_vbo_cuda) );

	update_particles<<<grid_size_1d, block_size_1d>>>( m_particle_num, d_vbo, dt, nx, ny, nz );

	CUDA_SAFE_CALL( cudaGraphicsUnmapResources(1, &m_vbo_cuda) );

}
