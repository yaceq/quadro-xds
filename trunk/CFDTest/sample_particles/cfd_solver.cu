#include "stdafx.h"
#include "cfd_solver.h"


/*-----------------------------------------------------------------------------
	Device functions :
-----------------------------------------------------------------------------*/

texture<float, 3, cudaReadModeElementType> tex_velocity_x;
texture<float, 3, cudaReadModeElementType> tex_velocity_y;
texture<float, 3, cudaReadModeElementType> tex_velocity_z;
texture<float, 3, cudaReadModeElementType> tex_pressure;
texture<float, 3, cudaReadModeElementType> tex_divergence;
int *d_boundry;




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

__global__ void update_particles ( int particle_num, FVertex *prt, float dt, int nx, int ny, int nz )
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= particle_num) {
        return;
	}
	//pos[tid].z = pos[tid].z - dt;

	float3 pos = make_float3( prt[tid].pos.x, prt[tid].pos.y, prt[tid].pos.z );
	float3 uvw = pos / make_float3(nx,ny,nz) + make_float3(0.5f, 0.5f, 0.5f);
	prt[tid].pos.x = prt[tid].pos.x + tex3D( tex_velocity_x, uvw.x, uvw.y, uvw.z );
	prt[tid].pos.y = prt[tid].pos.y + tex3D( tex_velocity_y, uvw.x, uvw.y, uvw.z );
	prt[tid].pos.z = prt[tid].pos.z + tex3D( tex_velocity_z, uvw.x, uvw.y, uvw.z );
	if ( prt[tid].pos.x >  nx/2 ) prt[tid].pos.x =  nx/2; 
	if ( prt[tid].pos.x < -nx/2 ) prt[tid].pos.x = -nx/2; 
	if ( prt[tid].pos.y >  nx/2 ) prt[tid].pos.y =  nx/2; 
	if ( prt[tid].pos.y < -nx/2 ) prt[tid].pos.y = -nx/2; 
	if ( prt[tid].pos.z >  nx/2 ) prt[tid].pos.z =  nx/2; 
	if ( prt[tid].pos.z < -nx/2 ) prt[tid].pos.z = -nx/2; 
	//if ( prt[tid].pos.x >  nx/2 ) prt[tid].pos.x -= nx; 
	//if ( prt[tid].pos.x < -nx/2 ) prt[tid].pos.x += nx; 
	//if ( prt[tid].pos.y >  nx/2 ) prt[tid].pos.y -= ny; 
	//if ( prt[tid].pos.y < -nx/2 ) prt[tid].pos.y += ny; 
	//if ( prt[tid].pos.z >  nx/2 ) prt[tid].pos.z -= nz; 
	//if ( prt[tid].pos.z < -nx/2 ) prt[tid].pos.z += nz; 
	prt[tid].color.x = 0.1;
	prt[tid].color.y = 0.1;
	prt[tid].color.z = 0.2;
	prt[tid].color.w = 0;
}




__global__ void advect( float *velx, float *vely, float *velz, float dt, int nx, int ny, int nz, float dx, float dy, float dz )
{
	int const x = blockIdx.x * blockDim.x + threadIdx.x;
	int const y = blockIdx.y * blockDim.y + threadIdx.y;
	int const z = blockIdx.z * blockDim.z + threadIdx.z;
	int const i = x + y*nx + z*nx*ny;

	float u = ((float)x + 0.5f) / (float)nx;
	float v = ((float)y + 0.5f) / (float)ny;
	float w = ((float)z + 0.5f) / (float)nz;

	float vx = tex3D( tex_velocity_x, u, v, w );
	float vy = tex3D( tex_velocity_y, u, v, w );
	float vz = tex3D( tex_velocity_z, u, v, w );

	velx[i]  = tex3D( tex_velocity_x,  u - vx * dt,  v - vy * dt,  w - vz * dt );
	vely[i]  = tex3D( tex_velocity_y,  u - vx * dt,  v - vy * dt,  w - vz * dt );
	velz[i]  = tex3D( tex_velocity_z,  u - vx * dt,  v - vy * dt,  w - vz * dt );
}


//#define ADDR(x,y,z) ((x) + (y)*nx + (z)*nx*ny)

__global__ void jacobi ( float *velx, float *vely, float *velz, int nx, int ny, int nz, float alpha, float beta )
{
	int const x = blockIdx.x * blockDim.x + threadIdx.x;
	int const y = blockIdx.y * blockDim.y + threadIdx.y;
	int const z = blockIdx.z * blockDim.z + threadIdx.z;
	int const i = x + y*nx + z*nx*ny;

	float du = 1.0 / (float)nx;
	float dv = 1.0 / (float)ny;
	float dw = 1.0 / (float)nz;
	float u = ((float)x + 0.5f) / (float)nx;
	float v = ((float)y + 0.5f) / (float)ny;
	float w = ((float)z + 0.5f) / (float)nz;

	
	velx[i]	=	tex3D( tex_velocity_x, u+du, v, w )
			+	tex3D( tex_velocity_x, u-du, v, w )
			+	tex3D( tex_velocity_x, u, v+dv, w )
			+	tex3D( tex_velocity_x, u, v-dv, w )
			+	tex3D( tex_velocity_x, u, v, w+dw )
			+	tex3D( tex_velocity_x, u, v, w-dw )
			+	tex3D( tex_velocity_x, u, v, w ) *	alpha;

	vely[i]	=	tex3D( tex_velocity_y, u+du, v, w )
			+	tex3D( tex_velocity_y, u-du, v, w )
			+	tex3D( tex_velocity_y, u, v+dv, w )
			+	tex3D( tex_velocity_y, u, v-dv, w )
			+	tex3D( tex_velocity_y, u, v, w+dw )
			+	tex3D( tex_velocity_y, u, v, w-dw )
			+	tex3D( tex_velocity_y, u, v, w ) *	alpha;

	velz[i]	=	tex3D( tex_velocity_z, u+du, v, w )
			+	tex3D( tex_velocity_z, u-du, v, w )
			+	tex3D( tex_velocity_z, u, v+dv, w )
			+	tex3D( tex_velocity_z, u, v-dv, w )
			+	tex3D( tex_velocity_z, u, v, w+dw )
			+	tex3D( tex_velocity_z, u, v, w-dw )
			+	tex3D( tex_velocity_z, u, v, w ) *	alpha;

	velx[i] /= beta;
	vely[i] /= beta;
	velz[i] /= beta;
}


__global__ void jacobi2 ( float *pressure, int nx, int ny, int nz, float alpha, float beta )
{
	int const x = blockIdx.x * blockDim.x + threadIdx.x;
	int const y = blockIdx.y * blockDim.y + threadIdx.y;
	int const z = blockIdx.z * blockDim.z + threadIdx.z;
	int const i = x + y*nx + z*nx*ny;

	float du = 1.0 / (float)nx;
	float dv = 1.0 / (float)ny;
	float dw = 1.0 / (float)nz;
	float u = ((float)x + 0.5f) / (float)nx;
	float v = ((float)y + 0.5f) / (float)ny;
	float w = ((float)z + 0.5f) / (float)nz;

	pressure[i]	=	tex3D( tex_pressure,   u+du, v, w )
				+	tex3D( tex_pressure,   u-du, v, w )
				+	tex3D( tex_pressure,   u, v+dv, w )
				+	tex3D( tex_pressure,   u, v-dv, w )
				+	tex3D( tex_pressure,   u, v, w+dw )
				+	tex3D( tex_pressure,   u, v, w-dw )
				+	tex3D( tex_divergence, u, v, w ) *	alpha;

	pressure[i] /= beta;
}



__global__ void divergence ( float *div, int nx, int ny, int nz, float dx, float dy, float dz )
{
	int const x = blockIdx.x * blockDim.x + threadIdx.x;
	int const y = blockIdx.y * blockDim.y + threadIdx.y;
	int const z = blockIdx.z * blockDim.z + threadIdx.z;
	int const i = x + y*nx + z*nx*ny;

	float du = 1.0 / (float)nx;
	float dv = 1.0 / (float)ny;
	float dw = 1.0 / (float)nz;
	float u = ((float)x + 0.5f) / (float)nx;
	float v = ((float)y + 0.5f) / (float)ny;
	float w = ((float)z + 0.5f) / (float)nz;

	div[i]	=	( tex3D( tex_velocity_x, u+du, v, w ) -	tex3D( tex_velocity_x, u-du, v, w ) ) / 2.0f / dx
			+	( tex3D( tex_velocity_y, u, v+dv, w ) -	tex3D( tex_velocity_y, u, v-dv, w ) ) / 2.0f / dy
			+	( tex3D( tex_velocity_z, u, v, w+dw ) -	tex3D( tex_velocity_z, u, v, w-dw ) ) / 2.0f / dz;
}



__global__ void boundry ( float *velx, float *vely, float *velz, float *pressure, int *bounds, int nx, int ny, int nz )
{
	int const x = blockIdx.x * blockDim.x + threadIdx.x;
	int const y = blockIdx.y * blockDim.y + threadIdx.y;
	int const z = blockIdx.z * blockDim.z + threadIdx.z;
	int const i = x + y*nx + z*nx*ny;
	int remap = bounds[i];
		
	if (remap>=0) {
		velx[i]		= -velx		[ remap ];
		vely[i]		= -vely		[ remap ];
		velz[i]		= -velz		[ remap ];
		pressure[i]	= pressure	[ remap ];
	}
}



__global__ void gradient ( float *velx, float *vely, float *velz, int nx, int ny, int nz, float dx, float dy, float dz )
{
	int const x = blockIdx.x * blockDim.x + threadIdx.x;
	int const y = blockIdx.y * blockDim.y + threadIdx.y;
	int const z = blockIdx.z * blockDim.z + threadIdx.z;
	int const i = x + y*nx + z*nx*ny;

	float du = 1.0 / (float)nx;
	float dv = 1.0 / (float)ny;
	float dw = 1.0 / (float)nz;
	float u = ((float)x + 0.5f) / (float)nx;
	float v = ((float)y + 0.5f) / (float)ny;
	float w = ((float)z + 0.5f) / (float)nz;

	float grad_x = ( tex3D( tex_pressure, u+du, v, w ) -	tex3D( tex_pressure, u-du, v, w ) ) / 2.0f / dx;
	float grad_y = ( tex3D( tex_pressure, u, v+dv, w ) -	tex3D( tex_pressure, u, v-dv, w ) ) / 2.0f / dy;
	float grad_z = ( tex3D( tex_pressure, u, v, w+dw ) -	tex3D( tex_pressure, u, v, w-dw ) ) / 2.0f / dz;

	velx[i] -= grad_x;
	vely[i] -= grad_y;
	velz[i] -= grad_z;
}



__global__ void propeller ( float *velx, float *vely, float *velz, int nx, int ny, int nz, float sx, float sy, float sz, int px, int py, int pz )
{
	//	cmpute voxel indices
	int const x = blockIdx.x * blockDim.x + threadIdx.x;
	int const y = blockIdx.y * blockDim.y + threadIdx.y;
	int const z = blockIdx.z * blockDim.z + threadIdx.z;
	int const i = x + y*nx + z*nx*ny;
  
	float3 v = make_float3(0,0,0);

	if ( x >= px-1 && x <= px+1 )
	if ( y >= py-1 && y <= py+1 )
	if ( z >= pz-1 && z <= pz+1 )
		v = make_float3(sx,sy,sz);

	//if(x == 14 && ( y == 14 || y == 15 ) && z == 14) v = make_float3(sx,sy,sz);
	//if(x == 18 && ( y == 14 || y == 15 ) && z == 14) v = make_float3(sx,sy,sz);
	//if(x == 14 && ( y == 18 || y == 19 ) && z == 14) v = make_float3(sx,sy,sz);
	//if(x == 18 && ( y == 18 || y == 19 ) && z == 14) v = make_float3(sx,sy,sz);


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
    tex.addressMode[0]	=	 cudaAddressModeClamp;   // wrap texture coordinates
    tex.addressMode[1]	=	 cudaAddressModeClamp;
    tex.addressMode[2]	=	 cudaAddressModeClamp;

    CUDA_SAFE_CALL( cudaBindTextureToArray( tex, array, desc ) );
}



void cfd_solver::init_gpu()
{
	bind_array_to_texture( d_divergence_array, tex_divergence, volume_chan_desc );
	bind_array_to_texture( d_velocity_x_array, tex_velocity_x, volume_chan_desc );
	bind_array_to_texture( d_velocity_y_array, tex_velocity_y, volume_chan_desc );
	bind_array_to_texture( d_velocity_z_array, tex_velocity_z, volume_chan_desc );
	bind_array_to_texture( d_pressure_array,   tex_pressure,   volume_chan_desc );

	int sz = nx*ny*nz * sizeof(int);
	std::vector<int> h_boundry(sz);
	CUDA_SAFE_CALL( cudaMalloc( &d_boundry, sz ) );

	#define ADDR(i,j,k) (i) + (j) * nx + (k) * nx * ny

	for (int i=0; i<nx; i++)
	for (int j=0; j<ny; j++)
	for (int k=0; k<nz; k++)
	{											   
		h_boundry[ ADDR(i,j,k) ] = -1;
		if (i==0)    {	h_boundry[ ADDR(i,j,k) ] = ADDR(i+1,j,k);	};
		if (i==nx-1) {	h_boundry[ ADDR(i,j,k) ] = ADDR(i-1,j,k);	};
		if (j==0)    {	h_boundry[ ADDR(i,j,k) ] = ADDR(i,j+1,k);	};
		if (j==ny-1) {	h_boundry[ ADDR(i,j,k) ] = ADDR(i,j-1,k);	};
		if (k==0)    {	h_boundry[ ADDR(i,j,k) ] = ADDR(i,j,k+1);	};
		if (k==nz-1) {	h_boundry[ ADDR(i,j,k) ] = ADDR(i,j,k-1);	};
	}
	CUDA_SAFE_CALL( cudaMemcpy( d_boundry, &h_boundry[0], sz, cudaMemcpyHostToDevice ) );
}


float t = 0;

float y = 4;

void cfd_solver::launch_gpu( float dt )
{
	t += dt;

	float dx = size_x / nx;
	float dy = size_y / ny;
	float dz = size_z / nz;

    const int block_size_1d  = 64;		// num thread per block
    const int grid_size_1d   = iDivUp( m_particle_num, block_size_1d );	//	num grids

	dim3 block_size_3d ( 4, 4, 32 ); 
	dim3 grid_size_3d  ( iDivUp( nx, block_size_3d.x ), iDivUp( ny, block_size_3d.y ), iDivUp( nz, block_size_3d.z ) );


	/*---------------------------------------------------------------
	*	advection :
	---------------------------------------------------------------*/

	//	make propeller turbulence :
	advect<<<grid_size_3d, block_size_3d>>>( d_velocity_x, d_velocity_y, d_velocity_z, dt, nx, ny, nz, dx, dy, dz );

	copy_back( d_velocity_x_array, d_velocity_x );
	copy_back( d_velocity_y_array, d_velocity_y );
	copy_back( d_velocity_z_array, d_velocity_z );

	int N = 10;
	float visc  = 0.073f;
	float alpha = dx*dx / visc / dt;
	float beta  = 6 + alpha;

	
	/*---------------------------------------------------------------
	*	poisson viscosity :
	---------------------------------------------------------------*/

	y += dt;

	for (int i=0; i<N; i++) {
		jacobi<<<grid_size_3d, block_size_3d>>>( d_velocity_x, d_velocity_y, d_velocity_z, nx, ny, nz, alpha, beta );

		boundry<<<grid_size_3d, block_size_3d>>>( d_velocity_x, d_velocity_y, d_velocity_z, d_pressure, d_boundry, nx, ny, nz );

		copy_back( d_velocity_x_array, d_velocity_x );
		copy_back( d_velocity_y_array, d_velocity_y );
		copy_back( d_velocity_z_array, d_velocity_z );

	} //*/


	/*---------------------------------------------------------------
	*	external forces :
	---------------------------------------------------------------*/

	float r0  = ((float)rand()/(float)RAND_MAX*2-1) / 8.0f;
	float r1  = ((float)rand()/(float)RAND_MAX*2-1) / 8.0f;
	float r2  = ((float)rand()/(float)RAND_MAX*2-1) / 8.0f;
	float r3  = ((float)rand()/(float)RAND_MAX*2-1) / 8.0f;

	propeller<<<grid_size_3d, block_size_3d>>> ( d_velocity_x, d_velocity_y, d_velocity_z, nx, ny, nz, 0+r0, -2+r0, 0+r0, 13, y, 13 );
	propeller<<<grid_size_3d, block_size_3d>>> ( d_velocity_x, d_velocity_y, d_velocity_z, nx, ny, nz, 0+r1, -2+r1, 0+r1, 19, y, 13 );
	propeller<<<grid_size_3d, block_size_3d>>> ( d_velocity_x, d_velocity_y, d_velocity_z, nx, ny, nz, 0+r2, -2+r2, 0+r2, 13, y, 19 );
	propeller<<<grid_size_3d, block_size_3d>>> ( d_velocity_x, d_velocity_y, d_velocity_z, nx, ny, nz, 0+r3, -2+r3, 0+r3, 19, y, 19 );

	/*---------------------------------------------------------------
	*	poisson pressure :
	---------------------------------------------------------------*/

	divergence<<<grid_size_3d, block_size_3d>>> ( d_divergence, nx, ny, nz, dx, dy, dz );

	copy_back( d_divergence_array, d_divergence );

	alpha = -dx*dx;
	beta  = 6;

	for (int i=0; i<N; i++) {

		jacobi2<<<grid_size_3d, block_size_3d>>>( d_pressure, nx, ny, nz, alpha, beta );

		boundry<<<grid_size_3d, block_size_3d>>>( d_velocity_x, d_velocity_y, d_velocity_z, d_pressure, d_boundry, nx, ny, nz );

		copy_back( d_pressure_array, d_pressure );
	} //*/							   


	/*---------------------------------------------------------------
	*	gradient :
	---------------------------------------------------------------*/

	gradient<<<grid_size_3d, block_size_3d>>> ( d_velocity_x, d_velocity_y, d_velocity_z, nx, ny, nz, dx, dy, dz );

	boundry<<<grid_size_3d, block_size_3d>>>( d_velocity_x, d_velocity_y, d_velocity_z, d_pressure, d_boundry, nx, ny, nz );

	copy_back( d_velocity_x_array, d_velocity_x );
	copy_back( d_velocity_y_array, d_velocity_y );
	copy_back( d_velocity_z_array, d_velocity_z );


	/*---------------------------------------------------------------
	*	Stufff...
	---------------------------------------------------------------*/
	cudaError err = cudaGetLastError();
    if( cudaSuccess != err) {                                           
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", __FILE__, __LINE__, cudaGetErrorString( err) );         
    } 


	/*---------------------------------------------------------------
	*	update particles
	---------------------------------------------------------------*/

	CUDA_SAFE_CALL( cudaGraphicsMapResources(1, &m_vbo_cuda) );

	FVertex * d_vbo = NULL;
	size_t vbo_size = 0;
	CUDA_SAFE_CALL( cudaGraphicsResourceGetMappedPointer( (void**)&d_vbo, &vbo_size, m_vbo_cuda) );

	update_particles<<<grid_size_1d, block_size_1d>>>( m_particle_num, d_vbo, dt, nx, ny, nz );

	CUDA_SAFE_CALL( cudaGraphicsUnmapResources(1, &m_vbo_cuda) );

}
