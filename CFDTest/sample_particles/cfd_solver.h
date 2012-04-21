#pragma once


class cfd_solver {

	public:
					cfd_solver		( float xsz, float ysz, float zsz );
					~cfd_solver		( void );
		void		solve			( float dt );

		GLuint		get_vbo			( void ) { return m_vbo; }
		uint		get_prt_num		( void ) { return prt_num; }

	protected:
		
		void		launch_gpu	( float dt );

		int nx, ny, nz;

		uint	grid_size;
		float	size_x;
		float	size_y;
		float	size_z;

		uint	prt_num;
				
		GLuint					m_vbo;
		cudaGraphicsResource	*m_vbo_cuda;

		std::vector<float3>		m_position;

		float3	*d_position[2];		//	particles position
		float3	*d_velocity[2];		//	fluid velocity field
		float	*d_pressure[2];		//	fluid pressure field
	};

