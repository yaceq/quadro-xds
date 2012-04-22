#pragma once


class cfd_solver {

	public:
					cfd_solver		( float xsz, float ysz, float zsz );
					~cfd_solver		( void );
		void		solve			( float dt );

		GLuint		get_vbo			( void ) { return m_vbo; }
		uint		get_prt_num		( void ) { return m_particle_num; }

	protected:
		
		void		init_gpu	();
		void		launch_gpu	( float dt );
		void		copy_back	( cudaArray *dst, float *src );

		int nx, ny, nz;

		uint	grid_size;
		float	size_x;
		float	size_y;
		float	size_z;

		uint	m_particle_num;
				
		GLuint					m_vbo;
		cudaGraphicsResource	*m_vbo_cuda;

		std::vector<float3>		m_position;

		cudaExtent				volume_extent;
		cudaChannelFormatDesc	volume_chan_desc;
		int						volume_size;

		cudaArray	*d_velocity_x_array;
		cudaArray	*d_velocity_y_array;
		cudaArray	*d_velocity_z_array;
		cudaArray	*d_pressure_array;

		float	*d_velocity_x;
		float	*d_velocity_y;
		float	*d_velocity_z;
		float	*d_pressure;

	};

