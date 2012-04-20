#pragma once


class cfd_solver {

	public:
					cfd_solver		( float xsz, float ysz, float zsz );
					~cfd_solver		( void );
		void		solve			( float dt );

	protected:
		float size_x;
		float size_y;
		float size_z;
	};

