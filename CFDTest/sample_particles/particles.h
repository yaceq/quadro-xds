#pragma once

struct Vertex {
	float4 pos;
	float4 color;
};

enum RunMode { MODE_CPU, MODE_GPU_NAIVE, MODE_GPU_TILED, MODE_NUM };

static const char *mode_names[3] = { "MODE_CPU", "MODE_GPU_NAIVE", "MODE_GPU_TILED" };

class ParticleSystem : noncopyable {
	public:

				ParticleSystem(int N);
		virtual ~ParticleSystem();

		void	Update( float dt, float view_dist );

		int		GetParticleNum() const { return m_particle_num; }
		GLuint	GetParticleVBO() const { return m_vbo; }

        int getRunMode() const {  return m_run_mode; }
        void setRunMode(int mode) { m_run_mode = mode; }


		void	LaunchCPU	( float dt, float view_dist );
		void	LaunchGPU	( float dt, float view_dist );

        float   getLastIterTime() const;

	private:
        int m_run_mode;

		int		m_particle_num;
		GLuint	m_vbo;
		cudaGraphicsResource * m_vbo_cuda;
		float3 *d_pos;
		float3 *d_vel;
		float3 *d_force;

        CudaStopWatch m_cuda_timer;

        bool m_host_valid;
        std::vector<float3> m_pos;
        std::vector<float3> m_vel;
        std::vector<float3> m_force;
        std::vector<Vertex> m_verts;
	};

inline __host__ __device__ float3 calcForce(float3 p0, float3 p1)
{
    const float SofteningSqr = 0.01f;

    float3 r = p1 - p0;
    float invDist = rsqrtf( dot(r, r) + SofteningSqr );
    return r * (invDist * invDist * invDist);
}

inline __host__ __device__ Vertex calcVisualProps(float3 pos, float dist)
{
    Vertex v;
	v.pos.x = pos.x;
	v.pos.y = pos.y;
	v.pos.z = pos.z;
	v.pos.w = 1;
	float light = 2000.0 / (1+dist*dist);
	v.color.x = 0.9*light;
	v.color.y = 1.0*light;
	v.color.z = 1.1*light;
	v.color.w = 1;
    return v;
}