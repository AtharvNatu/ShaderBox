#pragma once

#include <cuda_runtime.h>
#include <cufft.h>

#define _USE_MATH_DEFINES
#include <math.h>
#include <cstdlib>

#define MESH_SIZE               256
#define SPECTRUM_SIZE_W 	    MESH_SIZE + 4
#define SPECTRUM_SIZE_H 	    MESH_SIZE + 1
#define OCEAN_WAVES_SPEED       0.0125F
#define CUDART_PI_F             3.141592654F
#define CUDART_SQRT_HALF_F      0.707106781F

// Ocean Related Variables
// ---------------------------------
float *heightMap_CPU = NULL;
float2 *slope_CPU = NULL;
float2 h_ht_arr[MESH_SIZE * MESH_SIZE];

unsigned int spectrumW = MESH_SIZE + 4;
unsigned int spectrumH = MESH_SIZE + 1;

bool animate = true;
bool drawPoints = false;
bool wireFrame = false;
bool g_hasDouble = false;

// FFT data
cufftHandle fftPlan;
float2 *d_h0 = 0; // heightfield at time 0
float2 *h_h0 = 0;
float2 *d_ht = 0; // heightfield at time t
float2 *d_slope = 0;

// pointers to device object
float *g_hptr = NULL;
float2 *g_sptr = NULL;

unsigned int meshSize = 2;
const float gravitationalConstant = 9.81f;		 // gravitational constant
const float waveScaleFactor = 1e-7f;		 // wave scale factor
const float patchSize = 100; // patch size
float windSpeed = 100.0f;
float windDir = CUDART_PI_F / 3.0f;
float dirDepend = 0.07f;

// StopWatchInterface *timer = NULL;
float animTime = 0.0f;
float prevTime = 0.0f;
float animationRate = -0.001f;

cudaError_t cudaResult;
struct cudaGraphicsResource *positionVertexBufferResource = NULL;
struct cudaGraphicsResource *heightVertexBufferResource = NULL;
struct cudaGraphicsResource *slopeVertexBufferResource = NULL;

// BOOL onGPU = FALSE;
// BOOL updateBuffer = FALSE;
// BOOL displayWireFrame = FALSE;

// GLfloat rotateX = 20.0f;
// GLfloat rotateY = 0.0f;
// GLfloat rotateZ = 0.0f;

// GLfloat translateX = 0.0f;
// GLfloat translateY = 0.0f;
// GLfloat translateZ = -2.0f;

// GLfloat animationDelayValue = 0.0f;
// ---------------------------------

float urand() { return rand() / (float)RAND_MAX; }

// Generates Gaussian random number with mean 0 and standard deviation 1.
float gauss()
{
	float u1 = urand();
	float u2 = urand();

	if (u1 < 1e-6f)
	{
		u1 = 1e-6f;
	}

	return sqrtf(-2 * logf(u1)) * cosf(2 * CUDART_PI_F * u2);
}

// Phillips spectrum
// (Kx, Ky) - normalized wave vector
// Vdir - wind angle in radians
// V - wind speed
// A - constant
float phillips(float Kx, float Ky, float Vdir, float V, float A,
			   float dir_depend)
{
	float k_squared = Kx * Kx + Ky * Ky;

	if (k_squared == 0.0f)
	{
		return 0.0f;
	}

	// largest possible wave from constant wind of velocity v
	float L = V * V / gravitationalConstant;

	float k_x = Kx / sqrtf(k_squared);
	float k_y = Ky / sqrtf(k_squared);
	float w_dot_k = k_x * cosf(Vdir) + k_y * sinf(Vdir);

	float phillips = A * expf(-1.0f / (k_squared * L * L)) /
					 (k_squared * k_squared) * w_dot_k * w_dot_k;

	// filter out waves moving opposite to wind
	if (w_dot_k < 0.0f)
	{
		phillips *= dir_depend;
	}

	// damp out waves with very small length w << l
	// float w = L / 10000;
	// phillips *= expf(-k_squared * w * w);

	return phillips;
}

// Generate base heightfield in frequency space
void generate_h0(float2 *h0)
{
	for (unsigned int y = 0; y <= meshSize; y++)
	{
		for (unsigned int x = 0; x <= meshSize; x++)
		{
			float kx = (-(int)meshSize / 2.0f + x) * (2.0f * CUDART_PI_F / patchSize);
			float ky = (-(int)meshSize / 2.0f + y) * (2.0f * CUDART_PI_F / patchSize);

			float P = sqrtf(phillips(kx, ky, windDir, windSpeed, waveScaleFactor, dirDepend));

			if (kx == 0.0f && ky == 0.0f)
			{
				P = 0.0f;
			}

			// float Er = urand()*2.0f-1.0f;
			// float Ei = urand()*2.0f-1.0f;
			float Er = gauss();
			float Ei = gauss();

			float h0_re = Er * P * CUDART_SQRT_HALF_F;
			float h0_im = Ei * P * CUDART_SQRT_HALF_F;

			int i = y * spectrumW + x;
			h0[i].x = h0_re;
			h0[i].y = h0_im;
		}
	}
}


float2 cpu_conjugate(float2 arg)
{
    return make_float2(arg.x, -arg.y);
}

float2 cpu_complex_exp(float arg)
{
    return make_float2(cosf(arg), sinf(arg));
}

float2 cpu_complex_add(float2 a, float2 b)
{
    return make_float2(a.x + b.x, a.y + b.y);
}

float2 cpu_complex_mult(float2 ab, float2 cd)
{
    return make_float2(ab.x * cd.x - ab.y * cd.y, ab.x * cd.y + ab.y * cd.x);
}

// CUDA Kernel
int cuda_iDivUp(int a, int b)
{
  return ((a + (b - 1)) / b);
}

__device__ float2 conjugate(float2 arg)
{
  return (make_float2(arg.x, -arg.y));
}

__device__ float2 complex_exp(float arg)
{
  return (make_float2(cosf(arg), sinf(arg)));
}

__device__ float2 complex_add(float2 a, float2 b)
{
  return (make_float2(a.x + b.x, a.y + b.y));
}

__device__ float2 complex_mult(float2 ab, float2 cd)
{
  return (make_float2(ab.x * cd.x - ab.y * cd.y, ab.x * cd.y + ab.y * cd.x));
}

__global__ void generateSpectrumKernel(float2 *h0, float2 *ht, unsigned int in_width, unsigned int out_width, unsigned int out_height, float t, float patchSize)
{
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int in_index = y * in_width + x;
  unsigned int in_mindex = (out_height - y) * in_width + (out_width - x);
  unsigned int out_index = y * out_width + x;

  // calculate wave vector
  float2 k;
  k.x = (-(int)out_width / 2.0f + x) * (2.0f * CUDART_PI_F / patchSize);
  k.y = (-(int)out_width / 2.0f + y) * (2.0f * CUDART_PI_F / patchSize);

  // calculate dispersion w(k)
  float k_len = sqrtf(k.x * k.x + k.y * k.y);
  float w = sqrtf(9.81f * k_len);

  if ((x < out_width) && (y < out_height))
  {
    float2 h0_k = h0[in_index];
    float2 h0_mk = h0[in_mindex];

    // output frequency-space complex values
    ht[out_index] = complex_add(complex_mult(h0_k, complex_exp(w * t)), complex_mult(conjugate(h0_mk), complex_exp(-w * t)));
  }
}

__global__ void updateHeightmapKernel(float *heightMap, float2 *ht, unsigned int width)
{
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int i = y * width + x;

  float sign_correction = ((x + y) & 0x01) ? -1.0f : 1.0f;

  heightMap[i] = ht[i].x * sign_correction;
}

__global__ void updateHeightmapKernel_y(float *heightMap, float2 *ht, unsigned int width)
{
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int i = y * width + x;

  float sign_correction = ((x + y) & 0x01) ? -1.0f : 1.0f;

  heightMap[i] = ht[i].y * sign_correction;
}

__global__ void calculateSlopeKernel(float *h, float2 *slopeOut, unsigned int width, unsigned int height)
{
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int i = y * width + x;

  float2 slope = make_float2(0.0f, 0.0f);

  if ((x > 0) && (y > 0) && (x < width - 1) && (y < height - 1))
  {
    slope.x = h[i + 1] - h[i - 1];
    slope.y = h[i + width] - h[i - width];
  }

  slopeOut[i] = slope;
}

// wrapper functions
void cudaGenerateSpectrumKernel(float2 *d_h0, float2 *d_ht, unsigned int in_width, unsigned int out_width, unsigned int out_height, float animTime, float patchSize)
{
  dim3 block(8, 8, 1);
  dim3 grid(cuda_iDivUp(out_width, block.x), cuda_iDivUp(out_height, block.y), 1);

  generateSpectrumKernel<<<grid, block>>>(d_h0, d_ht, in_width, out_width, out_height, animTime, patchSize);
}

void cudaUpdateHeightmapKernel(float *d_heightMap, float2 *d_ht, unsigned int width, unsigned int height)
{
  dim3 block(8, 8, 1);
  dim3 grid(cuda_iDivUp(width, block.x), cuda_iDivUp(height, block.y), 1);

  updateHeightmapKernel<<<grid, block>>>(d_heightMap, d_ht, width);
}

void cudaCalculateSlopeKernel(float *hptr, float2 *slopeOut, unsigned int width, unsigned int height)
{
  dim3 block(8, 8, 1);
  dim3 grid2(cuda_iDivUp(width, block.x), cuda_iDivUp(height, block.y), 1);

  calculateSlopeKernel<<<grid2, block>>>(hptr, slopeOut, width, height);
}

void generateSpectrumKernel_CPU(float2 *h0, float *heightMap, float2 *slopeOut, unsigned int in_width, unsigned int out_width, unsigned int out_height, float t, float patchSize)
{
	unsigned int x;
	unsigned int y;

	for (y = 0; y < out_height; y++)
	{
		for (x = 0; x < out_width; x++)
		{
			unsigned int in_index = y * in_width + x;
			unsigned int in_mindex = (out_height - y) * in_width + (out_width - x);
			unsigned int out_index = y * out_width + x;

			// calculate wave vector
			float2 k;
			k.x = (-(int)out_width / 2.0f + x) * (2.0f * CUDART_PI_F / patchSize);
			k.y = (-(int)out_width / 2.0f + y) * (2.0f * CUDART_PI_F / patchSize);

			// calculate dispersion w(k)
			float k_len = sqrtf(k.x * k.x + k.y * k.y);
			float w = sqrtf(9.81f * k_len);

			if ((x < out_width) && (y < out_height))
			{
				float2 h0_k = h0[in_index];
				float2 h0_mk = h0[in_mindex];

				h_ht_arr[out_index] = cpu_complex_add(cpu_complex_mult(h0_k, cpu_complex_exp(w * t)), cpu_complex_mult(cpu_conjugate(h0_mk), cpu_complex_exp(-w * t)));

				cudaResult = cudaMemcpy(d_ht, h_ht_arr, meshSize * meshSize * sizeof(float2), cudaMemcpyHostToDevice);
				// if (cudaResult != cudaSuccess)
				// 	fprintf(gpFile, "\ncudaMemcpy() Failed");

				cufftResult result;
				result = cufftExecC2C(fftPlan, d_ht, d_ht, CUFFT_INVERSE);
				// if (result != CUFFT_SUCCESS)
				// 	fprintf(gpFile, "\ncufftExecC2C() Failed");

				cudaResult = cudaMemcpy(h_ht_arr, d_ht, meshSize * meshSize * sizeof(float2), cudaMemcpyDeviceToHost);
				// if (cudaResult != cudaSuccess)
				// 	fprintf(gpFile, "\ncudaMemcpy() Failed");
					
				for (y = 0; y < out_height; y++)
				{
					for (x = 0; x < out_width; x++)
					{
						unsigned int out_index = y * out_width + x;

						float sign_correction = ((x + y) & 0x01) ? -1.0f : 1.0f;
						heightMap[out_index] = h_ht_arr[out_index].x * sign_correction;
					}
				}

				for (y = 0; y < (out_height - 1); y++)
				{
					for (x = 0; x < (out_width - 1); x++)
					{
						unsigned int out_index = y * out_width + x;

						if ((x > 0) && (y > 0) && (x < out_width - 1) && (y < out_height - 1))
						{
							slopeOut[out_index].x = heightMap[out_index + 1] - heightMap[out_index - 1];
							slopeOut[out_index].y = heightMap[out_index + out_width] - heightMap[out_index - out_width];
						}
					}
				}

			}
		}
	}
}
