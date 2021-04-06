/*
  This file is part of the TRON package (http://github.com/davidssmith/TRON).

  The MIT License (MIT)

  Copyright (c) 2016-2017 David Smith

  Permission is hereby granted, free of charge, to any person obtaining a # copy
  of this software and associated documentation files (the "Software"), to # deal
  in the Software without restriction, including without limitation the # rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or # sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included # in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS # OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL # THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING # FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS # IN THE
  SOFTWARE.
*/

#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <err.h>
#include <errno.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <time.h>
#include <stdint.h>
#include <assert.h>
#include <cufft.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include "cublas_v2.h"
#include <iostream>
#include <cuda/api_wrappers.hpp>
#include <thrust/complex.h>

#include "ra.h"
#include "tron.h"

#define MAX(a,b) ((a)>(b)?(a):(b))
#define MIN(a,b) ((a)<(b)?(a):(b))
#define DPRINT if(flags.verbose)printf
//#define dprint(expr,fmt)  do{ if(flags.verbose)fprintf(stderr,"\e[90m%d: " #expr " = %" #fmt "\e[0m\n", __LINE__, expr); }while(0);
#define dprint(expr,fmt)  do{ if(flags.verbose)fprintf(stderr,"%d: " #expr " = %" #fmt "\e[0m\n", __LINE__, expr); }while(0);

// MISC GLOBAL VARIABLES
static cufftHandle fft_plan_os;

static int threads = 128;    // TWEAK: CUDA kernel parameters, optimize for your device
static int blocks = 4096;


// DEFAULT RECON CONFIGURATION
static float gridos = 2.f;  // TODO: compute ngrid from nx, ny and oversamp
static float kernwidth = 2.f;
static float data_undersamp = 1.f;

static int prof_slide = 0;         // # of profiles to slide through the data between reconstructed images
static int skip_angles = 0;        // # of angles to skip at beginning of image stack
static int peoffset = 0;
static int niter = 0;

static int nro = 256;
static int npe1work = 256;
static int nx = 256;
static int nc = 30;
static int nt = 1;
static int ny = 256;
static int nxos = 512;
static int nyos = 512;
static struct {
    unsigned adjoint       : 1;
    unsigned deapodize     : 1;
    unsigned koosh         : 1;
    unsigned verbose       : 1;
    unsigned golden_angle  : 4;   // padded to 8 bits
} flags = {1, 1, 0, 1, 1};

// CONSTANTS
static const float PHI = 1.9416089796736116f;

inline void
gpuAssert (cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s in %s at L%d\n", cudaGetErrorString(code), file, line);
        if (abort) { getchar(); exit(code); }
    }
}
#define cuTry(ans) { gpuAssert((ans), __FILE__, __LINE__); }

static const char *
_cufftGetErrorEnum(cufftResult error)
{
    switch (error) {
        case CUFFT_SUCCESS: return "CUFFT_SUCCESS";
        case CUFFT_INVALID_PLAN: return "CUFFT_INVALID_PLAN";
        case CUFFT_ALLOC_FAILED: return "CUFFT_ALLOC_FAILED";
        case CUFFT_INVALID_TYPE: return "CUFFT_INVALID_TYPE";
        case CUFFT_INVALID_VALUE: return "CUFFT_INVALID_VALUE";
        case CUFFT_INTERNAL_ERROR: return "CUFFT_INTERNAL_ERROR";
        case CUFFT_EXEC_FAILED: return "CUFFT_EXEC_FAILED";
        case CUFFT_SETUP_FAILED: return "CUFFT_SETUP_FAILED";
        case CUFFT_INVALID_SIZE: return "CUFFT_INVALID_SIZE";
        case CUFFT_UNALIGNED_DATA: return "CUFFT_UNALIGNED_DATA";
        default: return "<unknown>";
    }
}

static const char *
_cublasGetErrorEnum (cublasStatus_t error)
{
    switch (error) {
        case CUBLAS_STATUS_SUCCESS: return "Success";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "Not Initiaziled";
        case CUBLAS_STATUS_ALLOC_FAILED: return "Alloc Failed";
        case CUBLAS_STATUS_INVALID_VALUE: return "Invalid Value";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "Arch Mismatch";
        case CUBLAS_STATUS_MAPPING_ERROR: return "Mapping Error";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "Exec Failed";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "Internal Error";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "Not Supported";
        case CUBLAS_STATUS_LICENSE_ERROR: return "License Error";
        default: return "<unknown>";
    }
}

#define cufftSafeCall(err)  __cufftSafeCall(err, __FILE__, __LINE__)
inline void __cufftSafeCall (cufftResult err, const char *file, const int line)
{
    if (CUFFT_SUCCESS != err) {
        fprintf(stderr, "CUFFT error in file '%s', line %d\nerror %s: %d\nterminating!\n",__FILE__, __LINE__, \
                _cufftGetErrorEnum(err), (int)err);
        cudaDeviceReset();
        exit(1);
    }
}

#define cublasSafeCall(err)  __cublasSafeCall(err, __FILE__, __LINE__)
inline void __cublasSafeCall (cublasStatus_t err, const char *file, const int line)
{
    if (CUBLAS_STATUS_SUCCESS != err) {
        fprintf(stderr, "CUBLAS error in file '%s', line %d\nerror %s: %d\nterminating!\n",__FILE__, __LINE__, \
                _cublasGetErrorEnum(err), (int)err);
        exit(1);
    }
}

enum fftshift_direction { FFT_SHIFT_FORWARD, FFT_SHIFT_INVERSE };

__global__ void
fftshift (thrust::complex<float> *dst, thrust::complex<float> *src, const int n, const int nchan, int direction)
{
    int offset = direction == FFT_SHIFT_FORWARD ? n/2 : n - n/2;

    for (int idsrc = blockIdx.x * blockDim.x + threadIdx.x; idsrc < n*n; idsrc += blockDim.x * gridDim.x)
    {
        int xsrc = idsrc / n;
        int ysrc = idsrc % n;
        int xdst = (xsrc + offset) % n;
        int ydst = (ysrc + offset) % n;
        int iddst = n*xdst + ydst;
        for (int c = 0; c < nchan; ++c) {
          dst[iddst*nchan + c]= src[idsrc*nchan + c];
        }
    }
}



__host__ void
fft_init(cufftHandle *plan, const int nx, const int ny, const int nchan)
{
  // setup FFT
  if (nchan == 1)
      cufftSafeCall(cufftPlan2d(plan, nx, ny, CUFFT_C2C));
  else {
      const int rank = 2;
      int idist = 1, odist = 1, istride = nchan, ostride = nchan;
      int n[2] = {nx, ny};
      int inembed[]  = {nx, ny};
      int onembed[]  = {nx, ny};
      cufftSafeCall(cufftPlanMany(plan, rank, n, onembed, ostride, odist,
          inembed, istride, idist, CUFFT_C2C, nchan));
  }
}


__global__ void
coilcombinesos (float *img, const thrust::complex<float> * __restrict__ coilimg, const int nimg, const int nchan) //used method!!
{
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < nimg*nimg; id += blockDim.x * gridDim.x) {
        if (nchan > 1) {
          float val = 0.f;
          for (int c = 0; c < nchan; ++c)
              val += thrust::norm(coilimg[nchan*id + c]);
          img[id] = sqrtf(val);
        } else
          img[id] = thrust::norm(coilimg[id]);
    }

}



__device__ float
besseli0 (const float x)
{
    if (x == 0.f) return 1.f;
    float z = x * x;
    float num = (z* (z* (z* (z* (z* (z* (z* (z* (z* (z* (z* (z* (z*
        (z* 0.210580722890567e-22  + 0.380715242345326e-19 ) +
        0.479440257548300e-16) + 0.435125971262668e-13 ) +
        0.300931127112960e-10) + 0.160224679395361e-7  ) +
        0.654858370096785e-5)  + 0.202591084143397e-2  ) +
        0.463076284721000e0)   + 0.754337328948189e2   ) +
        0.830792541809429e4)   + 0.571661130563785e6   ) +
        0.216415572361227e8)   + 0.356644482244025e9   ) +
        0.144048298227235e10);
    float den = (z*(z*(z-0.307646912682801e4)+
        0.347626332405882e7)-0.144048298227235e10);
    return -num/den;
}

__device__ inline float
kernel_shape (const float kernwidth, const float gridos)
{
//#define BEATTY_BETA
#ifdef BEATTY_BETA
    float a = kernwidth / gridos;
    float b = gridos - 0.5f;
    return M_PI*sqrtf(a*a*b*b - 0.8f);
#else
    //return M_PI*(2.f - 1.f/gridos);
    return 2.34f*2.0f*kernwidth; // beta = 4.68w
#endif
}


__device__ inline float
gridkernel (const float x, const float kernwidth, const float sigma)
{
  // x in [-kernwidth,kernwidth]
  float beta = kernel_shape(kernwidth, sigma);
  if (fabsf(x) < kernwidth) {
      float r = x/kernwidth;
      float f = sqrtf(1.0f - r*r);
      return 0.5f*besseli0(beta*f)/kernwidth; // gridding function
  } else
      return 0.0f;
}

__device__ inline float
gridkernelhat (const float u, const float kernwidth, const float sigma) //deapodization kernel \frac{sinc(\sqrt{(2\pi w x)^2 - \beta^2)})}{\sqrt{(2\pi w x)^2 - \beta^2)} where x \elem [-1/2, 1/2]
{
    // u in [-1/sigma,1/sigma]
    float J = 2.0f*kernwidth; //factor 2w // why no pi here?
    float beta = kernel_shape(kernwidth, sigma); //again find beta
    float r = M_PI*J*u; 
    float q = r*r - beta*beta; // whats under the square root
    float y, z;
    if (q > 0) {
        z = sqrtf(q);
        y = sinf(z) / z; //final result
    } else if (q < 0) {
        z = sqrtf(-q); // no imaginary calues
        y = sinhf(z) / z;
    } else
        y = 1;
    // identity: J_1/2(z) = sin(z) * sqrt(2/pi/z)
    return y;
}

__device__ inline float
modang (const float x)   /* rescale arbitrary angles to [0,2PI] interval */
{
    const float TWOPI = 2.f*M_PI;
    float y = fmodf(x, TWOPI);
    return y < 0.f ? y + TWOPI : y;
}

__device__ inline float
minangulardist(const float a, const float b)
{
    float d1 = fabsf(modang(a - b));
    float d2 = fabsf(modang(a + M_PI) - b);
    float d3 = 2.f*M_PI - d1;
    float d4 = 2.f*M_PI - d2;
    return fminf(fminf(d1,d2),fminf(d3,d4));
}

__global__ void
deapodkernel (thrust::complex<float>  *d_a, const int n, const int nrep, const float m, const float sigma)
{
    for (size_t id = blockIdx.x * blockDim.x + threadIdx.x; id < n*n; id += blockDim.x * gridDim.x)
    {
        float x = id / float(n) - (n + 1) / 2;  // TODO: simplify this
        float y = float(id % n) - (n + 1) / 2;
        float scale = 1.f / n / sigma;  // adding sigma was a hack to allow deapod after cropping
        float wgt = gridkernelhat(x*scale, m, sigma) * gridkernelhat(y*scale, m, sigma);
        for (int c = 0; c < nrep; ++c)
            d_a[nrep*id + c] /= (wgt > 0.f ? wgt : 1.f);
    }
}


__global__ void
precompensate (thrust::complex<float> *nudata, const int nchan, const int nro, const int npe1work)
{
    float a = (2.f  - 2.f / float(npe1work)) / float(nro);
    float b = 1.f / float(npe1work);
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < npe1work; id += blockDim.x * gridDim.x)
        for (int r = 0; r < nro; ++r) {
            float sdc = a*fabsf(r - float(nro/2)) + b;
            for (int c = 0; c < nchan; ++c)
                nudata[nro*nchan*id + nchan*r + c] *= sdc;
        }
}

__global__ void
crop (thrust::complex<float>* dst, const int nxdst, const int nydst, const thrust::complex<float>* __restrict__ src, const int nxsrc, const int nysrc, const int nchan)
{
    const int nsrc = nxsrc, ndst = nxdst;  // TODO: eliminate this
    const int w = (nsrc - ndst) / 2;
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < ndst*ndst; id += blockDim.x * gridDim.x)
    {
        int xdst = id / ndst;
        int ydst = id % ndst;
        int srcid = (xdst + w)*nsrc + ydst + w;
        for (int c = 0; c < nchan; ++c)
            dst[nchan*id + c] = src[nchan*srcid + c];
    }
}






extern "C" {  // don't mangle name, so can call from other languages

/*
    grid a single 2D image from input radial data
*/
__global__ void
gridradial2d (thrust::complex<float> *udata, const thrust::complex<float> * __restrict__ nudata, const int nxos,
    const int nchan, const int nro, const int npe, const float kernwidth, const float gridos,
const int skip_angles, const int flag_golden_angle)
{
    // udata: [NCHAN x NGRID x NGRID], nudata: NCHAN x NRO x NPE
    thrust::complex<float> utmp[MAXCHAN];
    const int blocksx = 4; // TODO: optimize this blocking
    const int blocksy = 4;
    const int warpsize = blocksx*blocksy;
    int nblockx = nxos / blocksx;
    int nblocky = nxos / blocksy; // # of blocks along y dimension

    for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < nxos * nxos; tid += blockDim.x * gridDim.x)
    {
        for (int ch = 0; ch < nchan; ch++){
            utmp[ch].real(0);
			utmp[ch].imag(0);
		}

        // figure out this thread's Cartesian and radial coordinates
        // int Y = id / nxos - nxos/2;
        // int X = (id % nxos) - nxos/2;

        // more complicated, but faster ... can probably optimize better by sorting, though
        int z = tid / warpsize; // not a real z, just a block label
        int by = z / nblocky;
        int bx = z % nblocky;
        int zid = tid % warpsize;
        int Y = zid / blocksy + blocksx*by;
        int X = zid % blocksy + blocksy*bx;
        int id = Y*nxos + X; // computed linear array index for uniform data
        X -= nxos/2;
        Y -= nxos/2;

        float R = hypotf(float(X), float(Y));

        // define a circular band around the uniform point
        int Rhi = fminf(floorf(R + kernwidth), nxos/2-1);
        int Rlo = fmaxf(ceilf(R - kernwidth), 0);

        // get uniform point coordinate in non-uniform system, (r,theta) in this case
        float T = modang(atan2f(float(Y),float(X)));   //[0,2pi]

        for (int pe = 0; pe < npe; ++pe)
        {
            float t = modang((pe + 1) * PHI) + M_PI/2;
            float st, ct;
            __sincosf(t, &st, &ct);
            for (int r = Rlo; r <= Rhi; ++r)  // aligned profiles
            {
                float kx = r*ct; // [-nxos/2 ... nxos/2-1]
                float ky = r*st; // [-nyos/2 ... nyos/2-1]
                float wgt = gridkernel(kx-X, kernwidth, gridos) * gridkernel(ky-Y, kernwidth, gridos);
                int ridx = (r * nro) / nxos;
                for (int ch = 0; ch < nchan && wgt > 0.f; ch++)
                    utmp[ch] += wgt*nudata[nchan*(nro*pe + ridx + nro/2) + ch];
            }
            for (int r = -Rhi; r <= -Rlo; ++r)  // anti-aligned profiles
            {
                float kx = r*ct; // [-nxos/2 ... nxos/2-1]
                float ky = r*st; // [-nyos/2 ... nyos/2-1]
                float wgt = gridkernel(kx-X, kernwidth, gridos) * gridkernel(ky-Y, kernwidth, gridos);
                int ridx = (r * nro) / nxos;
                for (int ch = 0; ch < nchan && wgt > 0.f; ch++)
                    utmp[ch] += wgt*nudata[nchan*(nro*pe + ridx + nro/2) + ch];
            }
        }
        //float scale_factor =  sqrtf(M_PI/2.f) / nxos / npe;
        float scale_factor =  1.f / nxos / npe;
        for (int ch = 0; ch < nchan; ++ch)
            udata[nchan*id + ch] = utmp[ch] * scale_factor;
    }
}



void
tron_init ()
{
    DPRINT("Kernels configured with %d blocks of %d threads\n", threads, blocks);
	std::cout << "nro: " << nro << std::endl;
	std::cout << "npe1work: " << npe1work << std::endl;
        fft_init(&fft_plan_os, nxos, nyos, nc);
	std::cout << "Done TRON Init " << std::endl;
}

void
tron_shutdown()
{
    DPRINT("freeing device memory ... ");
    DPRINT("done.\n");
}

// need to resort data for channels
void
tron_nufft_adj_radial2d (thrust::complex<float> *channel_images, thrust::complex<float> *domain_values, float *img, thrust::complex<float> *cropped_images, thrust::complex<float> *tmp_buffer, thrust::complex<float> *tmp_buffer_2)
{
	auto device = cuda::device::current::get();

    // NUFFT adjoint begin
    precompensate<<<blocks,threads>>>(domain_values, nc*nt, nro, npe1work);
    gridradial2d<<<blocks,threads>>>(channel_images, domain_values, nxos, nc*nt, nro, npe1work, kernwidth,
        gridos, skip_angles+peoffset, flags.golden_angle);
	device.synchronize();
	float sum_1 = 0;
	for (int i = 0; i < 30 * 512 * 512; ++i){
		sum_1 += thrust::norm(channel_images[i]);
	}
	float sum_2 = 0;
	for (int i = 0; i < 30 * 256 * 256; ++i){
		sum_2 += thrust::norm(domain_values[i]);
	}

	std::cout << "Sum Channel Images: " << sum_1 << std::endl;
	std::cout << "Sum Domain values: " << sum_2 << std::endl;
    fftshift<<<blocks,threads>>>(tmp_buffer, channel_images, nxos, nt*nc, FFT_SHIFT_INVERSE);
	std::cout << "Pre FFT" << std::endl;
    cufftSafeCall(cufftExecC2C(fft_plan_os, reinterpret_cast<cufftComplex*>(tmp_buffer), reinterpret_cast<cufftComplex*>(tmp_buffer), CUFFT_INVERSE));
	std::cout << "Post FFT" << std::endl;
    fftshift<<<blocks,threads>>>(channel_images, tmp_buffer, nxos, nc*nt, FFT_SHIFT_FORWARD);
    crop<<<blocks,threads>>>(cropped_images, nx, ny, channel_images, nxos, nyos, nc*nt);
    deapodkernel<<<blocks,threads>>>(cropped_images, nx, nc*nt, kernwidth, gridos);
	std::cout << "Done Deapod" << std::endl;
	coilcombinesos<<<blocks,threads>>>(img, cropped_images, nx, nc);
}



}


