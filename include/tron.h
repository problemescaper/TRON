#ifndef _TRON_H
#define _TRON_H

/*
  This file is part of the TRON package (http://github.com/davidssmith/tron).

  The MIT License (MIT)

  Copyright (c) 2016 David Smith

  Permission is hereby granted, free of charge, to any person obtaining a # copy
  of this software and associated documentation files (the "Software"), to #
  deal in the Software without restriction, including without limitation the #
  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or #
  sell copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included # in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS # OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL # THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING #
  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS #
  IN THE SOFTWARE.
*/

#include <complex.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <err.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "ra.h"

// CONFIGURATION PARAMETERS
// TODO: softcode as many as possible
#define NCHAN 30
#define MAXCHAN 30
// static int threads = 96;    // TWEAK: CUDA kernel parameters, optimize for
// your device static int blocks = 1024;

extern "C" {  // don't mangle name, so can call from other languages

/* grid a single 2D image from input radial data */
__global__ void gridradial2d(thrust::complex<float> *udata,
			     const thrust::complex<float> *__restrict__ nudata,
			     const int ngrid, const int nchan, const int nro,
			     const int npe, const float kernwidth,
			     const float grid_oversamp, const int skip_angles,
			     const int flag_golden_angle);

/*  generate 2D radial data from an input 2D image */
__global__ void degridradial2d(thrust::complex<float> *nudata,
			       const thrust::complex<float> *__restrict__ udata,
			       const int nimg, const int nchan, const int nro,
			       const int npe, const float kernwidth,
			       const float gridos, const int skip_angles,
			       const int flag_golden_angle);

/*  Reconstruct images from 2D radial data.  This host routine calls the
   appropriate CUDA kernels in the correct order depending on the direction of
   recon.   */
__host__ void recon_radial_2d(
    thrust::complex<float> *h_outdata,
    const thrust::complex<float> *__restrict__ h_indata);

void tron_nufft_adj_radial2d(thrust::complex<float> *d_out,
			     thrust::complex<float> *d_in, float *img,
			     thrust::complex<float> *cropped_images,
			     thrust::complex<float> *tmp_buffer, int res_image,
			     int os);

void tron_init(int res_kspace);
}
#endif /* _TRON_H */
