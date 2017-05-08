/**
*  cudh --- high performance GPU distance histogram code
*
*  (C) Klaus Reuter, khr@rzg.mpg.de, 2015 - 2017
*
*  This file is part of the Cadishi package.  See README.rst,
*  LICENSE.txt, and the documentation for details.
*/

#ifdef BUILD_C_LIBRARY
#include "cudh.h"
#else
#include <Python.h>
#include <numpy/ndarrayobject.h>
#endif

#include <stdint.h>
#include <cuda.h>

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <string>

#include "common.hpp"
#include "exceptions.hpp"
#include "cuda_exceptions.hpp"


enum _implementations {
   SIMPLE,
   TILED
};

const int n_box = 6;
enum _box_indices {
   idx_tri_0,
   idx_tri_1,
   idx_tri_2,
   idx_ortho,
   idx_inverse,
   idx_half
};

// --- parameters for the tiled kernels
// maximum number of bins fitting into the 48kB of shared memory available on relevant GPUs
const int smem_n_bins_max = 12288;

// GPU constant memory usage
const int histo_tiled_cmem_bytes = 64000; // max 64k
// static GPU constant memory
__constant__ char histo_tiled_coords_cmem[histo_tiled_cmem_bytes];

// --- parameters for the simple kernels
// cuda thread block size
const int histo_simple_block_x=8;
const int histo_simple_block_y=8;

// divisors to reduce the cuda grid below the n1 x n2 size,
// this number is identical to the work_per_thread
const int histo_simple_grid_x_div=50;
const int histo_simple_grid_y_div=50;


// --- helper function to print CUDA setup information
void
print_setup(const dim3 &grid, const dim3 &block, const int div_x, const int div_y, const int n1, const int n2) {
    if (n2 > 0) {
        printf("histo2_simple : n1=%d, n2=%d, div={%d,%d}, grid={%d,%d}, block={%d,%d}\n",
                                   n1,    n2, div_x,div_y, grid.x, grid.y, block.x, block.y);
    } else {
        printf("histo1_simple : n1=%d,        div={%d,%d}, grid={%d,%d}, block={%d,%d}\n",
                                   n1,        div_x,div_y, grid.x, grid.y, block.x, block.y);
    }
}


__device__ __forceinline__
uint32_t myAtomicAdd(uint32_t *address, uint32_t val) {
   return (uint32_t)atomicAdd((unsigned int *)address, (unsigned int)val);
}

__device__ __forceinline__
uint64_t myAtomicAdd(uint64_t *address, uint64_t val) {
   return (uint64_t)atomicAdd((unsigned long long int *)address, (unsigned long long int)val);
}

__device__ __forceinline__
uint64_t myAtomicAdd(uint64_t *address, uint32_t val) {
   uint64_t val_uint64 = (uint64_t)val;
   return myAtomicAdd(address, val_uint64);
}

__device__ __forceinline__
void increment(uint32_t *c) {
   myAtomicAdd(c, (uint32_t)1);
}

__device__ __forceinline__
void increment(uint64_t *c) {
   myAtomicAdd(c, (uint64_t)1);
}


// --- one-species simple histogram kernel (global memory, no caching)
template <typename TUPLE3_T, typename COUNTER_T, typename FLOAT_T, bool check_input, int box_type_id>
__global__ void
histo1_simple_knl(const TUPLE3_T* const r1, const int n1,
                  COUNTER_T *bins, const FLOAT_T scal,
                  const int n_bins,
                  COUNTER_T *err_flag,
                  const TUPLE3_T * const box)
{
   const int i = blockIdx.x*blockDim.x + threadIdx.x;
   const int j = blockIdx.y*blockDim.y + threadIdx.y;
   const int di = blockDim.x * gridDim.x;
   const int dj = blockDim.y * gridDim.y;

   TUPLE3_T box_ortho, box_inv, box_half;
   switch (box_type_id) {
      case orthorhombic:
         box_ortho = box[idx_ortho];
         box_inv = box[idx_inverse];
         break;
      case triclinic:
         box_half = box[idx_half];
         break;
   }

   for (int ii=i; ii<n1; ii+=di) {
      for (int jj=j; jj<ii; jj+=dj) {
         int idx = (int)(scal * dist<TUPLE3_T, FLOAT_T, box_type_id>(r1[jj], r1[ii], box, box_ortho, box_inv, box_half));
         // Error handling:
         // * If no box is used and check_input is requested, outliers do trigger the error condition.
         // * If a box is used and check_input is requested, outliers are simply ignored.
         // * If check_input is disabled, the memory location at idx is incremented wherever it may be located.
         if (check_input && (idx>=n_bins)) {
            if (box_type_id == none) {
               *err_flag = 1;
            }
         } else {
            increment( &bins[idx] );
         }
      }
   }
}

// --- two-species simple histogram kernel (global memory, no caching)
template <typename TUPLE3_T, typename COUNTER_T, typename FLOAT_T, bool check_input, int box_type_id>
__global__ void
histo2_simple_knl(const TUPLE3_T* const r1, const int n1,
                  const TUPLE3_T* const r2, const int n2,
                  COUNTER_T *bins,
                  const FLOAT_T scal,
                  const int n_bins,
                  COUNTER_T *err_flag,
                  const TUPLE3_T* const box)
{
   const int i = blockIdx.x*blockDim.x + threadIdx.x;
   const int j = blockIdx.y*blockDim.y + threadIdx.y;
   const int ni = blockDim.x * gridDim.x;
   const int nj = blockDim.y * gridDim.y;

   TUPLE3_T box_ortho, box_inv, box_half;
   switch (box_type_id) {
      case orthorhombic:
         box_ortho = box[idx_ortho];
         box_inv = box[idx_inverse];
         break;
      case triclinic:
         box_half = box[idx_half];
         break;
   }

   for (int ii=i; ii<n1; ii+=ni) {
      for (int jj=j; jj<n2; jj+=nj) {
         int idx = (int)(scal * dist<TUPLE3_T, FLOAT_T, box_type_id>(r2[jj], r1[ii], box, box_ortho, box_inv, box_half));
         // Error handling:
         // * If no box is used and check_input is requested, outliers do trigger the error condition.
         // * If a box is used and check_input is requested, outliers are simply ignored.
         // * If check_input is disabled, the memory location at idx is incremented wherever it may be located.
         if (check_input && (idx>=n_bins)) {
            if (box_type_id == none) {
               *err_flag = 1;
            }
         } else {
            increment( &bins[idx] );
         }
      }
   }
}

// --- driver (kernel) to launch the simple histogram kernels
template <typename TUPLE3_T, typename COUNTER_T, typename FLOAT_T, bool check_input, int box_type_id>
inline void
histo_simple_launch_knl(const TUPLE3_T* const r1, const int n1,
                        const TUPLE3_T* const r2, const int n2,
                        COUNTER_T *bins, const int n_bins,
                        const FLOAT_T scal,
                        COUNTER_T *err_flag,
                        const TUPLE3_T* const box)
{
   dim3 block;
   dim3 grid;

   block.x = histo_simple_block_x;
   block.y = histo_simple_block_y;
   grid.x  = (unsigned)ceil(double(n1)/double(block.x * histo_simple_grid_x_div));

   if (r1 != r2) {
      grid.y = (unsigned)ceil(double(n2)/double(block.y * histo_simple_grid_y_div));
      histo2_simple_knl <TUPLE3_T,COUNTER_T,FLOAT_T,check_input,box_type_id> <<<grid,block>>>
         (r1, n1, r2, n2, bins, scal, n_bins, err_flag, box);
   } else {
      grid.y = (unsigned)ceil(double(n1)/double(block.y * histo_simple_grid_y_div));
      histo1_simple_knl <TUPLE3_T,COUNTER_T,FLOAT_T,check_input,box_type_id> <<<grid,block>>>
         (r1, n1, bins, scal, n_bins, err_flag, box);
   }
}


// --- one-species histogram kernel, const- and shared- memory tiling
template <typename TUPLE3_T, typename COUNTER_T, typename FLOAT_T, bool check_input, int box_type_id>
__global__ void
histo1_tiled_knl(
   const TUPLE3_T* const r2, const int n2,
   COUNTER_T *bins, const int n_bins,
   const FLOAT_T scal,
   const int n1_tile_global_offset, const int n1_tile_size,
   const int histo_tiled_smem_nbins,
   COUNTER_T *err_flag,
   const TUPLE3_T* const box)
{
   TUPLE3_T *r1p;
   r1p = (TUPLE3_T*) histo_tiled_coords_cmem;

   TUPLE3_T box_ortho, box_inv, box_half;
   switch (box_type_id) {
      case orthorhombic:
         box_ortho = box[idx_ortho];
         box_inv = box[idx_inverse];
         break;
      case triclinic:
         box_half = box[idx_half];
         break;
   }

   extern __shared__ uint32_t smem_bins[];
   for (int i=threadIdx.x; i<histo_tiled_smem_nbins; i+=blockDim.x)
      smem_bins[i] = 0;
   __syncthreads();

   // --- global i2 index for r2
   const int i2 = blockIdx.x*blockDim.x + threadIdx.x;
   // --- local upper limit for the i1-loop
   const int q1 = i2 - n1_tile_global_offset;

   // --- map the binning range via the y grid
   const int bin_lo = histo_tiled_smem_nbins *  blockIdx.y;
   int n_bins_loc;  // n_bins local to the thread block
   if (/* bin_hi == */ histo_tiled_smem_nbins*(blockIdx.y+1) > n_bins) {
      n_bins_loc = n_bins - bin_lo;
   } else {
      n_bins_loc = histo_tiled_smem_nbins;
   }

   if (i2 < n2) {
      TUPLE3_T r2r = r2[i2];
      for (int i1=0; i1<n1_tile_size; ++i1) {
         if (i1 < q1) {
            int idx = (int)(scal * dist<TUPLE3_T, FLOAT_T, box_type_id>(r1p[i1], r2r, box, box_ortho, box_inv, box_half));
            // Error handling:
            // * If no box is used and check_input is requested, outliers do trigger the error condition.
            // * If a box is used and check_input is requested, outliers are simply ignored.
            // * If check_input is disabled, the memory location at idx is incremented wherever it may be located.
            if (check_input && (idx>=n_bins)) {
               if (box_type_id == none) {
                  *err_flag = 1;
               }
            } else {
               idx = idx - bin_lo;
               if ((idx >= 0) && (idx < n_bins_loc)) {
                  increment( &smem_bins[idx] );
               }
            }

         }
      }
   }
   __syncthreads();

   bins += bin_lo;
   for (int i=threadIdx.x; i<n_bins_loc; i+=blockDim.x)
      myAtomicAdd(&bins[i], smem_bins[i]);
}

// --- two-species histogram kernel, const- and shared- memory tiling
template <typename TUPLE3_T, typename COUNTER_T, typename FLOAT_T, bool check_input, int box_type_id>
__global__ void
histo2_tiled_knl(
   const TUPLE3_T* const r2, const int n2,
   COUNTER_T *bins, const int n_bins,
   const FLOAT_T scal,
   const int n1_tile_size,
   const int histo_tiled_smem_nbins,
   COUNTER_T *err_flag,
   const TUPLE3_T* const box)
{
   TUPLE3_T *r1p;
   r1p = (TUPLE3_T*) histo_tiled_coords_cmem;

   TUPLE3_T box_ortho, box_inv, box_half;
   switch (box_type_id) {
      case orthorhombic:
         box_ortho = box[idx_ortho];
         box_inv = box[idx_inverse];
         break;
      case triclinic:
         box_half = box[idx_half];
         break;
   }

   extern __shared__ uint32_t smem_bins[];
   for (int i=threadIdx.x; i<histo_tiled_smem_nbins; i+=blockDim.x)
      smem_bins[i] = 0;
   __syncthreads();

   const int i2 = blockIdx.x*blockDim.x + threadIdx.x;

   // --- map the range to be binned via the y grid
   const int bin_lo = histo_tiled_smem_nbins *  blockIdx.y;
   int n_bins_loc;
   if (/* bin_hi== */ histo_tiled_smem_nbins*(blockIdx.y+1) > n_bins) {
      n_bins_loc = n_bins - bin_lo;
   } else {
      n_bins_loc = histo_tiled_smem_nbins;
   }

   if (i2 < n2) {
      TUPLE3_T r2r = r2[i2];
      for (int i1=0; i1<n1_tile_size; ++i1) {
         int idx = (int)(scal * dist<TUPLE3_T, FLOAT_T, box_type_id>(r1p[i1], r2r, box, box_ortho, box_inv, box_half));
         // Error handling:
         // * If no box is used and check_input is requested, outliers do trigger the error condition.
         // * If a box is used and check_input is requested, outliers are simply ignored.
         // * If check_input is disabled, the memory location at idx is incremented wherever it may be located.
         if (check_input && (idx>=n_bins)) {
            if (box_type_id == none) {
               *err_flag = 1;
            }
         } else {
            idx = idx - bin_lo;
            if ((idx >= 0) && (idx < n_bins_loc)) {
               increment( &smem_bins[idx] );
            }
         }
      }
   }
   __syncthreads();

   bins += bin_lo;
   for (int i=threadIdx.x; i<n_bins_loc; i+=blockDim.x)
      myAtomicAdd( &bins[i], smem_bins[i] );
}



// --- compute histograms for a single frame
// throws std::runtime_error
template <typename TUPLE3_T, typename COUNTER_T, typename FLOAT_T, bool check_input, int box_type_id>
void histo_gpu(TUPLE3_T *coords, int n_tot,
               int *n_per_el, int n_el,
               COUNTER_T *bins, int n_bins, int n_Hij,
               FLOAT_T r_max, int *mask,
               int dev,
               /* --- optional arguments with defaults below --- */
               int histo_tiled_block_x_inp = 0,
               bool do_histo2_only = false,
               bool verbose = false)
{
   CU_CHECK( cudaSetDevice(dev) );
   cudaDeviceProp prop;
   CU_CHECK( cudaGetDeviceProperties(&prop, dev) );

   if (verbose) {
      printf("%s\n", SEP);
      printf("CUDH histo_gpu() running on device %d\n", dev);
      printf("input coordinate check : %s\n", (check_input ? "ON" : "OFF"));
      fflush(stdout);
      printf("%s\n", SEP);
   }

   // --- thread block size for the smem kernels
   int histo_tiled_block_x;
   // --- threshold value *below* which the tiled kernels should be used
   int histo_tiled_nbins_threshold;
   // --- set parameters depending on the compute capability based on simple performance measurements
   if (prop.major >= 5) {
      // MAXWELL
      histo_tiled_block_x = 384;
      histo_tiled_nbins_threshold = 4*smem_n_bins_max;
   } else {
      // earlier devices, tested with KEPLER
      histo_tiled_block_x = 960;
      histo_tiled_nbins_threshold = 2*smem_n_bins_max;
   }
   // --> for performance testing, the following values are useful:
//   histo_tiled_nbins_threshold = 0;  // always use the simple kernels
//   histo_tiled_nbins_threshold = (2 << 28);  // always use the tiled kernels

   if (histo_tiled_block_x_inp > 0) {
      printf("%s\n", SEP);
      if ((histo_tiled_block_x_inp % 32 == 0) && (histo_tiled_block_x_inp <= 1024)) {
         printf("GPU %d : input parameter : histo_tiled_block_x=%d\n", dev, histo_tiled_block_x_inp);
         histo_tiled_block_x = histo_tiled_block_x_inp;
      } else {
         printf("GPU %d : IGNORING input : histo_tiled_block_x\n", dev);
      }
      printf("%s\n", SEP);
      fflush(stdout);
   }

   const FLOAT_T scal = FLOAT_T(n_bins)/r_max;

   // --- device memory
   COUNTER_T *histo_d;
   TUPLE3_T *coord_d;
   TUPLE3_T *box_d;

   size_t coord_bytes = n_tot*sizeof(TUPLE3_T);
   CU_CHECK( cudaMalloc((void**)&coord_d, coord_bytes) );
   CU_CHECK( cudaMemcpy(coord_d, coords, coord_bytes, cudaMemcpyHostToDevice) );
   box_d = &coord_d[n_tot - n_box];

   // we add one more element to the histogram array to be used as an error flag
   size_t histo_bytes = n_bins*(n_Hij+1)*sizeof(COUNTER_T);
   const int idx_error_flag = n_bins*(n_Hij+1);
   CU_CHECK( cudaMalloc((void**)&histo_d, histo_bytes + sizeof(COUNTER_T)) );
   CU_CHECK( cudaMemset(histo_d, 0, histo_bytes + sizeof(COUNTER_T)) );

   int algorithm;
   if (n_bins > histo_tiled_nbins_threshold)
      algorithm = SIMPLE;
   else
      algorithm = TILED;

   switch (algorithm) {
      case SIMPLE:
      {  // curly brackets define a separate scope within the case block
         int histogramIdx = 0;
         int iOffset = 0;
         for (int i=0; i<n_el; ++i) {
            int jOffset = iOffset;
            // --- allow histo2 to be timed
            int j;
            if (do_histo2_only) {
               if (n_el != 2) {
                  RT_ERROR("Error: To time the histo2_*() routine, exactly two species must be used!");
               }
               j=i+1;
               jOffset += n_per_el[i];
            } else {
               j=i;
            }
            // ---
            for (/*int j=i*/; j<n_el; ++j) {
               ++histogramIdx;
               // ---
               if (mask[histogramIdx - 1] > 0) {
                  int histoOffset = histogramIdx*n_bins;
                  histo_simple_launch_knl <TUPLE3_T, COUNTER_T, FLOAT_T, check_input, box_type_id>
                        (&coord_d[iOffset], n_per_el[i],
                         &coord_d[jOffset], n_per_el[j],
                         &histo_d[histoOffset],
                         n_bins, scal,
                         &histo_d[idx_error_flag],
                         box_d);
                  CU_CHECK(cudaDeviceSynchronize());
               }
               // ---
               jOffset += n_per_el[j];
            }
            iOffset += n_per_el[i];
         }
      }
      break;

      case TILED:
      {
         dim3 block;
         dim3 grid;
         block.x = histo_tiled_block_x;
         block.y = 1;
         block.z = 1;
         grid.x  = 0; // grid parameter is set inside the loop
         grid.y  = 0; // grid parameter is set below based on smem
         grid.z  = 1;
         // --- number of atom coordinate tuples fitting into one constant memory tile
         const int cmem_tile_size = histo_tiled_cmem_bytes/sizeof(TUPLE3_T);
         // --- Set up number of shared memory tiles required to hold the full histogram,
         //     which is equivalent to the grid size in y direction.
         //     The goal is to minimize the number of tiles under the contraint
         //     of having the shared memory size as small as possible for best
         //     occupancy.
         //
         // calculate the number of tiles required for n_bins
         const int smem_n_tiles = (int)ceil(double(n_bins)/double(smem_n_bins_max));
         // calculate the size of a tile
         const int smem_tile_size = (int)ceil(double(n_bins)/double(smem_n_tiles));
         // calculate the tilesize in bytes
         const int smem_bytes = smem_tile_size*sizeof(uint32_t);
         // the loop over the histogram tiles is mapped via the Y-grid
         grid.y = (unsigned)smem_n_tiles;
         if (verbose) {
            printf("CUDA tiled kernel configuration:\n");
            printf("  block.x = %d\n", block.x);
            printf("  cmem_tile_size = %d\n", cmem_tile_size);
            printf("  smem_n_tiles = %d\n", smem_n_tiles);
            printf("  smem_tile_size = %d\n", smem_tile_size);
            printf("  smem_bytes = %d\n", smem_bytes);
         }
         // --- loop over species combinations
         int histogramIdx = 0;
         int iOffset = 0;
         for (int i=0; i<n_el; ++i) {
            // --- number of completely filled constant memory tiles
            //     for the species indexed by "i"
            const int cmem_n_tiles    = n_per_el[i] / cmem_tile_size;
            const int cmem_rem_n_elem = n_per_el[i] % cmem_tile_size;
            // --- copy of the outer iOffset value to restore it inside the i_tile loop
            int iOffset_0 = iOffset;
            // --- copy of the outer histogramIdx value to restore it inside the i_tile loop
            int histogramIdx_0 = histogramIdx;
            // --- loop over constant memory tiles,
            //     this introduces the complication of getting the indices inside right
            for (int i_tile = 0; i_tile <= cmem_n_tiles; ++i_tile) {
               int cmem_tile_n_elem;
               if (i_tile < cmem_n_tiles)
                  cmem_tile_n_elem = cmem_tile_size;
               else
                  cmem_tile_n_elem = cmem_rem_n_elem;
               const int cmem_tile_offset = i_tile*cmem_tile_size;
               // --- copy coordinate set to GPU constant memory
               CU_CHECK(
                  cudaMemcpyToSymbol(histo_tiled_coords_cmem, &coord_d[iOffset], cmem_tile_n_elem*sizeof(TUPLE3_T));
               );
               int jOffset = iOffset_0;
               histogramIdx = histogramIdx_0;
               // --- allow histo2 to be timed
               int j;
               if (do_histo2_only) {
                  if (n_el != 2) {
                    RT_ERROR("Error: To time the histo2_*() routine, exactly two species must be used!");
                  }
                  j=i+1;
                  jOffset += n_per_el[i];
               } else {
                  j=i;
               }
               // ---
               for (/*int j=i*/; j<n_el; ++j) {
                  ++histogramIdx;
                  if (mask[histogramIdx - 1] > 0) {
                     const int histoOffset = histogramIdx*n_bins;
                     grid.x  = (unsigned)ceil(double(n_per_el[j])/double(block.x));
                     if (i != j) {
                        histo2_tiled_knl <TUPLE3_T,COUNTER_T,FLOAT_T,check_input, box_type_id>
                           <<<grid,block,smem_bytes>>>
                              (&coord_d[jOffset], n_per_el[j],
                               &histo_d[histoOffset], n_bins,
                               scal,
                               cmem_tile_n_elem,
                               smem_tile_size,
                               &histo_d[idx_error_flag],
                               box_d);
                     } else {
                        histo1_tiled_knl <TUPLE3_T,COUNTER_T,FLOAT_T,check_input, box_type_id>
                           <<<grid,block,smem_bytes>>>
                              (&coord_d[jOffset], n_per_el[j],
                               &histo_d[histoOffset], n_bins,
                               scal,
                               cmem_tile_offset, cmem_tile_n_elem,
                               smem_tile_size,
                               &histo_d[idx_error_flag],
                               box_d);
                     }
                     CU_CHECK(cudaDeviceSynchronize());
                  }
                  jOffset += n_per_el[j];
               }
               iOffset += cmem_tile_n_elem;
            }
         }
      }
      break;

      default:
         RT_ERROR("unknown implementation requested");
   } // end switch

   // copy histograms and the error flag back to the host
   CU_CHECK( cudaMemcpy(bins, histo_d, histo_bytes + sizeof(COUNTER_T), cudaMemcpyDeviceToHost) );

   if (check_input && (bins[idx_error_flag] != COUNTER_T(0))) {
      OVERFLOW_ERROR(overflow_error_msg);
   }

   CU_CHECK( cudaFree(coord_d) );
   CU_CHECK( cudaFree(histo_d) );
}


template <typename NP_TUPLE3_T, typename TUPLE3_T, typename FLOAT_T>
void histograms_template_dispatcher(NP_TUPLE3_T *r_ptr,   // coordinate tuples
                                    int n_tot,            // total number of coordinate tuples
                                    int *nel_ptr,         // number of atoms per species
                                    int n_El,             // number of species
                                    int n_Hij,            // number of histograms
                                    uint64_t *histo_ptr,  // histograms
                                    int n_bins,           // histogram width
                                    double r_max,         // histogram cutoff
                                    int *mask_ptr,        // boolean mask specifying if nth histogram shall be computed
                                    double *box_ptr,      // periodic box specifier
                                    int box_type_id,      // type of periodic box
                                    int check_input,      // switch if distance should be checked before binning
                                    int gpu_id,           // id of the GPU to be used
                                    int thread_block_x,   // CUDA thread block size
                                    int do_histo2_only,
                                    int verbose) {
   TUPLE3_T * r_copy;
   TUPLE3_T * box_copy;
   // box information is forwarded using the last n_box elements of r_copy (even if no box is present at all)
   CU_CHECK( cudaMallocHost((void **)&r_copy, (n_tot+n_box)*sizeof(TUPLE3_T)) );
   for (int i=0; i<n_tot; ++i) {
      r_copy[i].x = FLOAT_T(r_ptr[i].x);
      r_copy[i].y = FLOAT_T(r_ptr[i].y);
      r_copy[i].z = FLOAT_T(r_ptr[i].z);
   }
   // pointer to the first element of the box data
   box_copy = &r_copy[n_tot];
   memset(box_copy, 0, n_box*sizeof(TUPLE3_T));
   switch (box_type_id) {
      case none:
         break;
      case orthorhombic:
         // "box_ortho"
         box_copy[idx_ortho].x = FLOAT_T(box_ptr[0]);  // concatenate box vectors
         box_copy[idx_ortho].y = FLOAT_T(box_ptr[4]);  // into a
         box_copy[idx_ortho].z = FLOAT_T(box_ptr[8]);  // single tuple
         // "box_inv"
         box_copy[idx_inverse].x = FLOAT_T(1.) / box_copy[idx_ortho].x;
         box_copy[idx_inverse].y = FLOAT_T(1.) / box_copy[idx_ortho].y;
         box_copy[idx_inverse].z = FLOAT_T(1.) / box_copy[idx_ortho].z;
         break;
      case triclinic:
         for (int i=0; i<3; ++i) {
            box_copy[i].x = FLOAT_T(box_ptr[3*i  ]);
            box_copy[i].y = FLOAT_T(box_ptr[3*i+1]);
            box_copy[i].z = FLOAT_T(box_ptr[3*i+2]);
         }
         // "box_inv"
         box_copy[idx_inverse].x = FLOAT_T(1.) / box_copy[0].x;
         box_copy[idx_inverse].y = FLOAT_T(1.) / box_copy[1].y;
         box_copy[idx_inverse].z = FLOAT_T(1.) / box_copy[2].z;
         // "box_half"
         box_copy[idx_half].x = FLOAT_T(0.5) * box_copy[0].x;
         box_copy[idx_half].y = FLOAT_T(0.5) * box_copy[1].y;
         box_copy[idx_half].z = FLOAT_T(0.5) * box_copy[2].z;
         // ---
#pragma omp parallel for default(shared) schedule(guided)
         for (int i=0; i<n_tot; ++i) {
            move_coordinates_into_triclinic_box<TUPLE3_T, FLOAT_T>(r_copy[i], box_copy, box_copy[idx_inverse]);
         }
         break;
   }
   // --- NOTE: n_tot is redefined below for all nested code ---
   n_tot += n_box;

   // --- create a local pinned copy of the histogram array
   uint64_t *histo_loc;
   size_t histo_bytes = n_bins*(n_Hij+1)*sizeof(uint64_t);  // raw size of the histograms
   // we allocate one more element at the end to be used internally as an error flag
   CU_CHECK( cudaMallocHost((void **)&histo_loc, histo_bytes + sizeof(uint64_t)) );

   // --- explicitly generate code for the kernels with/without coordinate input checks and with/without periodic boxes ---
   if (check_input) {
      switch (box_type_id) {
         case none:
            histo_gpu <TUPLE3_T, uint64_t, FLOAT_T, true, none>
               (r_copy, n_tot, nel_ptr, n_El, histo_loc, n_bins, n_Hij,
                FLOAT_T(r_max), mask_ptr,
                gpu_id, thread_block_x, do_histo2_only, (verbose != 0));
            break;
         case orthorhombic:
            histo_gpu <TUPLE3_T, uint64_t, FLOAT_T, true, orthorhombic>
               (r_copy, n_tot, nel_ptr, n_El, histo_loc, n_bins, n_Hij,
                FLOAT_T(r_max), mask_ptr,
                gpu_id, thread_block_x, do_histo2_only, (verbose != 0));
            break;
         case triclinic:
            histo_gpu <TUPLE3_T, uint64_t, FLOAT_T, true, triclinic>
               (r_copy, n_tot, nel_ptr, n_El, histo_loc, n_bins, n_Hij,
                FLOAT_T(r_max), mask_ptr,
                gpu_id, thread_block_x, do_histo2_only, (verbose != 0));
            break;
      }
   } else {
      switch (box_type_id) {
         case none:
            histo_gpu <TUPLE3_T, uint64_t, FLOAT_T, false, none>
               (r_copy, n_tot, nel_ptr, n_El, histo_loc, n_bins, n_Hij,
                FLOAT_T(r_max), mask_ptr,
                gpu_id, thread_block_x, do_histo2_only, (verbose != 0));
            break;
         case orthorhombic:
            histo_gpu <TUPLE3_T, uint64_t, FLOAT_T, false, orthorhombic>
               (r_copy, n_tot, nel_ptr, n_El, histo_loc, n_bins, n_Hij,
                FLOAT_T(r_max), mask_ptr,
                gpu_id, thread_block_x, do_histo2_only, (verbose != 0));
            break;
         case triclinic:
            histo_gpu <TUPLE3_T, uint64_t, FLOAT_T, false, triclinic>
               (r_copy, n_tot, nel_ptr, n_El, histo_loc, n_bins, n_Hij,
                FLOAT_T(r_max), mask_ptr,
                gpu_id, thread_block_x, do_histo2_only, (verbose != 0));
            break;
      }
   }
   // copy the result histograms back
   memcpy(histo_ptr, histo_loc, histo_bytes);
   CU_CHECK( cudaFreeHost(histo_loc) );
   CU_CHECK( cudaFreeHost(r_copy) );
}


#ifdef BUILD_C_LIBRARY

int histograms_gpu_single(np_tuple3s_t *r_ptr,  // coordinate tuples
                          int n_tot,            // total number of coordinate tuples
                          int *nel_ptr,         // number of atoms per species
                          int n_El,             // number of species
                          int n_Hij,            // number of histograms
                          uint64_t *histo_ptr,  // histograms
                          int n_bins,           // histogram width
                          double r_max,         // histogram cutoff
                          int *mask_ptr,        // boolean mask specifying if nth histogram shall be computed
                          double *box_ptr,      // periodic box specifier
                          int box_type_id,      // type of periodic box
                          int check_input,      // switch if distance should be checked before binning
                          int gpu_id,           // id of the GPU to be used
                          int thread_block_x,   // CUDA thread block size
                          int do_histo2_only,
                          int verbose) {
   int exit_status = 0;
   try {
      histograms_template_dispatcher <np_tuple3s_t, tuple3s_t, float>
         (r_ptr, n_tot,
          nel_ptr, n_El, n_Hij,
          histo_ptr, n_bins, r_max,
          mask_ptr,
          box_ptr, box_type_id,
          check_input,
          gpu_id, thread_block_x,
          do_histo2_only, verbose);
   }
   catch (std::overflow_error & err) {
      const std::string msg = std::string(err.what());
      printf("%s\n", msg.c_str());
      exit_status = 1;
   }
   catch (std::runtime_error & err) {
      const std::string msg = std::string(err.what());
      printf("%s\n", msg.c_str());
      exit_status = 2;
   }
   catch (...) {
      // --- general unknown error
      exit_status = 3;
   }
   return exit_status;
}

int histograms_gpu_double(np_tuple3d_t *r_ptr,  // coordinate tuples
                          int n_tot,            // total number of coordinate tuples
                          int *nel_ptr,         // number of atoms per species
                          int n_El,             // number of species
                          int n_Hij,            // number of histograms
                          uint64_t *histo_ptr,  // histograms
                          int n_bins,           // histogram width
                          double r_max,         // histogram cutoff
                          int *mask_ptr,        // boolean mask specifying if nth histogram shall be computed
                          double *box_ptr,      // periodic box specifier
                          int box_type_id,      // type of periodic box
                          int check_input,      // switch if distance should be checked before binning
                          int gpu_id,           // id of the GPU to be used
                          int thread_block_x,   // CUDA thread block size
                          int do_histo2_only,
                          int verbose) {
   int exit_status = 0;
   try {
        histograms_template_dispatcher <np_tuple3d_t, tuple3d_t, double>
           (r_ptr, n_tot,
            nel_ptr, n_El, n_Hij,
            histo_ptr, n_bins, r_max,
            mask_ptr,
            box_ptr, box_type_id,
            check_input,
            gpu_id, thread_block_x,
            do_histo2_only, verbose);
   }
   catch (std::overflow_error & err) {
      const std::string msg = std::string(err.what());
      printf("%s\n", msg.c_str());
      exit_status = 1;
   }
   catch (std::runtime_error & err) {
      const std::string msg = std::string(err.what());
      printf("%s\n", msg.c_str());
      exit_status = 2;
   }
   catch (...) {
      // --- general unknown error
      exit_status = 3;
   }
   return exit_status;
}

#else

// --- return the number of usable CUDA devices
static PyObject* get_num_devices(PyObject* self, PyObject* args)
{
   int n;
   if (cudaGetDeviceCount(&n) != cudaSuccess) {
      n = 0;
   }
   return Py_BuildValue("i", n);
}

// --- calculate distance histograms for a complete frame ---
static PyObject* histograms(PyObject* self, PyObject* args)
{
   // --- required parameters
   PyArrayObject *coords;
   PyArrayObject *nelems;
   PyArrayObject *histos;
   double r_max;
   PyArrayObject *mask;
   PyArrayObject *box;
   int box_type_id = none;
   // --- optional parameters
   int precision = single_precision;
   int gpu_id = 0;
   int do_histo2_only = 0;
   int thread_block_x = 0;
   int check_input = 0;
   int verbose = 0;
   int exit_status;

   exit_status = 0;
   try {
      if (!PyArg_ParseTuple(args, "OOOdOOi|iiiiii", &coords, &nelems, &histos, &r_max, &mask, &box, &box_type_id,
               /*optional parameters:*/ &precision, &gpu_id, &do_histo2_only,
                                        &thread_block_x, &check_input, &verbose))
         return NULL;

      // --- 2D double precision coordinate array for all species
      RT_ASSERT( coords->nd == 2 );
      RT_ASSERT( coords->dimensions[1] == 3 );
      int n_tot = coords->dimensions[0];
      np_tuple3d_t *r_ptr = (np_tuple3d_t*) coords->data;

      // --- 1D integer array containing the number of elements for each species
      RT_ASSERT( nelems->nd == 1 );
      int n_El = nelems->dimensions[0];
      int *nel_ptr = (int*) nelems->data;

      // --- 2D integer array containing the histograms
      RT_ASSERT( histos->nd == 2 );
      int n_bins = histos->dimensions[0];
      int n_Hij = (histos->dimensions[1])-1;
      uint64_t *histo_ptr = (uint64_t*) histos->data;

      RT_ASSERT(n_Hij == mask->dimensions[0]);
      int *mask_ptr = (int*) mask->data;

      RT_ASSERT(box->dimensions[0] == 3);
      RT_ASSERT(box->dimensions[1] == 3);
      double *box_ptr = (double*) box->data;

      if ( precision == single_precision ) {
        histograms_template_dispatcher <np_tuple3d_t, tuple3s_t, float>
           (r_ptr, n_tot,
            nel_ptr, n_El, n_Hij,
            histo_ptr, n_bins, r_max,
            mask_ptr,
            box_ptr, box_type_id,
            check_input,
            gpu_id, thread_block_x,
            do_histo2_only, verbose);
      } else if ( precision == double_precision ) {
         histograms_template_dispatcher <np_tuple3d_t, tuple3d_t, double>
            (r_ptr, n_tot,
             nel_ptr, n_El, n_Hij,
             histo_ptr, n_bins, r_max,
             mask_ptr,
             box_ptr, box_type_id,
             check_input,
             gpu_id, thread_block_x,
             do_histo2_only, verbose);
      } else {
         RT_ERROR(std::string("unknown precision identifier passed"));
      }
   }
   catch (std::overflow_error & err) {
      const std::string msg = std::string(err.what());
      printf("%s\n", msg.c_str());
      exit_status = 1;
   }
   catch (std::runtime_error & err) {
      const std::string msg = std::string(err.what());
      printf("%s\n", msg.c_str());
      exit_status = 2;
   }
   catch (...) {
      // --- general unknown error
      exit_status = 3;
   }

   return Py_BuildValue("i", exit_status);
}


// --- free any device memory allocated by previous calls
static PyObject* free(PyObject* self, PyObject* args)
{
   int exit_status;
   exit_status = 0;
   return Py_BuildValue("i", exit_status);
}


// --- register C functions as Python modules
static PyMethodDef c_cudh_Methods[] = {
   {"get_num_devices", get_num_devices, METH_VARARGS, "return the number of CUDA devices available on the system"},
   {"histograms", histograms, METH_VARARGS, "calculate distance histograms for a complete frame"},
   {"free", free, METH_VARARGS, "free CUDA memory allocated by histograms()"},
   {NULL, NULL, 0, NULL}
};


// --- Python module initialization
PyMODINIT_FUNC
initc_cudh(void)
{
   (void) Py_InitModule("c_cudh", c_cudh_Methods);
   // --- the following helper function must be called
   import_array();
}

#endif // BUILD_C_LIBRARY
