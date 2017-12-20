/**
* Cadishi --- CAlculation of DIStance HIstograms
*
* Copyright (c) Klaus Reuter, Juergen Koefinger
* See the file AUTHORS.rst for the full list of contributors.
*
* Released under the MIT License, see the file LICENSE.txt.
*
*
* Header file for the cudh kernel, when used as standalone C library.
*/

#ifndef _CUDH_H_
#define _CUDH_H_

#include "common.h"
#include "config.h"
#include <stdint.h>


int get_num_cuda_devices();

int histograms_gpu(np_tuple3d_t *r_ptr,
                   int n_tot,
                   int *nel_ptr,
                   int n_El,
                   uint64_t *histo_ptr,
                   int n_bins,
                   int n_Hij,
                   double r_max,
                   int *mask_ptr,
                   double *box_ptr,
                   int box_type_id,
                   const config & cfg);

/*
int histograms_gpu_single(np_tuple3s_t *r_ptr,  // coordinate tuples
                          int n_tot,            // total number of coordinate tuples
                          int *nel_ptr,         // number of atoms per species
                          int n_El,             // number of species
                          int n_Hij,            // number of histograms
                          uint64_t *histo_loc,  // histograms
                          int n_bins,           // histogram width
                          double r_max,         // histogram cutoff
                          int *mask_ptr,        // boolean mask specifying if nth histogram shall be computed
                          double *box_ptr,      // periodic box specifier
                          int box_type_id,      // type of periodic box
                          int check_input,      // switch if distance should be checked before binning
                          int gpu_id,           // id of the GPU to be used
                          int thread_block_x,   // CUDA thread block size
                          int do_histo2_only,
                          int verbose);

int histograms_gpu_double(np_tuple3d_t *r_ptr,  // coordinate tuples
                          int n_tot,            // total number of coordinate tuples
                          int *nel_ptr,         // number of atoms per species
                          int n_El,             // number of species
                          int n_Hij,            // number of histograms
                          uint64_t *histo_loc,  // histograms
                          int n_bins,           // histogram width
                          double r_max,         // histogram cutoff
                          int *mask_ptr,        // boolean mask specifying if nth histogram shall be computed
                          double *box_ptr,      // periodic box specifier
                          int box_type_id,      // type of periodic box
                          int check_input,      // switch if distance should be checked before binning
                          int gpu_id,           // id of the GPU to be used
                          int thread_block_x,   // CUDA thread block size
                          int do_histo2_only,
                          int verbose);
*/

#endif
