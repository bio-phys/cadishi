/**
*  Header file for the cudh kernel, when used as standalone C library.
*
*  (C) Klaus Reuter, khr@rzg.mpg.de, 2015 - 2017
*
*  This file is part of the Cadishi package.  See README.rst,
*  LICENSE.txt, and the documentation for details.
*/

#ifndef _CUDH_H_
#define _CUDH_H_

#include "common.h"
#include <stdint.h>

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

#endif
