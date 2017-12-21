/**
* Cadishi --- CAlculation of DIStance HIstograms
*
* Copyright (c) Klaus Reuter, Juergen Koefinger
* See the file AUTHORS.rst for the full list of contributors.
*
* Released under the MIT License, see the file LICENSE.txt.
*
*
* Header file for the pydh kernel, when used as standalone C library.
*/

#ifndef _PYDH_H_
#define _PYDH_H_

#include "common.h"
#include "config.h"
#include <stdint.h>


int distances_cpu(np_tuple3d_t *r_ptr,
                  int n_tot,
                  double *distances,
                  double *box_ptr,
                  int box_type_id,
                  const config & cfg);

int histograms_cpu(np_tuple3d_t *r_ptr,
                   int n_tot,
                   int *nel_ptr,
                   int n_El,
                   uint64_t *histo_ptr,
                   int n_bins,
                   double r_max,
                   int *mask_ptr,
                   double *box_ptr,
                   int box_type_id,
                   const config & cfg);

/*
int histograms_cpu_single(np_tuple3s_t *r_ptr,
                          int n_tot,
                          int *nel_ptr,
                          int n_El,
                          uint64_t *histo_ptr,
                          int n_bins,
                          double r_max,
                          int *mask_ptr,
                          double *box_ptr,
                          bool check_input,
                          int box_type_id);

int histograms_cpu_double(np_tuple3d_t *r_ptr,
                          int n_tot,
                          int *nel_ptr,
                          int n_El,
                          uint64_t *histo_ptr,
                          int n_bins,
                          double r_max,
                          int *mask_ptr,
                          double *box_ptr,
                          bool check_input,
                          int box_type_id);
*/

#endif
