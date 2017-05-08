/**
*  Header file for the pydh kernel, when used as standalone C library.
*
*  (C) Klaus Reuter, khr@rzg.mpg.de, 2015 - 2017
*
*  This file is part of the Cadishi package.  See README.rst,
*  LICENSE.txt, and the documentation for details.
*/

#ifndef _PYDH_H_
#define _PYDH_H_

#include "common.h"
#include <stdint.h>

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

#endif
