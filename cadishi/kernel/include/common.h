/**
* Cadishi --- CAlculation of DIStance HIstograms
*
* Copyright (c) Klaus Reuter, Juergen Koefinger
* See the file AUTHORS.rst for the full list of contributors.
*
* Released under the MIT License, see the file LICENSE.txt.
*
*
* Common code used by the pydh and cudh kernels, either as Python module or as C library.
*/


#ifndef _COMMON_H_
#define _COMMON_H_

// enum/integer values used to select the box type (also see common.py)
enum box_type {
    none,
    orthorhombic,
    triclinic
};

// double precision coordinate triple (used to access the NumPy data pointer passed by Python)
typedef struct np_tuple3d_t {
    double x;
    double y;
    double z;
} np_tuple3d_t;

// single precision coordinate triple
typedef struct np_tuple3s_t {
    float x;
    float y;
    float z;
} np_tuple3s_t;

#endif
