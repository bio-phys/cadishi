/**
* Cadishi --- CAlculation of DIStance HIstograms
*
* Copyright (c) Klaus Reuter, Juergen Koefinger, Max Linke
* See the file AUTHORS.rst for the full list of contributors.
*
* Released under the MIT License, see the file LICENSE.txt.
*
*
* Common header code used by the c_pydh and c_cudh kernels.
*/


#ifndef _CPP_COMMON_HPP_
#define _CPP_COMMON_HPP_

#include <cmath>
#include <stdint.h>
//#include <limits>
#include <float.h>
#include "common.h"

#if defined(__CUDACC__) // NVCC
#define ALIGN(n) __align__(n)
#define DEVICE __device__
#elif defined(__GNUC__) // GCC
#define ALIGN(n) __attribute__((aligned(n)))
#define DEVICE
// #elif defined(_MSC_VER) // MSVC
//   #define MY_ALIGN(n) __declspec(align(n))
#else
#error "please provide a compiler-compatible definition for the ALIGN macro"
#endif

// --- single precision coordinate data structure
typedef struct tuple3s_t {
    float x;
    float y;
    float z;
    float w;  // padding element
} tuple3s_t;

// --- double precision coordinate data structure
typedef struct tuple3d_t {
    double x;
    double y;
    double z;
    double w;  // padding element
} tuple3d_t;

// enum/integer values used to select the precision (also see common.py)
enum precision {
    single_precision,
    double_precision
};
const char id_single[] = "single";
const char id_double[] = "double";

const char SEP[] = "-----------------------------------------------------------------------------";


// Cadishi round wrapper functions for both the precisions
// Note: C++11 would provide std::round()
#pragma omp declare simd
DEVICE inline float cad_round(const float &val) {
    return roundf(val);
}
#pragma omp declare simd
DEVICE inline double cad_round(const double &val) {
    return round(val);
}

// Cadishi min wrapper functions for both the precisions
#pragma omp declare simd
DEVICE inline float cad_min(const float &v, const float &w) {
    return fminf(v, w);
}
#pragma omp declare simd
DEVICE inline double cad_min(const double &v, const double &w) {
    return fmin(v, w);
}

// The C++ default function std::numeric_limits<FLOAT_T>::max();
// cannot be run on the GPU so we use the custom functions below.
#pragma omp declare simd
DEVICE inline float float_t_maxval(float dummy) {
    return FLT_MAX;
}
#pragma omp declare simd
DEVICE inline double float_t_maxval(double dummy) {
    return DBL_MAX;
}



// minimum image convention for orthorhombic systems
#pragma omp declare simd
template <typename TUPLE3_T, typename FLOAT_T>
DEVICE inline void
mic_orthorhombic(TUPLE3_T &dp, const TUPLE3_T &box, const TUPLE3_T &box_inv) {
    TUPLE3_T t;
    t.x = box_inv.x * dp.x;
    t.y = box_inv.y * dp.y;
    t.z = box_inv.z * dp.z;

    dp.x = box.x * (t.x - cad_round(t.x));
    dp.y = box.y * (t.y - cad_round(t.y));
    dp.z = box.z * (t.z - cad_round(t.z));
}


// new attempt to implement a correct triclinic minimum image convention
// credit: Max Linke, pbc_distances
#pragma omp declare simd
template <typename TUPLE3_T, typename FLOAT_T>
DEVICE inline FLOAT_T
mic_triclinic(TUPLE3_T &dp, const TUPLE3_T * const box, const TUPLE3_T &box_inv) {
    FLOAT_T dsq_min = float_t_maxval(FLOAT_T());

/*
    dp.x = dp.x - box[2].x * cad_round(dp.z / box[2].z);
    dp.y = dp.y - box[2].y * cad_round(dp.z / box[2].z);
    dp.z = dp.z - box[2].z * cad_round(dp.z / box[2].z);

    dp.x = dp.x - box[1].x * cad_round(dp.y / box[1].y);
    dp.y = dp.y - box[1].y * cad_round(dp.y / box[1].y);

    dp.x = dp.x - box[0].x * cad_round(dp.x / box[0].x);
*/
    const FLOAT_T frac_z = cad_round(box_inv.z * dp.z);
    dp.x = dp.x - frac_z * box[2].x;
    dp.y = dp.y - frac_z * box[2].y;
    dp.z = dp.z - frac_z * box[2].z;

    const FLOAT_T frac_y = cad_round(box_inv.y * dp.y);
    dp.x = dp.x - frac_y * box[1].x;
    dp.y = dp.y - frac_y * box[1].y;

    const FLOAT_T frac_x = cad_round(box_inv.x * dp.x);
    dp.x = dp.x - frac_x * box[0].x;

    // search images to find the minimum distance
    for (int x = -1; x <= 1; ++x) {
        TUPLE3_T dpx = dp;
        dpx.x += box[0].x * static_cast<FLOAT_T>(x);
        for (int y = -1; y <= 1; ++y) {
            TUPLE3_T dpy = dpx;
            dpy.x += box[1].x * static_cast<FLOAT_T>(y);
            dpy.y += box[1].y * static_cast<FLOAT_T>(y);
            for (int z = -1; z <= 1; ++z) {
                TUPLE3_T dpz = dpy;
                dpz.x += box[2].x * static_cast<FLOAT_T>(z);
                dpz.y += box[2].y * static_cast<FLOAT_T>(z);
                dpz.z += box[2].z * static_cast<FLOAT_T>(z);

                FLOAT_T dsq = dpz.x * dpz.x + dpz.y * dpz.y + dpz.z * dpz.z;

                dsq_min = cad_min(dsq, dsq_min);
            }
        }
    }

    return dsq_min;
}


// distance calculation
#pragma omp declare simd
template <typename TUPLE3_T, typename FLOAT_T, int box_type_id>
DEVICE inline FLOAT_T
dist(const TUPLE3_T &p1, const TUPLE3_T &p2,
     const TUPLE3_T * const box,
     const TUPLE3_T &box_ortho,
     const TUPLE3_T &box_inv) {
    TUPLE3_T dp;
    FLOAT_T dsq;
    dp.x = p1.x - p2.x;
    dp.y = p1.y - p2.y;
    dp.z = p1.z - p2.z;
    switch (box_type_id) {
    case orthorhombic:
        mic_orthorhombic <TUPLE3_T, FLOAT_T> (dp, box_ortho, box_inv);
        dsq = dp.x * dp.x + dp.y * dp.y + dp.z * dp.z;
        break;
    case none:
        dsq = dp.x * dp.x + dp.y * dp.y + dp.z * dp.z;
        break;
    case triclinic:
        dsq = mic_triclinic<TUPLE3_T, FLOAT_T>(dp, box, box_inv);
        break;
    }
    return std::sqrt(dsq);
}


// UNUSED CODE BELOW, KEPT FOR DOCUMENTATION PURPOSES


// Functions for triclinic periodic box handling below, closely following Tuckerman.
// Warning: results are correct up to the half of the box length, then bogus!

// calculate the inverse (h^{-1}) of the triclinic box matrix
template <typename TUPLE3_T, typename FLOAT_T>
void
calclulate_inverse_triclinic_box(const TUPLE3_T * const box,
                                 TUPLE3_T * box_tri_inv) {
    FLOAT_T x0, x1, x2;
    FLOAT_T y1, y2;
    FLOAT_T z2;
    x0 = box[0].x;
    x1 = box[1].x;
    x2 = box[2].x;
    y1 = box[1].y;
    y2 = box[2].y;
    z2 = box[2].z;
    box_tri_inv[0].x = FLOAT_T(1.)/x0;
    box_tri_inv[0].y = FLOAT_T(-1.) * x1 / (x0*y1);
    box_tri_inv[0].z = (x1*y2 - x2*y1)/(x0 * y1 * z2);
    box_tri_inv[1].x = FLOAT_T(0.);
    box_tri_inv[1].y = FLOAT_T(1.)/y1;
    box_tri_inv[1].z = FLOAT_T(-1.) * y2 / (y1 * z2);
    box_tri_inv[2].x = FLOAT_T(0.);
    box_tri_inv[2].y = FLOAT_T(0.);
    box_tri_inv[2].z = FLOAT_T(1.)/z2;
}

// transform a cartesian coordinate tuple to triclic coordinates
template <typename TUPLE3_T, typename FLOAT_T>
inline void
transform_to_triclinic_coordinates(TUPLE3_T &p, const TUPLE3_T * const box_tri_inv) {
    // We are allowed to overwrite the p components like that because of
    // zero-valued entries in box_tri_inv.
    p.x = box_tri_inv[0].x*p.x + box_tri_inv[0].y*p.y + box_tri_inv[0].z*p.z;
    p.y =                        box_tri_inv[1].y*p.y + box_tri_inv[1].z*p.z;
    p.z =                                               box_tri_inv[2].z*p.z;
}

// transform a triclinic coordinate tuple to cartesian coordinates
template <typename TUPLE3_T, typename FLOAT_T>
DEVICE inline void
transform_to_cartesian_coordinates(TUPLE3_T &p, const TUPLE3_T * const box) {
    // Again, we are allowed to overwrite the p components like that because of
    // zero-valued entries in the box.
    p.x = box[0].x*p.x + box[1].x*p.y + box[2].x*p.z;
    p.y =                box[1].y*p.y + box[2].y*p.z;
    p.z =                               box[2].z*p.z;
}

// apply the triclinic mic to a coordinate tuple
template <typename TUPLE3_T, typename FLOAT_T>
DEVICE inline void
triclinic_minimum_image_convention(TUPLE3_T &p) {
    p.x = p.x - cad_round(p.x);
    p.y = p.y - cad_round(p.y);
    p.z = p.z - cad_round(p.z);
}

// --- end of Tuckerman functions ---

#endif
