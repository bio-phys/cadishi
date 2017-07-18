/**
*  Common header code used by the c_pydh and c_cudh kernels.
*
*  (C) Klaus Reuter, khr@rzg.mpg.de, 2015 - 2017
*
*  This file is part of the Cadishi package.  See README.rst,
*  LICENSE.txt, and the documentation for details.
*/


#ifndef _CPP_COMMON_HPP_
#define _CPP_COMMON_HPP_

#include <cmath>
#include <stdint.h>
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


// round wrapper functions for both the precisions
DEVICE inline float round_wrapper(const float &val) {
   return roundf(val);
}
DEVICE inline double round_wrapper(const double &val) {
   return round(val);
}


// --- reimplementation of basic functions for triclinic periodic box handling,
// closely following Tuckerman ---

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
  p.x = p.x - round_wrapper(p.x);
  p.y = p.y - round_wrapper(p.y);
  p.z = p.z - round_wrapper(p.z);
}


// minimum image convention for orthorhombic systems
template <typename TUPLE3_T, typename FLOAT_T>
DEVICE inline void
mic_orthorhombic(TUPLE3_T &dp, const TUPLE3_T &box, const TUPLE3_T &box_inv) {
   TUPLE3_T t;
   t.x = box_inv.x * dp.x;
   t.y = box_inv.y * dp.y;
   t.z = box_inv.z * dp.z;

   // Note: C++11 would provide std::round()
   dp.x = box.x * (t.x - round_wrapper(t.x));
   dp.y = box.y * (t.y - round_wrapper(t.y));
   dp.z = box.z * (t.z - round_wrapper(t.z));
}


// distance calculation
template <typename TUPLE3_T, typename FLOAT_T, int box_type_id>
DEVICE inline FLOAT_T
dist(const TUPLE3_T &p1, const TUPLE3_T &p2,
     const TUPLE3_T * const box,
     const TUPLE3_T &box_ortho,
     const TUPLE3_T &box_ortho_inv) {
   TUPLE3_T dp;
   dp.x = p1.x - p2.x;
   dp.y = p1.y - p2.y;
   dp.z = p1.z - p2.z;
   switch (box_type_id) {
      case orthorhombic:
         mic_orthorhombic <TUPLE3_T, FLOAT_T> (dp, box_ortho, box_ortho_inv);
         break;
      case triclinic:
         triclinic_minimum_image_convention<TUPLE3_T, FLOAT_T>(dp);
         transform_to_cartesian_coordinates<TUPLE3_T, FLOAT_T>(dp, box);
         break;
      case none:
         break;
   }
   FLOAT_T arg = dp.x * dp.x + dp.y * dp.y + dp.z * dp.z;
   return std::sqrt(arg);
}

#endif
