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


// --- round wrapper functions for both the precisions ---
DEVICE inline float my_round(const float &val) {
   return roundf(val);
}
DEVICE inline double my_round(const double &val) {
   return round(val);
}

// --- generic branch-free signum functions ---
// adapted from
// https://stackoverflow.com/questions/1903954/is-there-a-standard-sign-function-signum-sgn-in-c-c
// NOTE: a little slower than the bitmask-based functions below
template <typename T>
DEVICE inline T sgn(const T &val) {
   return T((T(0) < val) - (val < T(0)));
}
template <typename T>
DEVICE inline T neg_sgn(const T &val) {
   return T((val < T(0) - (T(0) < val)));
}

// --- bitmask-based signum functions ---
DEVICE double double_one = 1.0;
DEVICE const uint64_t double_sign_bitmask = uint64_t(1) << 63;
//const uint64_t double_one_bits = reinterpret_cast<uint64_t&>(double_one);
DEVICE const uint64_t double_one_bits = 4607182418800017408;
DEVICE float float_one = 1.0f;
DEVICE const uint32_t float_sign_bitmask = uint32_t(1) << 31;
//const uint32_t float_one_bits = reinterpret_cast<uint32_t&>(float_one);
DEVICE const uint32_t float_one_bits = 1065353216;
// return signum of d as -1.0 or +1.0
DEVICE inline double signum(const double &d) {
   // obtain the sign bit
   uint64_t bits = reinterpret_cast<const uint64_t&>(d) & double_sign_bitmask;
   // apply sign to floating point one
   uint64_t sign = bits ^ double_one_bits;
   return reinterpret_cast<double&>(sign);
}
// return signum of d as -1.0 or +1.0
DEVICE inline float signum(const float &f) {
   uint32_t bits = reinterpret_cast<const uint32_t&>(f) & float_sign_bitmask;
   uint32_t sign = bits ^ float_one_bits;
   return reinterpret_cast<float&>(sign);
}
// return the negative signum of d as -1.0 or +1.0
DEVICE inline double negative_signum(const double &d) {
   uint64_t bits = (reinterpret_cast<const uint64_t&>(d) & double_sign_bitmask) ^ double_sign_bitmask;
   uint64_t sign = bits ^ double_one_bits;
   return reinterpret_cast<double&>(sign);
}
// return the negative signum of d as -1.0 or +1.0
DEVICE inline float negative_signum(const float &f) {
   uint32_t bits = (reinterpret_cast<const uint32_t&>(f) & float_sign_bitmask) ^ float_sign_bitmask;
   uint32_t sign = bits ^ float_one_bits;
   return reinterpret_cast<float&>(sign);
}


// compute triclinic shift factor, branch-free version (see mic_triclinic() below)
template <typename T>
DEVICE inline T get_mic_triclinic_factor(const T &dp, const T &box_half) {
   return T(std::fabs(dp) > box_half) * negative_signum(dp);
}


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
  box_tri_inv[0].x = FLOAT_T(0.);
  box_tri_inv[0].y = FLOAT_T(0.);
  box_tri_inv[0].z = FLOAT_T(1.)/z2;
}


template <typename TUPLE3_T, typename FLOAT_T>
inline void
transform_to_scaled_coordinates(TUPLE3_T &p, const TUPLE3_T * const box_tri_inv) {
  // We are allowed to overwrite the p components like that because of
  // zero-valued entries in box_tri_inv.
  p.x = box_tri_inv[0].x*p.x + box_tri_inv[0].y*p.y + box_tri_inv[0].z*p.z;
  p.y =                        box_tri_inv[1].y*p.y + box_tri_inv[1].z*p.z;
  p.z =                                               box_tri_inv[2].z*p.z;
}


template <typename TUPLE3_T, typename FLOAT_T>
inline void
transform_to_cartesian_coordinates(TUPLE3_T &p, const TUPLE3_T * const box) {
  // Again, we are allowed to overwrite the p components like that because of
  // zero-valued entries in box.
  // TODO: check correctness
  p.x = box[0].x*p.x + box[1].x*p.y + box[2].x*p.z;
  p.y =                box[1].y*p.y + box[2].y*p.z;
  p.z =                               box[2].z*p.z;
}


template <typename TUPLE3_T, typename FLOAT_T>
inline void
triclinic_minimum_image_convention(TUPLE3_T &p) {
  p.x = p.x - my_round(p.x);
  p.y = p.y - my_round(p.y);
  p.z = p.z - my_round(p.z);
}


// --- moves coordinates into triclinic cell (host function) ---
template <typename TUPLE3_T, typename FLOAT_T>
inline void
move_coordinates_into_triclinic_box(TUPLE3_T &p,
                                    const TUPLE3_T * const box,
                                    const TUPLE3_T &box_inv) {
   FLOAT_T f;
   f = -std::floor(p.z * box_inv.z);
   p.x += f * box[2].x;
   p.y += f * box[2].y;
   p.z += f * box[2].z;
   f = -std::floor(p.y * box_inv.y);
   p.x += f * box[1].x;
   p.y += f * box[1].y;
   f = -std::floor(p.x * box_inv.x);
   p.x += f * box[0].x;
}


// --- apply minimum image convention for orthorhombic systems
template <typename TUPLE3_T, typename FLOAT_T>
DEVICE inline void
mic_orthorhombic(TUPLE3_T &dp, const TUPLE3_T &box, const TUPLE3_T &box_inv) {
   TUPLE3_T t;
   t.x = box_inv.x * dp.x;
   t.y = box_inv.y * dp.y;
   t.z = box_inv.z * dp.z;

   // Note: C++11 would provide std::round()
   dp.x = box.x * (t.x - my_round(t.x));
   dp.y = box.y * (t.y - my_round(t.y));
   dp.z = box.z * (t.z - my_round(t.z));
}


// --- apply minimum image convention for triclinic systems
template <typename TUPLE3_T, typename FLOAT_T>
DEVICE inline void
mic_triclinic(TUPLE3_T &dp, const TUPLE3_T * const box, const TUPLE3_T &box_half) {
   // branch-free code
   FLOAT_T f;
   f = get_mic_triclinic_factor(dp.z, box_half.z);
   dp.x += f * box[2].x;
   dp.y += f * box[2].y;
   dp.z += f * box[2].z;
   f = get_mic_triclinic_factor(dp.y, box_half.y);
   dp.x += f * box[1].x;
   dp.y += f * box[1].y;
   f = get_mic_triclinic_factor(dp.x, box_half.x);
   dp.x += f * box[0].x;

   // --- previous naive code (performs much worse on the CPU) ---
//   if (std::fabs(dp.z) > box_half.z) {
//      FLOAT_T f = negative_signum(dp.z);
//      dp.x += f * box[2].x;
//      dp.y += f * box[2].y;
//      dp.z += f * box[2].z;
//   }
//   if (std::fabs(dp.y) > box_half.y) {
//      FLOAT_T f = negative_signum(dp.y);
//      dp.x += f * box[1].x;
//      dp.y += f * box[1].y;
//   }
//   if (std::fabs(dp.x) > box_half.x) {
//      FLOAT_T f = negative_signum(dp.x);
//      dp.x += f * box[0].x;
//   }
}


// --- distance calculation, templated version
template <typename TUPLE3_T, typename FLOAT_T, int box_type_id>
DEVICE inline FLOAT_T
dist(const TUPLE3_T &p1, const TUPLE3_T &p2,
     const TUPLE3_T * const box, const TUPLE3_T &box_ortho, const TUPLE3_T &box_inv, const TUPLE3_T &box_half) {
   TUPLE3_T dp;
   dp.x = p1.x - p2.x;
   dp.y = p1.y - p2.y;
   dp.z = p1.z - p2.z;
   switch (box_type_id) {
      case orthorhombic:
         mic_orthorhombic <TUPLE3_T, FLOAT_T> (dp, box_ortho, box_inv);
         break;
      case triclinic:
         mic_triclinic <TUPLE3_T, FLOAT_T> (dp, box, box_half);
         break;
      case none:
         break;
   }
   FLOAT_T arg = dp.x * dp.x + dp.y * dp.y + dp.z * dp.z;
   return std::sqrt(arg);
}


// --- distance calculation, templated version
template <typename TUPLE3_T, typename FLOAT_T, int box_type_id>
DEVICE inline FLOAT_T
dist_new(const TUPLE3_T &p1, const TUPLE3_T &p2,
     const TUPLE3_T * const box, const TUPLE3_T &box_ortho,
     const TUPLE3_T &box_inv, const TUPLE3_T * const box_tri_inv) {
   TUPLE3_T dp;
   dp.x = p1.x - p2.x;
   dp.y = p1.y - p2.y;
   dp.z = p1.z - p2.z;
   switch (box_type_id) {
      case orthorhombic:
         mic_orthorhombic <TUPLE3_T, FLOAT_T> (dp, box_ortho, box_inv);
         break;
      case triclinic:
         triclinic_minimum_image_convention<TUPLE3_T, FLOAT_T>(dp);
         transform_to_cartesian_coordinates<TUPLE3_T, FLOAT_T>(dp, box_tri_inv);
         break;
      case none:
         break;
   }
   FLOAT_T arg = dp.x * dp.x + dp.y * dp.y + dp.z * dp.z;
   return std::sqrt(arg);
}


#endif
