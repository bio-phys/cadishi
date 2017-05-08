/**
*  Code used by the c_cudh kernels for exception handling.
*
*  (C) Klaus Reuter, khr@rzg.mpg.de, 2015 - 2017
*
*  This file is part of the Cadishi package.  See README.rst,
*  LICENSE.txt, and the documentation for details.
*/

#ifndef _CUDA_EXCEPTIONS_HPP_
#define _CUDA_EXCEPTIONS_HPP_

#include <string>
#include <sstream>
#include <stdexcept>


// --- Debug/error macro handling CUDA calls.
// NOTE: Kernel errors can be traced using the call
//       CU_CHECK(cudaDeviceSynchronize());
//       directly after the kernel invocation.
#ifdef CUDA_DEBUG

#define CU_CHECK( CU_CALL ) \
{ \
  cudaError_t status = CU_CALL; \
  if (status != cudaSuccess) { \
    std::string msg; \
    msg += "CUDA error at "; \
    msg += std::string(__FILE__); \
    msg += ":"; \
    std::stringstream ss; \
    ss << __LINE__; \
    msg += ss.str(); \
    msg += ": "; \
    msg += std::string(cudaGetErrorString(status)); \
    throw std::runtime_error(msg); \
  } \
}

#else

// dummy version without check
#define CU_CHECK( CU_CALL ) \
{ \
  CU_CALL; \
}

#endif

#endif
