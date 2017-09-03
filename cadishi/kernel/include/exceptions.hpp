/**
* Cadishi --- CAlculation of DIStance HIstograms
*
* Copyright (c) Klaus Reuter, Juergen Koefinger
* See the file AUTHORS.rst for the full list of contributors.
*
* Released under the MIT License, see the file LICENSE.txt.
*
*
* Common code used by the c_pydh and c_cudh kernels for exception handling.
*/

#ifndef _CPP_EXCEPTIONS_HPP_
#define _CPP_EXCEPTIONS_HPP_

#include <string>
#include <sstream>
#include <stdexcept>

const std::string overflow_error_msg("at least one pair distance exceeded r_max; please check n_bins, r_max, and the coordinates");

#define OVERFLOW_ERROR( MSG ) \
{ \
  std::string msg; \
  msg += "Error at "; \
  msg += std::string(__FILE__); \
  msg += ":"; \
  std::stringstream ss; \
  ss << __LINE__; \
  msg += ss.str(); \
  msg += ": "; \
  msg += std::string(MSG); \
  throw std::overflow_error(msg); \
}

#define RT_ERROR( MSG ) \
{ \
  std::string msg; \
  msg += "Error at "; \
  msg += std::string(__FILE__); \
  msg += ":"; \
  std::stringstream ss; \
  ss << __LINE__; \
  msg += ss.str(); \
  msg += ": "; \
  msg += std::string(MSG); \
  throw std::runtime_error(msg); \
}

#define RT_ASSERT( COND ) \
{ \
  if (!( COND )) \
  { \
    std::string msg; \
    msg += "Error at "; \
    msg += std::string(__FILE__); \
    msg += ":"; \
    std::stringstream ss; \
    ss << __LINE__; \
    msg += ss.str(); \
    msg += ": "; \
    msg += "assertion violated."; \
    throw std::runtime_error(msg); \
  } \
}

#endif
