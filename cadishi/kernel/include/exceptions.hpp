/**
*  Common code used by the c_pydh and c_cudh kernels for exception handling.
*
*  (C) Klaus Reuter, khr@rzg.mpg.de, 2015 - 2017
*
*  This file is part of the Cadishi package.  See README.rst,
*  LICENSE.txt, and the documentation for details.
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
