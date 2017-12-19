/**
* Cadishi --- CAlculation of DIStance HIstograms
*
* Copyright (c) Klaus Reuter, Juergen Koefinger
* See the file AUTHORS.rst for the full list of contributors.
*
* Released under the MIT License, see the file LICENSE.txt.
*
*
* Config data structure and setter methods used by pydh and cudh.
*/


#ifndef _CONFIG_HPP_
#define _CONFIG_HPP_

#include "common.h"
#include "config.h"

config::config() {
    precision = single_precision;
    check_input = true;
    histo2_only = false;
    cpu_threads = 1;
    cpu_blocksize = -1;  // TODO
    gpu_id = 0;
    gpu_thread_block_x = -1;  // TODO
    gpu_algorithm = -1;  // TODO
}

void config::set_precision(int val) {
    precision = val;
}

void config::set_check_input(bool val) {
    check_input = val;
}

void config::set_histo2_only(bool val) {
    histo2_only = val;
}

void config::set_cpu_threads(int val) {
    cpu_threads = val;
}

void config::set_cpu_blocksize(int val) {
    cpu_blocksize = val;
}

void config::set_gpu_id(int val) {
    gpu_id = val;
}

void config::set_gpu_thread_block_x(int val) {
    gpu_thread_block_x = val;
}

void config::set_gpu_algorithm(int val) {
    gpu_algorithm = val;
}

#endif
