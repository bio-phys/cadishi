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

#include <cstdio>
#include "common.h"
#include "config.h"

config::config() {
    precision = single_precision;
    check_input = true;
    histo2_only = false;
    verbose = false;
    cpu_threads = 1;
    cpu_blocksize = -1;  // TODO
    gpu_id = 0;
    gpu_thread_block_x = -1;  // TODO
    gpu_algorithm = -1;  // TODO
}

void config::print_config() const {
    printf("--- config ---\n");
    printf(" precision            %d\n", precision);
    printf(" check_input          %d\n", check_input);
    printf(" histo2_only          %d\n", histo2_only);
    printf(" verbose              %d\n", verbose);
    printf(" cpu_threads          %d\n", cpu_threads);
    printf(" cpu_blocksize        %d\n", cpu_blocksize);
    printf(" gpu_id               %d\n", gpu_id);
    printf(" gpu_thread_block_x   %d\n", gpu_thread_block_x);
    printf(" gpu_algorithm        %d\n", gpu_algorithm);
    printf("--------------\n");
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

void config::set_verbose(bool val) {
    verbose = val;
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
