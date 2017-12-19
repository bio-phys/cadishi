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


// config data structure for pydh and cuda, in order not to alter the parameter list later when adding new stuff
class config {
    // parameters relevant to both the kernels
    int precision;
    bool check_input;
    bool histo2_only;
    // parameters relevant only to the CPU kernel
    int cpu_threads;
    int cpu_blocksize;
    // parameters relevant only to the GPU kernel
    int gpu_id;
    int gpu_thread_block_x;
    int gpu_algorithm;
    // --- methods below ---
    config();
    void set_precision(int);
    void set_check_input(bool);
    void set_histo2_only(bool);
    void set_cpu_threads(int);
    void set_cpu_blocksize(int);
    void set_gpu_id(int);
    void set_gpu_thread_block_x(int);
    void set_gpu_algorithm(int);
};

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
