import numpy as np
cimport numpy as np
from libc cimport stdint
from libcpp cimport bool


cdef extern from "config.h":
    cdef cppclass config:
        int precision
        bool histo2_only
        int cpu_threads
        bool check_input
        int cpu_blocksize
        int gpu_id
        int gpu_thread_block_x
        int gpu_algorithm
        config()
        void set_precision(int)
        void set_check_input(bool)
        void set_verbose(bool)
        void set_cpu_threads(int)
        void set_cpu_blocksize(int)
        void set_gpu_id(int)
        void set_gpu_thread_block_x(int)
        void set_gpu_algorithm(int)


cdef extern from "common.h":
    ctypedef struct np_tuple3d_t:
        np.float64_t x
        np.float64_t y
        np.float64_t z


cdef extern from "c_cudh.h":
    int _cudaGetDeviceCount()
    int histograms_gpu( np_tuple3d_t *r_ptr,
                        int n_tot,
                        int *nel_ptr,
                        int n_El,
                        stdint.uint64_t *histo_ptr,
                        int n_bins,
                        int n_Hij,
                        double r_max,
                        int *mask_ptr,
                        double *box_ptr,
                        int box_type_id,
                        const config & cfg) nogil


def get_num_cuda_devices():
    cdef int n
    n = _cudaGetDeviceCount()
    return n


# wrapper to make histograms_cpu() accessible from Python
def histograms(np.ndarray r_ptr,
               np.ndarray nel_ptr,
               np.ndarray histo_ptr,
               float r_max,
               np.ndarray mask_ptr,
               np.ndarray box_ptr,
               int box_type_id,
               # ---
               int precision,       # single or double precision
               int check_input,     # perform distance check before binning
               int verbose,         #  verbose output
               int gpu_id,          #  id of the GPU to be used
               int thread_block_x,  #  CUDA thread block size
               int algorithm):      #  algorithm selection

    # derive dimensions from NumPy data structures
    cdef int n_tot
    cdef int n_El
    cdef int n_bins
    cdef int n_Hij
    n_tot = r_ptr.shape[0]
    n_El = nel_ptr.shape[0]
    n_bins = histo_ptr.shape[0]
    n_Hij = histo_ptr.shape[1] - 1  # mind the original meaning of n_Hij
    # checked OK:
    # print n_tot
    # print n_El
    # print n_bins

    # create Python instance of C++ config class
    cdef config cfg
    cfg.set_precision(precision)
    cfg.set_check_input(check_input)
    cfg.set_verbose(verbose)
    cfg.set_gpu_id(gpu_id)
    cfg.set_gpu_thread_block_x(thread_block_x)
    cfg.set_gpu_algorithm(algorithm)

    cdef int exit_status
    with nogil:
        exit_status = histograms_gpu(<np_tuple3d_t*> r_ptr.data,
                                     <int> n_tot,
                                     <int*> nel_ptr.data,
                                     <int> n_El,
                                     <stdint.uint64_t*> histo_ptr.data,
                                     <int> n_bins,
                                     <int> n_Hij,
                                     <double> r_max,
                                     <int*> mask_ptr.data,
                                     <double*> box_ptr.data,
                                     <int> box_type_id,
                                     cfg)
    return exit_status
