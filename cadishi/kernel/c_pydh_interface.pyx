import numpy as np
cimport numpy as np
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
        void set_histo2_only(bool)
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


cdef extern from "c_pydh.h":
    int histograms_cpu( np_tuple3d_t *r_ptr,
                        int n_tot,
                        int *nel_ptr,
                        int n_El,
                        np.uint64_t *histo_ptr,
                        int n_bins,
                        double r_max,
                        int *mask_ptr,
                        double *box_ptr,
                        int box_type_id,
                        const config & cfg)

    int distances_cpu(  np_tuple3d_t *r_ptr,
                        int n_tot,
                        double *distances,
                        double *box_ptr,
                        int box_type_id,
                        const config & cfg)


# wrapper to make histograms_cpu() accessible from Python
def histograms(np.ndarray r_ptr,
               np.ndarray nel_ptr,
               np.ndarray histo_ptr,
               float r_max,
               np.ndarray mask_ptr,
               np.ndarray box_ptr,
               int box_type_id,
               int precision,
               int n_threads,
               bool check_input,
               bool do_histo2_only):

    # derive dimensions from NumPy data structures
    cdef int n_tot
    cdef int n_El
    cdef int n_bins
    n_tot = r_ptr.shape[0]
    n_El = nel_ptr.shape[0]
    n_bins = histo_ptr.shape[0]
    # checked OK:
    # print n_tot
    # print n_El
    # print n_bins

    # create Python instance of C++ config class
    cdef config cfg
    cfg.set_precision(precision)
    cfg.set_cpu_threads(n_threads)
    cfg.set_check_input(check_input)
    cfg.set_histo2_only(do_histo2_only)

    cdef int exit_status
    exit_status = histograms_cpu(<np_tuple3d_t*> r_ptr.data,
                                 <int> n_tot,
                                 <int*> nel_ptr.data,
                                 <int> n_El,
                                 <np.uint64_t*> histo_ptr.data,
                                 <int> n_bins,
                                 <double> r_max,
                                 <int*> mask_ptr.data,
                                 <double*> box_ptr.data,
                                 <int> box_type_id,
                                 cfg)
    return exit_status


# wrapper to make histograms_cpu() accessible from Python
def distances(np.ndarray r_ptr,
              np.ndarray dists,
              np.ndarray box_ptr,
              int box_type_id,
              int precision):
    # derive dimension from NumPy data structure
    cdef int n_tot
    n_tot = r_ptr.shape[0]

    # create Python instance of C++ config class
    cdef config cfg
    cfg.set_precision(precision)

    cdef int exit_status
    exit_status = distances_cpu(<np_tuple3d_t*> r_ptr.data,
                                <int> n_tot,
                                <double*> dists.data,
                                <double*> box_ptr.data,
                                <int> box_type_id,
                                cfg)
    return exit_status
