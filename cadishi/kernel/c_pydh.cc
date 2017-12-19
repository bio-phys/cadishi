/**
* Cadishi --- CAlculation of DIStance HIstograms
*
* Copyright (c) Klaus Reuter, Juergen Koefinger
* See the file AUTHORS.rst for the full list of contributors.
*
* Released under the MIT License, see the file LICENSE.txt.
*
*
* pydh --- high performance CPU distance histogram code
*/

#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <stdint.h>

#ifdef BUILD_C_LIBRARY
#include "pydh.h"
#else
#include <Python.h>
#include <numpy/ndarrayobject.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif
#define OMP_SCHEDULE schedule(guided)

#include "config.hpp"
#include "common.hpp"
#include "exceptions.hpp"

// Attempt to make the distance array in the inner loop fit into the CPU cache,
// this does not seem to be beneficial in most use cases, so we disable it for the moment.
// #define USE_BLOCKING
#undef USE_BLOCKING
#ifdef USE_BLOCKING
const int inner_loop_blocksize = 131072;
#endif

// alignment passed to posix_memalign()
const size_t alignment = 64;

/**
 * Function: thread_histo_increment
 *
 * Increment the bin value of 'histo' for each value of the array 'd'
 * where 'd' contains 'n' rescaled integer-converted distance values.
 *
 * Optionally, 'n_out' values are subtracted from the zeroth element of 'histo'.
 */
inline void
thread_histo_increment(uint32_t * histo, int * d, const int n, const int n_out=0) {
    for (int i=0; i<n; ++i) {
        int k = d[i];
        ++histo[k];
    }
    if (n_out > 0) {
        histo[0] -= n_out;
    }
}


/**
 * Function: thread_dist_trim
 *
 * Count and zero the number of outliers in 'd'
 * where 'd' contains 'n' rescaled integer-converted distance values.
 *
 * Returns the number of outliers.
 */
inline int
thread_dist_trim(int * d, const int n, const int n_bins) {
    int n_out = 0;
    for (int i=0; i<n; ++i) {
        if (d[i] >= n_bins) {
            d[i] = 0;
            ++n_out;
        }
    }
    return n_out;
}


/**
 * Function: thread_histo_flush
 *
 * Add the values from the thread-local 'histo_thread' to the shared
 * 'histo' using atomic operations.  Optionally 'histo_thread' is zeroed.
 */
inline void
thread_histo_flush(uint64_t * histo, uint32_t * histo_thread, const int n_bins, const bool zero) {
    for (int i=0; i<n_bins; ++i) {
        uint64_t bin_val = (uint64_t)histo_thread[i];
// NOTE: performance penalty of atomic is minimal (gcc >=4.9, checked using Intel VTUNE & Advisor)
        #pragma omp atomic update
        histo[i] = histo[i] + bin_val;
    }
    if (zero)
        memset(histo_thread, 0, n_bins*sizeof(uint32_t));
}


/**
 * Function: hist_1
 *
 * Single-species distance histogram kernel.
 */
template <typename TUPLE3_T, typename FLOAT_T, bool check_input, int box_type_id>
void hist_1(TUPLE3_T * __restrict__ p,
            const int nelem,
            bool * moved_into_box,
            uint64_t *histo,
            const int nbins,
            const FLOAT_T scal,
            const TUPLE3_T * const box,
            const TUPLE3_T &box_ortho,
            const TUPLE3_T &box_inv) {
    bool idx_error = false;
    memset(histo, 0, nbins*sizeof(uint64_t));
    #pragma omp parallel default(shared) reduction(|| : idx_error)
    {
        uint32_t * histo_thread = (uint32_t*) malloc(nbins*sizeof(uint32_t));
        memset(histo_thread, 0, nbins*sizeof(uint32_t));

        int *d = NULL;
#ifdef USE_BLOCKING
        posix_memalign((void**)&d, alignment, inner_loop_blocksize*sizeof(int));
#else
        posix_memalign((void**)&d, alignment, nelem*sizeof(int));
#endif

        uint32_t count = 0;
        #pragma omp for OMP_SCHEDULE
        for (int j=0; j<nelem; ++j) {
            if ( (count + (uint32_t)j) < count /*use wrap-around feature at overflow*/) {
                thread_histo_flush(histo, histo_thread, nbins, true);
                count = 0;
            }
            count += (uint32_t)j;
#ifdef USE_BLOCKING
            int block_n_elem;
            for (int i=0; i<j; i+=inner_loop_blocksize) {
                if ((j-i) < inner_loop_blocksize) {
                    block_n_elem = j-i;
                } else {
                    block_n_elem = inner_loop_blocksize;
                }
                for (int k=0; k<block_n_elem; ++k) {
                    d[k] = (int)(scal * dist<TUPLE3_T, FLOAT_T, box_type_id>
                                 (p[i+k], p[j], box, box_ortho, box_inv));
                }
                /**
                 * Checks of the previously calculated integer-converted distances.
                 *
                 * In case we do not use a periodic box, a distance greater than the maximum allowed one
                 * translates to an error condition.  The user has to take care about the input data set.
                 *
                 * In case we have a periodic box it may be desired to discard such values and proceed.
                 */
                switch (box_type_id) {
                case none:
                    if (check_input) {
                        if (idx_error) {
                            // --- error condition is already there, do nothing ---
                        } else if (thread_dist_trim(d, block_n_elem, nbins) > 0) {
                            #pragma omp atomic write
                            idx_error = true;
                        } else {
                            thread_histo_increment(histo_thread, d, block_n_elem);
                        }
                    } else {
                        thread_histo_increment(histo_thread, d, block_n_elem);
                    }
                    break;
                case orthorhombic:
                case triclinic:
                    if (check_input) {
                        int n_out = thread_dist_trim(d, block_n_elem, nbins);
                        thread_histo_increment(histo_thread, d, block_n_elem, n_out);
                    } else {
                        thread_histo_increment(histo_thread, d, block_n_elem);
                    }
                    break;
                }
            }
#else
            // loop vectorizes well (GCC >=4.9, checked using Intel VTUNE & Advisor)
            #pragma omp simd
            for (int i=0; i<j; ++i) {
                d[i] = (int) (scal * dist<TUPLE3_T, FLOAT_T, box_type_id>
                              (p[i], p[j], box, box_ortho, box_inv));
            }
            /**
             * Checks of the previously calculated integer-converted distances.
             *
             * In case we do not use a periodic box, a distance greater than the maximum allowed one
             * translates to an error condition.  The user has to take care about the input data set.
             *
             * In case we have a periodic box it may be desired to discard such values and proceed.
             */
            switch (box_type_id) {
            case none:
                if (check_input) {
                    if (idx_error) {
                        // --- error condition is already there, do nothing ---
                    } else if (thread_dist_trim(d, j, nbins) > 0) {
                        #pragma omp atomic write
                        idx_error = true;
                    } else {
                        thread_histo_increment(histo_thread, d, j);
                    }
                } else {
                    thread_histo_increment(histo_thread, d, j);
                }
                break;
            case orthorhombic:
            case triclinic:
                if (check_input) {
                    int n_out = thread_dist_trim(d, j, nbins);
                    thread_histo_increment(histo_thread, d, j, n_out);
                } else {
                    thread_histo_increment(histo_thread, d, j);
                }
                break;
            }
#endif
        }
        thread_histo_flush(histo, histo_thread, nbins, false);
        free(d);
        free(histo_thread);
    }
    if (check_input) {
        if (idx_error)
            OVERFLOW_ERROR(overflow_error_msg);
    }
}


/**
 * Function: hist_2
 *
 * Two-species distance histogram kernel.
 */
template <typename TUPLE3_T, typename FLOAT_T, bool check_input, int box_type_id>
void hist_2(TUPLE3_T * __restrict__ p1,
            const int nelem1,
            bool * moved_into_box_1,
            TUPLE3_T * __restrict__ p2,
            const int nelem2,
            bool * moved_into_box_2,
            uint64_t *histo,
            const int nbins,
            const FLOAT_T scal,
            const TUPLE3_T * const box,
            const TUPLE3_T &box_ortho,
            const TUPLE3_T &box_inv) {
    bool idx_error = false;
    memset(histo, 0, nbins*sizeof(uint64_t));
    #pragma omp parallel default(shared) reduction(|| : idx_error)
    {
        uint32_t * histo_thread = (uint32_t*) malloc(nbins*sizeof(uint32_t));
        memset(histo_thread, 0, nbins*sizeof(uint32_t));

        int *d = NULL;
#ifdef USE_BLOCKING
        posix_memalign((void**)&d, alignment, inner_loop_blocksize*sizeof(int));
#else
        posix_memalign((void**)&d, alignment, nelem2*sizeof(int));
#endif

        uint32_t count = 0;
        #pragma omp for OMP_SCHEDULE
        for (int j=0; j<nelem1; ++j) {
            if ( (count + (uint32_t)nelem2) < count /*use wrap-around feature at overflow*/) {
                thread_histo_flush(histo, histo_thread, nbins, true);
                count = 0;
            }
            count += (uint32_t)nelem2;
#ifdef USE_BLOCKING
            int block_n_elem;
            for (int i=0; i<nelem2; i+=inner_loop_blocksize) {
                if ((nelem2-i) < inner_loop_blocksize) {
                    block_n_elem = nelem2-i;
                } else {
                    block_n_elem = inner_loop_blocksize;
                }
                for (int k=0; k<block_n_elem; ++k) {
                    d[k] = (int)(scal * dist<TUPLE3_T, FLOAT_T, box_type_id>
                                 (p2[i+k], p1[j], box, box_ortho, box_inv));
                }
                /**
                 * Checks of the previously calculated integer-converted distances.
                 *
                 * In case we do not use a periodic box, a distance greater than the maximum allowed one
                 * translates to an error condition.  The user has to take care about the input data set.
                 *
                 * In case we have a periodic box it may be desired to discard such values and proceed.
                 */
                switch (box_type_id) {
                case none:
                    if (check_input) {
                        if (idx_error) {
                            // --- error condition is already there, do nothing ---
                        } else if (thread_dist_trim(d, block_n_elem, nbins) > 0) {
                            #pragma omp atomic write
                            idx_error = true;
                        } else {
                            thread_histo_increment(histo_thread, d, block_n_elem);
                        }
                    } else {
                        thread_histo_increment(histo_thread, d, block_n_elem);
                    }
                    break;
                case orthorhombic:
                case triclinic:
                    if (check_input) {
                        int n_out = thread_dist_trim(d, block_n_elem, nbins);
                        thread_histo_increment(histo_thread, d, block_n_elem, n_out);
                    } else {
                        thread_histo_increment(histo_thread, d, block_n_elem);
                    }
                    break;
                }
            }
#else
            // loop vectorizes well (gcc >=4.9, checked using Intel VTUNE & Advisor)
            #pragma omp simd
            for (int i=0; i<nelem2; ++i) {
                d[i] = (int)(scal * dist<TUPLE3_T, FLOAT_T, box_type_id>
                             (p2[i], p1[j], box, box_ortho, box_inv));
            }
            /**
             * Checks of the previously calculated integer-converted distances.
             *
             * In case we do not use a periodic box, a distance greater than the maximum allowed one
             * translates to an error condition.  The user has to take care about the input data set.
             *
             * In case we have a periodic box it may be desired to discard such values and proceed.
             */
            switch (box_type_id) {
            case none:
                if (check_input) {
                    if (idx_error) {
                        // --- error condition is already there, do nothing ---
                    } else if (thread_dist_trim(d, nelem2, nbins) > 0) {
                        #pragma omp atomic write
                        idx_error = true;
                    } else {
                        thread_histo_increment(histo_thread, d, nelem2);
                    }
                } else {
                    thread_histo_increment(histo_thread, d, nelem2);
                }
                break;
            case orthorhombic:
            case triclinic:
                if (check_input) {
                    int n_out = thread_dist_trim(d, nelem2, nbins);
                    thread_histo_increment(histo_thread, d, nelem2, n_out);
                } else {
                    thread_histo_increment(histo_thread, d, nelem2);
                }
                break;
            }
#endif
        }
        thread_histo_flush(histo, histo_thread, nbins, false);
        free(d);
        free(histo_thread);
    }
    if (check_input) {
        if (idx_error)
            OVERFLOW_ERROR(overflow_error_msg);
    }
}


/**
 * Function: histo_cpu
 *
 * Driver of the actual kernels 'hist_1' and 'hist_2'.
 *
 * Sets up periodic box descriptors, loops over the species.
 */
template <typename TUPLE3_T, typename FLOAT_T, bool check_input, int box_type_id>
void histo_cpu(TUPLE3_T *coords, int n_tot, int *n_per_el, int n_el,
               uint64_t *histos, int n_bins, FLOAT_T r_max, int *mask,
               const TUPLE3_T * const box, const int do_histo2_only=0) {
    const FLOAT_T scal = ((FLOAT_T)n_bins)/r_max;

    // --- box-related values, to be passed as constant references
    TUPLE3_T box_ortho = {0.0};
    TUPLE3_T box_inv = {0.0};
    switch (box_type_id) {
    case none:
        break;
    case orthorhombic:
        box_ortho.x = box[0].x;  // concatenate box vectors
        box_ortho.y = box[1].y;  // into a
        box_ortho.z = box[2].z;  // single tuple
        box_inv.x = FLOAT_T(1.) / box_ortho.x;
        box_inv.y = FLOAT_T(1.) / box_ortho.y;
        box_inv.z = FLOAT_T(1.) / box_ortho.z;
        break;
    case triclinic:
        box_inv.x = FLOAT_T(1.) / box[0].x;
        box_inv.y = FLOAT_T(1.) / box[1].y;
        box_inv.z = FLOAT_T(1.) / box[2].z;
        break;
    }
    bool * moved_into_box;
    const int n_bytes = n_el * sizeof(bool);
    moved_into_box = (bool*) malloc(n_bytes);
    memset(moved_into_box, 0, n_bytes);

    int histogramIdx = 0;
    int iOffset = 0;
    for (int i=0; i<n_el; ++i) {
        int jOffset = iOffset;
        int j=i;
        // ---
        for (/*int j=i*/; j<n_el; ++j) {
            ++histogramIdx;
            int histoOffset = histogramIdx*n_bins;
            // ---
            if (mask[histogramIdx - 1] > 0) {
                if (j != i) {
                    hist_2 <TUPLE3_T, FLOAT_T, check_input, box_type_id>
                    (&coords[iOffset], n_per_el[i], &moved_into_box[i],
                     &coords[jOffset], n_per_el[j], &moved_into_box[j],
                     &histos[histoOffset], n_bins, scal,
                     box, box_ortho, box_inv);
                } else {
                    if (! do_histo2_only) {
                        hist_1 <TUPLE3_T, FLOAT_T, check_input, box_type_id>
                        (&coords[iOffset], n_per_el[i], &moved_into_box[i],
                         &histos[histoOffset], n_bins, scal,
                         box, box_ortho, box_inv);
                    }
                }
            }
            // ---
            jOffset += n_per_el[j];
        }
        iOffset += n_per_el[i];
    }

    free(moved_into_box);
}


/**
 * Function: histograms_template_dispatcher
 *
 * Dispatches the templates on the check_input and box_type level.
 */
template <typename NP_TUPLE3_T, typename TUPLE3_T, typename FLOAT_T>
void histograms_template_dispatcher(NP_TUPLE3_T *r_ptr,
                                    int n_tot,
                                    int *nel_ptr,
                                    int n_El,
                                    uint64_t *histo_ptr,
                                    int n_bins,
                                    double r_max,
                                    int *mask_ptr,
                                    double *box_ptr,
                                    bool check_input,
                                    int box_type_id,
                                    const int do_histo2_only=0) {
    TUPLE3_T* r_copy = NULL;
    posix_memalign((void**)&r_copy, alignment, n_tot*sizeof(TUPLE3_T));

    for (int i=0; i<n_tot; ++i) {
        r_copy[i].x = FLOAT_T(r_ptr[i].x);
        r_copy[i].y = FLOAT_T(r_ptr[i].y);
        r_copy[i].z = FLOAT_T(r_ptr[i].z);
    }

    TUPLE3_T* box_copy;
    if (box_type_id != none) {
        box_copy = (TUPLE3_T*) malloc(3*sizeof(TUPLE3_T));
        for (int i=0; i<3; ++i) {
            box_copy[i].x = FLOAT_T(box_ptr[3*i  ]);
            box_copy[i].y = FLOAT_T(box_ptr[3*i+1]);
            box_copy[i].z = FLOAT_T(box_ptr[3*i+2]);
        }
    } else {
        box_copy = NULL;
    }

    // --- expand all relevant template parameter combinations
    //     to avoid branching inside loops at runtime
    if (check_input) {
        switch (box_type_id) {
        case none:
            histo_cpu <TUPLE3_T, FLOAT_T, true, none>
            (r_copy, n_tot, nel_ptr, n_El, histo_ptr, n_bins, FLOAT_T(r_max),
             mask_ptr, box_copy, do_histo2_only);
            break;
        case orthorhombic:
            histo_cpu <TUPLE3_T, FLOAT_T, true, orthorhombic>
            (r_copy, n_tot, nel_ptr, n_El, histo_ptr, n_bins, FLOAT_T(r_max),
             mask_ptr, box_copy, do_histo2_only);
            break;
        case triclinic:
            histo_cpu <TUPLE3_T, FLOAT_T, true, triclinic>
            (r_copy, n_tot, nel_ptr, n_El, histo_ptr, n_bins, FLOAT_T(r_max),
             mask_ptr, box_copy, do_histo2_only);
            break;
        }
    } else {
        switch (box_type_id) {
        case none:
            histo_cpu <TUPLE3_T, FLOAT_T, false, none>
            (r_copy, n_tot, nel_ptr, n_El, histo_ptr, n_bins, FLOAT_T(r_max),
             mask_ptr, box_copy, do_histo2_only);
            break;
        case orthorhombic:
            histo_cpu <TUPLE3_T, FLOAT_T, false, orthorhombic>
            (r_copy, n_tot, nel_ptr, n_El, histo_ptr, n_bins, FLOAT_T(r_max),
             mask_ptr, box_copy, do_histo2_only);
            break;
        case triclinic:
            histo_cpu <TUPLE3_T, FLOAT_T, false, triclinic>
            (r_copy, n_tot, nel_ptr, n_El, histo_ptr, n_bins, FLOAT_T(r_max),
             mask_ptr, box_copy, do_histo2_only);
            break;
        }
    }
    free(r_copy);
    if (box_type_id != none)
        free(box_copy);
}


#ifdef BUILD_C_LIBRARY

int histograms_cpu_single(np_tuple3s_t *r_ptr,
                          int n_tot,
                          int *nel_ptr,
                          int n_El,
                          uint64_t *histo_ptr,
                          int n_bins,
                          double r_max,
                          int *mask_ptr,
                          double *box_ptr,
                          bool check_input,
                          int box_type_id) {
    int exit_status = 0;
    try {
        histograms_template_dispatcher <np_tuple3s_t, tuple3s_t, float>
        (r_ptr, n_tot, nel_ptr, n_El, histo_ptr, n_bins, r_max, mask_ptr, box_ptr, check_input, box_type_id);
    } catch (std::overflow_error & err) {
        const std::string msg = std::string(err.what());
        printf("%s\n", msg.c_str());
        exit_status = 1;
    } catch (std::runtime_error & err) {
        const std::string msg = std::string(err.what());
        printf("%s\n", msg.c_str());
        exit_status = 2;
    } catch (...) {
        // --- general unknown error
        exit_status = 3;
    }
    return exit_status;
}

int histograms_cpu_double(np_tuple3d_t *r_ptr,
                          int n_tot,
                          int *nel_ptr,
                          int n_El,
                          uint64_t *histo_ptr,
                          int n_bins,
                          double r_max,
                          int *mask_ptr,
                          double *box_ptr,
                          bool check_input,
                          int box_type_id) {
    int exit_status = 0;
    try {
        histograms_template_dispatcher <np_tuple3d_t, tuple3d_t, double>
        (r_ptr, n_tot, nel_ptr, n_El, histo_ptr, n_bins, r_max, mask_ptr, box_ptr, check_input, box_type_id);
    } catch (std::overflow_error & err) {
        const std::string msg = std::string(err.what());
        printf("%s\n", msg.c_str());
        exit_status = 1;
    } catch (std::runtime_error & err) {
        const std::string msg = std::string(err.what());
        printf("%s\n", msg.c_str());
        exit_status = 2;
    } catch (...) {
        // --- general unknown error
        exit_status = 3;
    }
    return exit_status;
}


#else


/**
 * Function: histograms
 *
 * histograms() function exposed to Python.
 *
 * Dispatches the templates on the precision level.
 */
static PyObject* histograms(PyObject* self, PyObject* args) {
    // --- required parameters
    PyArrayObject *coords;
    PyArrayObject *nelems;
    PyArrayObject *histos;
    double r_max;
    PyArrayObject *mask;
    PyArrayObject *box;
    int box_type_id = none;
    // --- optional parameters
    int precision = single_precision;
    int n_threads = 1;
    int check_input = 1;
    int do_histo2_only = 0;
    // --- translate C++ exceptions into integer exit status
    int exit_status;
#ifdef _OPENMP
    // --- save current number of openmp threads
    static int omp_threads_val = -1;
#endif

    exit_status = 0;
    try {
        if (!PyArg_ParseTuple(args, "OOOdOOi|iiii", &coords, &nelems, &histos, &r_max, &mask, &box, &box_type_id,
                              /*optional parameters:*/ &precision, &n_threads, &check_input, &do_histo2_only))
            return NULL;

        // --- 2D double precision coordinate array for all species
        RT_ASSERT( coords->nd == 2 );
        RT_ASSERT( coords->dimensions[1] == 3 );
        int n_tot = coords->dimensions[0];
        np_tuple3d_t *r_ptr = (np_tuple3d_t*) coords->data;

        // --- 1D integer array containing the number of elements for each species
        RT_ASSERT( nelems->nd == 1 );
        int n_El = nelems->dimensions[0];
        int *nel_ptr = (int*) nelems->data;

        // --- 2D integer array containing the histograms
        RT_ASSERT( histos->nd == 2 );
        int n_bins = histos->dimensions[0];
        uint64_t *histo_ptr = (uint64_t*) histos->data;

        int n_Hij = (histos->dimensions[1])-1;
        RT_ASSERT(n_Hij == mask->dimensions[0]);
        int *mask_ptr = (int*) mask->data;

        RT_ASSERT(box->dimensions[0] == 3);
        RT_ASSERT(box->dimensions[1] == 3);
        double *box_ptr = (double*) box->data;

#ifdef _OPENMP
        if (n_threads != omp_threads_val) {
            omp_set_num_threads(n_threads);
            omp_threads_val = n_threads;
        }
#endif

        if ( precision == single_precision ) {
            histograms_template_dispatcher <np_tuple3d_t, tuple3s_t, float>
            (r_ptr, n_tot, nel_ptr, n_El, histo_ptr, n_bins, r_max, mask_ptr,
             box_ptr, check_input, box_type_id, do_histo2_only);
        } else if ( precision == double_precision ) {
            histograms_template_dispatcher <np_tuple3d_t, tuple3d_t, double>
            (r_ptr, n_tot, nel_ptr, n_El, histo_ptr, n_bins, r_max, mask_ptr,
             box_ptr, check_input, box_type_id, do_histo2_only);
        } else {
            RT_ERROR(std::string("unknown precision identifier passed"));
        }

    } catch (std::overflow_error & err) {
        const std::string msg = std::string(err.what());
        printf("%s\n", msg.c_str());
        exit_status = 1;
    } catch (std::runtime_error & err) {
        const std::string msg = std::string(err.what());
        printf("%s\n", msg.c_str());
        exit_status = 2;
    } catch (...) {
        // --- general unknown error
        exit_status = 3;
    }

    return Py_BuildValue("i", exit_status);
}


template <typename TUPLE3_T, typename FLOAT_T>
void dist_driver_template_dispatcher(np_tuple3d_t *r_ptr,
                                     int n_tot,
                                     FLOAT_T *distances_ptr,
                                     double *box_ptr,
                                     int box_type_id) {
    TUPLE3_T* r_copy = NULL;
    posix_memalign((void**)&r_copy, alignment, n_tot*sizeof(TUPLE3_T));

    for (int i=0; i<n_tot; ++i) {
        r_copy[i].x = FLOAT_T(r_ptr[i].x);
        r_copy[i].y = FLOAT_T(r_ptr[i].y);
        r_copy[i].z = FLOAT_T(r_ptr[i].z);
    }

    TUPLE3_T* box_copy;
    if (box_type_id != none) {
        box_copy = (TUPLE3_T*) malloc(3*sizeof(TUPLE3_T));
        for (int i=0; i<3; ++i) {
            box_copy[i].x = FLOAT_T(box_ptr[3*i  ]);
            box_copy[i].y = FLOAT_T(box_ptr[3*i+1]);
            box_copy[i].z = FLOAT_T(box_ptr[3*i+2]);
        }
    } else {
        box_copy = NULL;
    }

    // --- box-related values
    TUPLE3_T box_ortho = {0.0};
    TUPLE3_T box_inv = {0.0};
    switch (box_type_id) {
    case none:
        break;
    case orthorhombic:
        box_ortho.x = box_copy[0].x;  // concatenate box_copy vectors
        box_ortho.y = box_copy[1].y;  // into a
        box_ortho.z = box_copy[2].z;  // single tuple
        box_inv.x = FLOAT_T(1.) / box_ortho.x;
        box_inv.y = FLOAT_T(1.) / box_ortho.y;
        box_inv.z = FLOAT_T(1.) / box_ortho.z;
        break;
    case triclinic:
        box_inv.x = FLOAT_T(1.) / box_copy[0].x;
        box_inv.y = FLOAT_T(1.) / box_copy[1].y;
        box_inv.z = FLOAT_T(1.) / box_copy[2].z;
        break;
    }

    int idx = 0;
    for (int j=0; j<n_tot; ++j) {
        for (int i=0; i<j; ++i) {
            switch (box_type_id) {
            case none:
                distances_ptr[idx] = dist <TUPLE3_T, FLOAT_T, none>
                                     (r_copy[j], r_copy[i], box_copy, box_ortho, box_inv);
                break;
            case orthorhombic:
                distances_ptr[idx] = dist <TUPLE3_T, FLOAT_T, orthorhombic>
                                     (r_copy[j], r_copy[i], box_copy, box_ortho, box_inv);
                break;
            case triclinic:
                distances_ptr[idx] = dist <TUPLE3_T, FLOAT_T, triclinic>
                                     (r_copy[j], r_copy[i], box_copy, box_ortho, box_inv);
                break;
            }
            ++idx;
        }
    }

    free(r_copy);
    if (box_type_id != none)
        free(box_copy);
}


static PyObject* dist_driver(PyObject* self, PyObject* args) {
    PyArrayObject *coords;
    PyArrayObject *distances;
    PyArrayObject *box;
    int box_type_id = none;
    int precision = single_precision;
    int exit_status;

    exit_status = 0;
    try {
        if (!PyArg_ParseTuple(args, "OOOii", &coords, &distances, &box, &box_type_id, &precision))
            return NULL;

        // --- 2D double precision coordinate array
        RT_ASSERT( coords->nd == 2 );
        RT_ASSERT( coords->dimensions[1] == 3 );
        int n_tot = coords->dimensions[0];
        np_tuple3d_t *r_ptr = (np_tuple3d_t*) coords->data;

        // --- 1D array containing the distances
        int n_dist = n_tot * (n_tot - 1) / 2;
        RT_ASSERT( distances->nd == 1 );
        RT_ASSERT( distances->dimensions[0] == n_dist);

        RT_ASSERT(box->dimensions[0] == 3);
        RT_ASSERT(box->dimensions[1] == 3);
        double *box_ptr = (double*) box->data;

        if ( precision == single_precision ) {
            float *distances_ptr = (float*) malloc(n_dist*sizeof(float));
            memset(distances_ptr, 0, n_dist*sizeof(float));
            dist_driver_template_dispatcher <tuple3s_t, float>
            (r_ptr, n_tot, distances_ptr, box_ptr, box_type_id);
            double *distances_raw = (double*) distances->data;
            for (int i=0; i<n_dist; ++i) {
                distances_raw[i] = double(distances_ptr[i]);
            }
            free(distances_ptr);
        } else if ( precision == double_precision ) {
            double *distances_ptr = (double*) distances->data;
            dist_driver_template_dispatcher <tuple3d_t, double>
            (r_ptr, n_tot, distances_ptr, box_ptr, box_type_id);
        } else {
            RT_ERROR(std::string("unknown precision identifier passed"));
        }
    } catch (std::overflow_error & err) {
        const std::string msg = std::string(err.what());
        printf("%s\n", msg.c_str());
        exit_status = 1;
    } catch (std::runtime_error & err) {
        const std::string msg = std::string(err.what());
        printf("%s\n", msg.c_str());
        exit_status = 2;
    } catch (...) {
        // --- general unknown error
        exit_status = 3;
    }

    return Py_BuildValue("i", exit_status);
}


// --- register C functions as Python modules
static PyMethodDef c_pydh_Methods[] = {
    {"histograms",  histograms,  METH_VARARGS, "calculate distance histogram between two species"},
    {"dist_driver", dist_driver, METH_VARARGS, "driver for the distance calculation only, for testing purposes"},
    {NULL, NULL, 0, NULL}
};


// --- Python module initialization
PyMODINIT_FUNC
initc_pydh(void) {
    (void) Py_InitModule("c_pydh", c_pydh_Methods);
    // --- the following helper function must be called
    import_array();
}

#endif // BUILD_C_LIBRARY
