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
#include <algorithm>
#include <stdint.h>
#include "c_pydh.h"

#ifdef _OPENMP
#include <omp.h>
#endif
#define OMP_SCHEDULE schedule(guided)

#include "config.hpp"
#include "common.hpp"
#include "exceptions.hpp"

// alignment parameter passed to posix_memalign()
const size_t alignment = 64;


template <typename T>
void print_histo(T * histo, const int n) {
    printf("---\n");
    for (int i=0; i<n; ++i) {
        printf(" %d %d\n", i, histo[i]);
    }
    printf("---\n");
}


template <typename T>
inline void memset_value(T * arr, const T val, const int n) {
    #pragma omp simd
    for (int i=0; i<n; ++i) {
        arr[i] = val;
    }
}


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
        // negative values are used to mark unused array elements by the blocked kernels
        if (k >= 0)
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
            const int n,
            uint64_t *histo,
            const int n_bins,
            const FLOAT_T scal,
            const TUPLE3_T * const box,
            const TUPLE3_T &box_ortho,
            const TUPLE3_T &box_inv) {
    CHECKPOINT("hist_1()");
    bool idx_error = false;
    bool mem_error = false;
    memset(histo, 0, n_bins*sizeof(uint64_t));
    #pragma omp parallel default(shared) reduction(|| : idx_error, mem_error)
    {
        uint32_t * histo_thread = (uint32_t*) malloc(n_bins*sizeof(uint32_t));
        memset(histo_thread, 0, n_bins*sizeof(uint32_t));
        int *d = NULL;
        mem_error = (posix_memalign((void**)&d, alignment, n*sizeof(int)) != 0);
        if (! mem_error) {
            uint32_t count = 0;
            #pragma omp for OMP_SCHEDULE
            for (int j=0; j<n; ++j) {
                if ( (count + (uint32_t)j) < count /*use wrap-around feature at overflow*/) {
                    thread_histo_flush(histo, histo_thread, n_bins, true);
                    count = 0;
                }
                count += (uint32_t)j;
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
                        } else if (thread_dist_trim(d, j, n_bins) > 0) {
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
                        int n_out = thread_dist_trim(d, j, n_bins);
                        thread_histo_increment(histo_thread, d, j, n_out);
                    } else {
                        thread_histo_increment(histo_thread, d, j);
                    }
                    break;
                }
            }
            thread_histo_flush(histo, histo_thread, n_bins, false);
            free(d);
            free(histo_thread);
        }
    }
    print_histo(histo, n_bins);
    if (mem_error) {
        RT_ERROR("memory allocation")
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
            const int n1,
            TUPLE3_T * __restrict__ p2,
            const int n2,
            uint64_t *histo,
            const int n_bins,
            const FLOAT_T scal,
            const TUPLE3_T * const box,
            const TUPLE3_T &box_ortho,
            const TUPLE3_T &box_inv) {
    CHECKPOINT("hist_2()");
    bool idx_error = false;
    bool mem_error = false;
    memset(histo, 0, n_bins*sizeof(uint64_t));
    #pragma omp parallel default(shared) reduction(|| : idx_error, mem_error)
    {
        uint32_t * histo_thread = (uint32_t*) malloc(n_bins*sizeof(uint32_t));
        memset(histo_thread, 0, n_bins*sizeof(uint32_t));
        int *d = NULL;
        mem_error = (posix_memalign((void**)&d, alignment, n2*sizeof(int)) != 0);
        if (! mem_error) {
            uint32_t count = 0;
            #pragma omp for OMP_SCHEDULE
            for (int j=0; j<n1; ++j) {
                if ( (count + (uint32_t)n2) < count /*use wrap-around feature at overflow*/) {
                    thread_histo_flush(histo, histo_thread, n_bins, true);
                    count = 0;
                }
                count += (uint32_t)n2;
                // loop vectorizes well (gcc >=4.9, checked using Intel VTUNE & Advisor)
                #pragma omp simd
                for (int i=0; i<n2; ++i) {
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
                        } else if (thread_dist_trim(d, n2, n_bins) > 0) {
                            #pragma omp atomic write
                            idx_error = true;
                        } else {
                            thread_histo_increment(histo_thread, d, n2);
                        }
                    } else {
                        thread_histo_increment(histo_thread, d, n2);
                    }
                    break;
                case orthorhombic:
                case triclinic:
                    if (check_input) {
                        int n_out = thread_dist_trim(d, n2, n_bins);
                        thread_histo_increment(histo_thread, d, n2, n_out);
                    } else {
                        thread_histo_increment(histo_thread, d, n2);
                    }
                    break;
                }
            }
            thread_histo_flush(histo, histo_thread, n_bins, false);
            free(d);
            free(histo_thread);
        }
    }
    print_histo(histo, n_bins);
    if (mem_error) {
        RT_ERROR("memory allocation")
    }
    if (check_input) {
        if (idx_error)
            OVERFLOW_ERROR(overflow_error_msg);
    }
}


/**
 * Function calculate_blocksize()
 *
 * Calculates the block size in elements (assuming a 256 kB L2 cache size
 * such that block_dist_rectangle() and the thread local histogram do fit into L2.
 */
template <typename TUPLE3_T, typename FLOAT_T>
int calculate_blocksize(const int n_bins,
                        const int n_bytes_l2=1<<18,         // 256 kB
                        const int n_bytes_reserve=1<<14)    //  16 kB
{
    const float n_bytes_cache = n_bytes_l2 - n_bytes_reserve;
    const float n_bytes_tuple = sizeof(TUPLE3_T);
    const float b_bytes_word = sizeof(FLOAT_T);
    const float n_bytes_int = sizeof(int);
    // solve quadratic equation y = a * x**2 + b * x + c
    const float y = n_bytes_cache;
    const float a = n_bytes_int;
    const float b = 2 * n_bytes_tuple;
    const float c = n_bins * n_bytes_int;
    const float cy = c-y;
    const float n_block = (-b + std::sqrt(b*b - 4.*a*cy))/(2.*a);
    return int(std::floor(n_block));
}


template <typename TUPLE3_T, typename FLOAT_T, int box_type_id>
inline void block_dist_rectangle(
            const TUPLE3_T * const p1_stripe, const int jj_max,
            const TUPLE3_T * const p2_stripe, const int ii_max,
            const FLOAT_T scal,
            int * d,
            const int bs,
            const TUPLE3_T * const box,
            const TUPLE3_T &box_ortho,
            const TUPLE3_T &box_inv) {
    for (int jj=0; jj<jj_max; ++jj) {
        // Note: nested loop structure vectorizes thanks to 'd_stripe'
        int * d_stripe = &d[jj*bs];
        memset_value(d_stripe, -1, bs);
        #pragma omp simd
        for (int ii=0; ii<ii_max; ++ii) {
            d_stripe[ii] = int(scal * dist <TUPLE3_T, FLOAT_T, box_type_id>
                        (p2_stripe[ii], p1_stripe[jj], box, box_ortho, box_inv));
            printf("--> %d\n", d_stripe[ii]);
        }
    }
}


template <typename TUPLE3_T, typename FLOAT_T, int box_type_id>
inline void block_dist_bevel(
            const TUPLE3_T * const p1_stripe, const int jj_max,
            const TUPLE3_T * const p2_stripe, const int ii_max,
            const FLOAT_T scal,
            int * d,
            const int bs,
            const TUPLE3_T * const box,
            const TUPLE3_T &box_ortho,
            const TUPLE3_T &box_inv) {
    for (int jj=0; jj<jj_max; ++jj) {
        // Note: nested loop structure vectorizes thanks to 'd_stripe'
        int * d_stripe = &d[jj*bs];
        memset_value(d_stripe, -1, bs);
        #pragma omp simd
        for (int ii=0; ii<jj; ++ii) {
            d_stripe[ii] = int(scal * dist <TUPLE3_T, FLOAT_T, box_type_id>
                        (p2_stripe[ii], p1_stripe[jj], box, box_ortho, box_inv));
            printf("--> %d\n", d_stripe[ii]);
        }
    }
}


/**
 * Function: hist_blocked
 *
 * Two-species distance histogram kernel.
 */
template <typename TUPLE3_T, typename FLOAT_T, bool check_input, int box_type_id>
void hist_blocked(const TUPLE3_T * const p1,
            const int n1,
            const TUPLE3_T * const p2,
            const int n2,
            uint64_t *histo,
            const int n_bins,
            const FLOAT_T scal,
            const TUPLE3_T * const box,
            const TUPLE3_T &box_ortho,
            const TUPLE3_T &box_inv) {
    CHECKPOINT("hist_blocked()");

    // detect if an intra- or inter-species calculation is performed
    const bool q_intra_species = (p1 == p2);

    const int bs = calculate_blocksize<TUPLE3_T, FLOAT_T>(n_bins);
    const int nb = bs*bs;  // number of elements/distances per block

    printf("### calculated blocksize :: %d ###\n", bs);
    printf("### q_intra_species      :: %d ###\n", q_intra_species);

    bool idx_error = false;
    bool mem_error = false;
    memset(histo, 0, n_bins*sizeof(uint64_t));

    #pragma omp parallel default(shared) reduction(|| : idx_error, mem_error)
    {
        uint32_t * histo_thread = NULL;
        mem_error = mem_error || (posix_memalign((void**)&histo_thread, alignment, n_bins*sizeof(uint32_t)) != 0);
        memset(histo_thread, 0, n_bins*sizeof(uint32_t));

        int * d = NULL;
        mem_error = mem_error || (posix_memalign((void**)&d, alignment, nb*sizeof(int)) != 0);
        memset_value(d, -1, nb);

        TUPLE3_T * p1_stripe = NULL;
        mem_error = mem_error || (posix_memalign((void**)&p1_stripe, alignment, bs*sizeof(TUPLE3_T)) != 0);

        TUPLE3_T * p2_stripe = NULL;
        mem_error = mem_error || (posix_memalign((void**)&p2_stripe, alignment, bs*sizeof(TUPLE3_T)) != 0);

        uint32_t count = 0;

        // Note: omp collapse cannot be used here due to the non-static inner loop limit
        #pragma omp for OMP_SCHEDULE
        for (int j=0; j<n1; j+=bs) {
            int jj_max = std::min(n1-j, bs);
            memmove(p1_stripe, &p1[j], jj_max*sizeof(TUPLE3_T));
            for (int i=0; i<n2; i+=bs) {
                int ii_max = std::min(n2-i, bs);
                memmove(p2_stripe, &p2[i], ii_max*sizeof(TUPLE3_T));

                // flush per-thread histogram in case a bin might approach the 32 bit integer limit
                if ( (count + (uint32_t)nb) < count /*use wrap-around feature at overflow*/) {
                    thread_histo_flush(histo, histo_thread, n_bins, true);
                    count = 0;
                }
                count += (uint32_t)nb;

                printf("==> %d %d %d %d\n", i, j, ii_max, jj_max);

                if (q_intra_species) {
                    if (i+bs < j+1) {
                        printf("inner block\n");
                        // inner block
                        block_dist_rectangle <TUPLE3_T, FLOAT_T, box_type_id>
                            (p1_stripe, jj_max, p2_stripe, ii_max, scal, d, bs, box, box_ortho, box_inv);
                    } else {
                        printf("edge block\n");
                        // edge/beveled block
                        block_dist_bevel <TUPLE3_T, FLOAT_T, box_type_id>
                            (p1_stripe, jj_max, p2_stripe, ii_max, scal, d, bs, box, box_ortho, box_inv);
                    }
                } else {
                    block_dist_rectangle <TUPLE3_T, FLOAT_T, box_type_id>
                        (p1_stripe, jj_max, p2_stripe, ii_max, scal, d, bs, box, box_ortho, box_inv);
                }

                print_histo(d, nb);
                const int c = nb;

                switch (box_type_id) {
                case none:
                    if (check_input) {
                        if (idx_error) {
                            // --- error condition is already there, do nothing ---
                        } else if (thread_dist_trim(d, c, n_bins) > 0) {
                            #pragma omp atomic write
                            idx_error = true;
                        } else {
                            thread_histo_increment(histo_thread, d, c);
                        }
                    } else {
                        thread_histo_increment(histo_thread, d, c);
                    }
                    break;
                case orthorhombic:
                case triclinic:
                    if (check_input) {
                        int n_out = thread_dist_trim(d, c, n_bins);
                        thread_histo_increment(histo_thread, d, c, n_out);
                    } else {
                        thread_histo_increment(histo_thread, d, c);
                    }
                    break;
                }
            }
        }
        thread_histo_flush(histo, histo_thread, n_bins, false);
        free(d);
        free(p1_stripe);
        free(p2_stripe);
        free(histo_thread);
    }
    print_histo(histo, n_bins);
    if (mem_error) {
        RT_ERROR("memory allocation")
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
               const TUPLE3_T * const box, const int blocksize) {
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
                    if (blocksize > 0) {
                        hist_blocked <TUPLE3_T, FLOAT_T, check_input, box_type_id>
                                            (&coords[iOffset], n_per_el[i],
                                             &coords[jOffset], n_per_el[j],
                                             &histos[histoOffset], n_bins, scal,
                                             box, box_ortho, box_inv);
                    } else {
                        hist_2 <TUPLE3_T, FLOAT_T, check_input, box_type_id>
                            (&coords[iOffset], n_per_el[i],
                             &coords[jOffset], n_per_el[j],
                             &histos[histoOffset], n_bins, scal,
                             box, box_ortho, box_inv);
                    }
                } else {
                    if (blocksize > 0) {
                        hist_blocked <TUPLE3_T, FLOAT_T, check_input, box_type_id>
                                            (&coords[iOffset], n_per_el[i],
                                             &coords[jOffset], n_per_el[j],
                                             &histos[histoOffset], n_bins, scal,
                                             box, box_ortho, box_inv);
                    } else {
                        hist_1 <TUPLE3_T, FLOAT_T, check_input, box_type_id>
                            (&coords[iOffset], n_per_el[i],
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
                                    int box_type_id,
                                    const config & cfg) {
    TUPLE3_T* r_copy = NULL;
    if (posix_memalign((void**)&r_copy, alignment, n_tot*sizeof(TUPLE3_T)) != 0) {
        RT_ERROR("memory allocation");
    }

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
    if (cfg.check_input) {
        switch (box_type_id) {
        case none:
            histo_cpu <TUPLE3_T, FLOAT_T, true, none>
            (r_copy, n_tot, nel_ptr, n_El, histo_ptr, n_bins, FLOAT_T(r_max),
             mask_ptr, box_copy, cfg.cpu_blocksize);
            break;
        case orthorhombic:
            histo_cpu <TUPLE3_T, FLOAT_T, true, orthorhombic>
            (r_copy, n_tot, nel_ptr, n_El, histo_ptr, n_bins, FLOAT_T(r_max),
             mask_ptr, box_copy, cfg.cpu_blocksize);
            break;
        case triclinic:
            histo_cpu <TUPLE3_T, FLOAT_T, true, triclinic>
            (r_copy, n_tot, nel_ptr, n_El, histo_ptr, n_bins, FLOAT_T(r_max),
             mask_ptr, box_copy, cfg.cpu_blocksize);
            break;
        }
    } else {
        switch (box_type_id) {
        case none:
            histo_cpu <TUPLE3_T, FLOAT_T, false, none>
            (r_copy, n_tot, nel_ptr, n_El, histo_ptr, n_bins, FLOAT_T(r_max),
             mask_ptr, box_copy, cfg.cpu_blocksize);
            break;
        case orthorhombic:
            histo_cpu <TUPLE3_T, FLOAT_T, false, orthorhombic>
            (r_copy, n_tot, nel_ptr, n_El, histo_ptr, n_bins, FLOAT_T(r_max),
             mask_ptr, box_copy, cfg.cpu_blocksize);
            break;
        case triclinic:
            histo_cpu <TUPLE3_T, FLOAT_T, false, triclinic>
            (r_copy, n_tot, nel_ptr, n_El, histo_ptr, n_bins, FLOAT_T(r_max),
             mask_ptr, box_copy, cfg.cpu_blocksize);
            break;
        }
    }
    free(r_copy);
    if (box_type_id != none)
        free(box_copy);
}


template <typename NP_TUPLE3_T, typename TUPLE3_T, typename FLOAT_T>
void distances_template_dispatcher(NP_TUPLE3_T *r_ptr,
                                     int n_tot,
                                     FLOAT_T *distances_ptr,
                                     double *box_ptr,
                                     int box_type_id) {
    TUPLE3_T* r_copy = NULL;
    if (posix_memalign((void**)&r_copy, alignment, n_tot*sizeof(TUPLE3_T)) != 0) {
        RT_ERROR("memory allocation");
    }

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


// --- interfaces below, to be called from user code ---


int histograms_cpu(np_tuple3d_t *r_ptr,
                   int n_tot,
                   int *nel_ptr,
                   int n_El,
                   uint64_t *histo_ptr,
                   int n_bins,
                   double r_max,
                   int *mask_ptr,
                   double *box_ptr,
                   int box_type_id,
                   const config & cfg) {
#ifdef _OPENMP
    omp_set_num_threads(cfg.cpu_threads);
#endif
    if (cfg.verbose) {
        cfg.print_config();
    }
    int exit_status = 0;
    try {
        if (cfg.precision == single_precision) {
            // NOTE: histograms_template_dispatcher() does the conversion to single precision internally
            histograms_template_dispatcher <np_tuple3d_t, tuple3s_t, float>
                (r_ptr, n_tot, nel_ptr, n_El, histo_ptr, n_bins, r_max, mask_ptr, box_ptr, box_type_id, cfg);
        } else {
            histograms_template_dispatcher <np_tuple3d_t, tuple3d_t, double>
                (r_ptr, n_tot, nel_ptr, n_El, histo_ptr, n_bins, r_max, mask_ptr, box_ptr, box_type_id, cfg);
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
    return exit_status;
}

/*
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
*/

int distances_cpu(np_tuple3d_t *r_ptr,
                  int n_tot,
                  double *distances,
                  double *box_ptr,
                  int box_type_id,
                  const config & cfg) {
    int exit_status = 0;
    try {
        if (cfg.precision == single_precision) {
            throw std::runtime_error(std::string("single precision distances are currently not implemented"));
        } else {
            distances_template_dispatcher <np_tuple3d_t, tuple3d_t, double>
                (r_ptr, n_tot, distances, box_ptr, box_type_id);
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
    return exit_status;
}
