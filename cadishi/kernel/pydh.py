# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4 fileencoding=utf-8
#
# Cadishi --- CAlculation of DIStance HIstograms
#
# Copyright (c) Klaus Reuter, Juergen Koefinger
# See the file AUTHORS.rst for the full list of contributors.
#
# Released under the MIT License, see the file LICENSE.txt.

"""pydh Python interface.

Calls the c_pydh Python module.
"""
from __future__ import print_function

from builtins import str
from builtins import range

import numpy as np
from . import common

try:
    from . import c_pydh
    # from cadishi.kernel import c_pydh
except:
    have_c_pydh = False
else:
    have_c_pydh = True


from .. import pbc


def histograms(coordinate_sets,
               r_max,
               n_bins,
               precision="single",
               n_threads=1,
               blocksize=0,
               check_input=True,
               scale_factors=[],
               mask_array=[],
               box=[],
               force_triclinic=False,
               do_histo2_only=False,
               verbose=False):
    """Distance histogram calculation, Python interface.

    This function is the entry point to run distance histogram calculations
    from Python on the CPU.

    Parameters
    ----------
    coordinate_sets : list
        list of double precision numpy arrays [N, 3] containing coordinate triples
    r_max : float
        maximum distance that should be considered for the histograms
    n_bins : int
        number of histogram bins
    precision : string, optional
        String indicating the precision to be used, "single" (default) or "double"
    n_threads : int, optional
        Number of CPU (OpenMP) threads to be used for the computation, default 1.
    blocksize : int, optional
        Size of the blocks used to optimize cache usage in units of coordinate tuples.
        Values around 230 are best to optimize for a L2 cache size of 256 kb. Possible
        values are: '-1' to disable cache blocking and use naive (old) kernels,
        '0' to enable blocking and use internal logic to determine the block size
        (recommended), and positive values to set the block size manually (not recommended,
        only useful for performance exploration). Note that cache blocking is disabled
        internally for bin numbers larger than 48000.
    check_input : bool, optional
        Check the input, i.e. check it the distances fit into the histogram window
        defined by r_max. Harms performance to some degree but avoids potential
        memory corruption in case of bogus input. Default is 'True'.
    scale_factors : array-like, optional
        Array of floats used to rescale individual histograms after computation.
    mask_array : array-like, optional
        Array of integers to mask the computation of certain histograms, where
        '0' indicates that a histogram is skipped.  Internal default is all '1'.
    box : array-like, optional
        Periodic box specification.  Default is none.
    force_triclinic : bool, optional
        Use the triclinic box implementation also for simpler orthorhombic boxes.
        Default is 'False'.
    do_histo2_only : bool, optional
        Perform only the inter-species computation in case 2 species are given.
        Useful for debugging and performance optimization, default is 'False'.
    verbose : bool, optional
        Be verbose about internal states during computation.  Default is 'False'.

    Returns
    -------
    NumPy array, double precision
        Column 0 contains the radii, other columns contain the histograms.
    """

    if not have_c_pydh:
        raise RuntimeError(common.import_pydh_error_msg)

    for cs in coordinate_sets:
        assert(cs.dtype == np.float64)
    assert(r_max > 0.0)
    assert(n_bins > 0)

    n_El = len(coordinate_sets)
    n_Hij = (n_El * (n_El + 1)) // 2

    if do_histo2_only and (n_El != 2):
        raise ValueError(common.histo2_error_msg)

    # --- concatenate list of numpy arrays into a single numpy array
    np_coord = np.concatenate(coordinate_sets, axis=0)
    np_nelem = np.zeros((n_El), dtype=np.int32)
    for i in range(n_El):
        np_nelem[i] = coordinate_sets[i].shape[0]

    # --- To see the bins contiguously in memory from C, we use the following layout:
    np_histos = np.zeros((n_bins, n_Hij + 1), dtype=np.uint64, order='F')

    if do_histo2_only:
        np_mask = np.zeros(3, dtype=np.int32)
        np_mask[1] = 1
    else:
        if (n_Hij == len(mask_array)):
            np_mask = np.asarray(mask_array, dtype=np.int32)
        else:
            np_mask = np.ones(n_Hij, dtype=np.int32)

    np_box, box_type_id, box_type = pbc.get_standard_box(box,
                                                         force_triclinic=force_triclinic, verbose=False)

    if (box_type is not None):
        print(common.indent + "pydh box_type: " + str(box_type))

    precision = common.precision_to_enum(precision)

    # --- run the CUDH distance histogram kernel
    exit_status = c_pydh.histograms(np_coord, np_nelem, np_histos, r_max, np_mask,
                                    np_box, box_type_id,  # optional arguments follow
                                    precision, check_input, verbose, n_threads, blocksize)

    if (exit_status == 1):
        raise ValueError(common.overflow_error_msg)
    elif (exit_status >= 2):
        raise RuntimeError(common.general_error_msg)

    # --- re-scale histograms in case an appropriately sized scale factor array is passed
    if (n_Hij == len(scale_factors)):
        np_scales = np.asarray(scale_factors)
        np_histos = common.scale_histograms(np_histos, np_scales)

    return np_histos


def distances(coordinates, precision="single", box=[], force_triclinic=False):
    """Driver for the distance calculation functions."""

    if not have_c_pydh:
        raise RuntimeError(common.import_pydh_error_msg)

    np_coord = np.asanyarray(coordinates, dtype=np.float64)
    n_tot = np_coord.shape[0]
    assert(np_coord.shape[1] == 3)

    # --- To see the bins contiguously in memory from C, we use the following layout:
    n_dist = (n_tot * (n_tot - 1)) // 2
    np_dist = np.zeros(n_dist, dtype=np.float64)

    np_box, box_type_id, box_type = pbc.get_standard_box(box, force_triclinic=force_triclinic, verbose=False)

    if (len(box) > 0):
        print("distances box_type: " + str(box_type))

    precision = common.precision_to_enum(precision)

    exit_status = c_pydh.distances(np_coord, np_dist, np_box, box_type_id, precision)

    if (exit_status == 1):
        raise ValueError(common.overflow_error_msg)
    elif (exit_status >= 2):
        raise RuntimeError(common.general_error_msg)

    return np_dist
