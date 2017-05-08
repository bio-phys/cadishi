# vim:fileencoding=utf-8
"""pydh Python interface.

Exposes the c_pydh C++ high-performance kernel to Python via
the histograms() function.
"""
# This file is part of the Cadishi package.  See README.rst,
# LICENSE.txt, and the documentation for details.


__author__ = "Klaus Reuter"
__copyright__ = "Copyright (C) 2015-2016 Klaus Reuter"
__license__ = "license_tba"


import numpy as np
from six.moves import range
from . import common
try:
    from . import c_pydh
except:
    have_c_pydh = False
else:
    have_c_pydh = True
from .. import pbc


def histograms(coordinate_sets,
               r_max,
               n_bins,
               precision="single",
               pydh_threads=1,
               check_input=True,
               scale_factors=[],
               mask_array=[],
               box=[]):
    """Distance histogram calculation."""

    if not have_c_pydh:
        raise RuntimeError(common.import_pydh_error_msg)

    for cs in coordinate_sets:
        assert(cs.dtype == np.float64)
    assert(r_max > 0.0)
    assert(n_bins > 0)

    n_El = len(coordinate_sets)
    n_Hij = n_El * (n_El + 1) / 2

    # --- concatenate list of numpy arrays into a single numpy array
    np_coord = np.concatenate(coordinate_sets, axis=0)
    np_nelem = np.zeros((n_El), dtype=np.int32)
    for i in range(n_El):
        np_nelem[i] = coordinate_sets[i].shape[0]

    # --- To see the bins contiguously in memory from C, we use the following layout:
    np_histos = np.zeros((n_bins, n_Hij + 1), dtype=np.uint64, order='F')

    if (n_Hij == len(mask_array)):
        np_mask = np.asarray(mask_array, dtype=np.int32)
    else:
        np_mask = np.ones(n_Hij, dtype=np.int32)

    np_box, box_type_id, box_type = pbc.get_standard_box(box, verbose=False)

    if (len(box) > 0):
        print(common.indent + "pydh box_type: " + str(box_type))

    precision = common.precision_to_enum(precision)

    # --- run the CUDH distance histogram kernel
    exit_status = c_pydh.histograms(np_coord, np_nelem, np_histos, r_max, np_mask, \
                                    np_box, box_type_id, \
                                    precision, pydh_threads, check_input)

    if (exit_status == 1):
        raise ValueError(common.overflow_error_msg)
    elif (exit_status >= 2):
        raise RuntimeError(common.general_error_msg)

    # --- re-scale histograms in case an appropriately sized scale factor array is passed
    if (n_Hij == len(scale_factors)):
        np_scales = np.asarray(scale_factors)
        np_histos = common.scale_histograms(np_histos, np_scales)

    return np_histos



def dist_driver(coordinates,
                precision="single",
                box=[],
                force_triclinic=False):
    """Debug driver for the distance calculation functions."""

    if not have_c_pydh:
        raise RuntimeError(common.import_pydh_error_msg)

    np_coord = np.asanyarray(coordinates, dtype=np.float64)
    n_tot = np_coord.shape[0]
    assert(np_coord.shape[1] == 3)

    # --- To see the bins contiguously in memory from C, we use the following layout:
    n_dist = n_tot * (n_tot - 1) / 2
    np_dist = np.zeros(n_dist, dtype=np.float64)

    np_box, box_type_id, box_type = pbc.get_standard_box(box, force_triclinic=force_triclinic, verbose=False)

    if (len(box) > 0):
        print("dist_driver box_type: " + str(box_type))

    precision = common.precision_to_enum(precision)

    exit_status = c_pydh.dist_driver(np_coord, np_dist, np_box, box_type_id, precision)

    if (exit_status == 1):
        raise ValueError(common.overflow_error_msg)
    elif (exit_status >= 2):
        raise RuntimeError(common.general_error_msg)

    return np_dist
