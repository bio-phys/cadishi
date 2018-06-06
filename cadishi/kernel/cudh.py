# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4 fileencoding=utf-8
#
# Cadishi --- CAlculation of DIStance HIstograms
#
# Copyright (c) Klaus Reuter, Juergen Koefinger
# See the file AUTHORS.rst for the full list of contributors.
#
# Released under the MIT License, see the file LICENSE.txt.

"""cudh Python interface.

Calls the c_cudh Python module.
"""
from __future__ import print_function

import subprocess as sub

from builtins import str
from builtins import zip
from builtins import range

import numpy as np
from . import common
try:
    from . import c_cudh
except:
    have_c_cudh = False
else:
    have_c_cudh = True
from .. import pbc


def get_num_devices():
    """Get the number of available CUDA devices (i.e. GPUs).

    We do not use the function "c_cudh.get_num_devices()" because it is
    not allowed to fork and use CUDA in processes after a first CUDA call,
    which would be the case in <histograms.py> (and was hard to figure out).
    """
    n = 0
    cmd = "nvidia-smi -L".split()
    try:
        raw = sub.check_output(cmd).lower().split('\n')
        gpus = [x for x in raw if x.startswith("gpu")]
        n = len(gpus)
    except:
        pass
    return n


def histograms(coordinate_sets,
               r_max,
               n_bins,
               precision="single",
               gpu_id=0,
               do_histo2_only=False,
               thread_block_x=0,
               check_input=True,
               scale_factors=[],
               mask_array=[],
               box=[],
               force_triclinic=False,
               verbose=False,
               algorithm=-1):
    """Distance histogram calculation on NVIDIA GPUs using CUDA.

    Calculate distance histograms for sets of species coordinates by calling the
    CUDA kernels that are provided by the Python module c_cudh written in CUDA.

    Parameters
    ----------
    coordinate_sets : list of numpy.ndarray
        List of numpy arrays where each numpy array contains
        the atom coordinates for all the atoms of a species.
    r_max : float
        Cutoff radius for the distance calculation.
    n_bins : int
        Number of histogram bins.
    precision : string, optional
        String specifying the implementation and/or the precision.  "single" is the
        default value for single precision, use "double" for double precision.
    gpu_id : int, optional
        The GPU to be used to calculate the histograms.  0 is the default value.
    do_histo2_only : bool, optional
        In case only two sets of coordinates are given, calculate only the distance
        histogram between the species sets.  For benchmark purposes only.
    thread_block_x : int, optional
        Manually set the CUDA thread block size (x), overrides the internal default
        if set to a value larger than zero.  For benchmark and debugging purposes.

    Returns
    -------
    numpy.ndarray
        Two-dimensional numpy array containing the distance histograms.
    """

    if not have_c_cudh:
        raise RuntimeError(common.import_cudh_error_msg)

    for cs in coordinate_sets:
        assert(cs.dtype == np.float64)

    assert(r_max > 0.0)
    assert(n_bins > 0)

    n_El = len(coordinate_sets)
    n_Hij = n_El * (n_El + 1) / 2

    if do_histo2_only and (n_El != 2):
        raise ValueError(common.histo2_error_msg)

    # Reorder coordinate sets by size to maximize the performance of the CUDA
    # smem kernels, this is most advantageous when small and large sets are mixed.
    do_reorder = True
    if do_histo2_only:
        np_mask = np.zeros(3, dtype=np.int32)
        np_mask[1] = 1
        do_reorder = False
    else:
        if (n_Hij == len(mask_array)):
            np_mask = np.asarray(mask_array, dtype=np.int32)
            # TODO : implement reordering of mask_array for the general case
            if (np.sum(np.where(np_mask <= 0)) > 0):
                do_reorder = False
        else:
            np_mask = np.ones(n_Hij, dtype=np.int32)

    if do_reorder:
        # --- create lists containing (indices,sizes) sorted by size
        el_idx = list(range(n_El))
        el_siz = []
        for i in el_idx:
            el_siz.append(coordinate_sets[i].shape[0])
        idx_siz = list(zip(el_idx, el_siz))
        idx_siz.sort(key=lambda tup: tup[1])
        el_idx_srt, el_siz_srt = list(zip(*idx_siz))

        # --- create reordered concatenated numpy arrays
        n_coord = sum(el_siz)
        np_coord = np.zeros((n_coord, 3))
        np_nelem = np.zeros((n_El,), dtype=np.int32)
        jc = 0  # coordinate array offset
        jn = 0  # nelem array offset
        for tup in idx_siz:
            idx, siz = tup
            np_coord[jc:jc + siz, :] = coordinate_sets[idx]
            np_nelem[jn] = siz
            jc += siz
            jn += 1

        # --- create a hash, mapping (i,j) to the linear index used in
        #     the histograms array
        idx = {}
        count = 1
        for i in range(n_El):
            ii = el_idx_srt[i]
            for j in range(i, n_El):
                jj = el_idx_srt[j]
                tup = (ii, jj)
                idx[tup] = count
                tup = (jj, ii)
                idx[tup] = count
                count += 1
    else:
        # --- concatenate list of numpy arrays into a single numpy array
        np_coord = np.concatenate(coordinate_sets, axis=0)
        np_nelem = np.zeros((n_El), dtype=np.int32)
        for i in range(n_El):
            np_nelem[i] = coordinate_sets[i].shape[0]

    # --- To see the bins contiguously in memory from C, we use the following layout:
    histos = np.zeros((n_bins, n_Hij + 1), dtype=np.uint64, order='F')

    np_box, box_type_id, box_type = pbc.get_standard_box(box,
                                                         force_triclinic=force_triclinic, verbose=False)

    if (box_type is not None):
        print(common.indent + "cudh box_type: " + str(box_type))

    precision = common.precision_to_enum(precision)

    # --- run the CUDH distance histogram kernel
    exit_status = c_cudh.histograms(np_coord, np_nelem, histos, r_max, np_mask, np_box, box_type_id,
                                    precision, check_input, verbose, gpu_id, thread_block_x, algorithm)

    if (exit_status == 1):
        #c_cudh.free()
        raise ValueError(common.overflow_error_msg)
    elif (exit_status >= 2):
        #c_cudh.free()
        raise RuntimeError(common.general_error_msg)

    if do_reorder:
        # --- restore the expected order
        histo_ret = np.zeros_like(histos)
        count = 1
        for i in el_idx:
            for j in range(i, n_El):
                histo_ret[:, count] = histos[:, idx[(i, j)]]
                count += 1
    else:
        histo_ret = histos

    # --- re-scale histograms in case an appropriately sized scale factor array is passed
    if (n_Hij == len(scale_factors)):
        np_scales = np.asarray(scale_factors)
        histo_ret = common.scale_histograms(histo_ret, np_scales)

    return histo_ret
