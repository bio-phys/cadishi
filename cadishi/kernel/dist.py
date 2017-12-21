# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4 fileencoding=utf-8
#
# Cadishi --- CAlculation of DIStance HIstograms
#
# Copyright (c) Klaus Reuter, Juergen Koefinger
# See the file AUTHORS.rst for the full list of contributors.
#
# Released under the MIT License, see the file LICENSE.txt.

"""dist Python interface to c_dist.

Initial implementation of the distance histogram computation using Cython.
Legacy code, do not use for production. This code is only used by the test
suite to compare the results of (small) problems between the kernels.
"""


from builtins import range
import numpy as np
try:
    from . import c_dist
except:
    have_c_dist = False
else:
    have_c_dist = True


def histograms(coordinate_sets, r_max, n_bins):
    """Distance histogram calculation on the CPU using Cython kernels.

    Calculate histograms from sets of species coordinates by calling the
    pwd() and pwd2() CPU functions from the Cython dist_knl.  Serves as the
    reference implementation for the pydh and cudh packages.

    Parameters
    ----------
    coordinate_sets : list of numpy.ndarray
        List of numpy arrays where each numpy array contains
        the atom coordinates for all the atoms of a species.
    r_max : float
        Cutoff radius for the distance calculation.
    n_bins : int
        Number of histogram bins.

    Returns
    -------
    numpy.ndarray
        Two-dimensional numpy array containing the distance histograms.
    """

    if not have_c_dist:
        raise RuntimeError("could not import compiled dist kernel (c_dist.so)")

    n_El = len(coordinate_sets)
    n_Hij = n_El * (n_El + 1) / 2
    histos = np.zeros((n_bins, n_Hij + 1), order='F')
    idx = 0
    for i in range(n_El):
        for j in range(i, n_El):
            idx += 1
            histo = np.zeros(n_bins)
            if (i == j):
                c_dist.pwd(coordinate_sets[i], histo, r_max, n_bins)
            else:
                c_dist.pwd2(coordinate_sets[i], coordinate_sets[j], histo, r_max, n_bins)
            histos[:, idx] += histo[:]
    return histos
