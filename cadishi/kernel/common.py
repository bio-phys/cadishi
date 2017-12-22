# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4 fileencoding=utf-8
#
# Cadishi --- CAlculation of DIStance HIstograms
#
# Copyright (c) Klaus Reuter, Juergen Koefinger
# See the file AUTHORS.rst for the full list of contributors.
#
# Released under the MIT License, see the file LICENSE.txt.

"""Common code used by the Cadishi kernels pydh and cudh.
"""


from builtins import range
import numpy as np


# --- whitespace compatible with script_histogram_par's time stamp
indent = "                       "


overflow_error_msg = "at least one pair distance exceeded r_max; please check n_bins, r_max, and the coordinates"
histo2_error_msg = "exactly two coordinate sets have to be given to perform the histo2 calculation only"
general_error_msg = "general error occurred during kernel execution"
import_pydh_error_msg = "could not import compiled PYDH kernel (c_pydh.so)"
import_cudh_error_msg = "could not import compiled CUDH kernel (c_cudh.so)"

# --- translate string into C enum for interfacing, see <common.hpp>
_enum_precision = {}
_enum_precision['single'] = 0
_enum_precision['single_precision'] = 0
_enum_precision[4] = 0
_enum_precision['double'] = 1
_enum_precision['double_precision'] = 1
_enum_precision[8] = 1


def precision_to_enum(str_id):
    """Helper function to translate a string precision identification to the integer
    identification used inside the kernels PYDH and CUDH."""
    return _enum_precision[str_id]


def scale_histograms(np_histos, np_scales):
    """Scale histograms with scale factors."""
    # get the number of partial histograms, the first column contains the bin radii
    n_ij = np_scales.shape[0]
    assert(n_ij == np_histos.shape[1] - 1)
    # raw histograms are by default int64, rescaled ones are in general float64
    np_histos_scaled = np_histos.astype(np.float64)
    for i in range(n_ij):
        np_histos_scaled[:, i + 1] *= np_scales[i]
    return np_histos_scaled
