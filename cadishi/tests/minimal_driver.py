#!/usr/bin/env python2.7
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4 fileencoding=utf-8
#
# Cadishi --- CAlculation of DIStance HIstograms
#
# Copyright (c) Klaus Reuter, Juergen Koefinger
# See the file AUTHORS.rst for the full list of contributors.
#
# Released under the MIT License, see the file LICENSE.txt.

"""A set of unit tests for the pydh CPU and cudh GPU histogram modules.
"""


import os
import sys
import numpy as np
from cadishi import util
from cadishi.kernel import pydh


# --- global variables ---
# r_max, coordinates are in a unit box
r_max = np.sqrt(3.0)


def get_triclinic_box():
    """Return an (arbitrarily defined) triclinic box."""
    return np.asarray([0.66, 0.75, 0.88, 33., 45., 66.])


def get_orthorhombic_triclinic_box():
    """Return an (arbitrarily defined) orthorhombic box (using a triclinic specifier)."""
    return np.asarray([0.66, 0.75, 0.88, 90., 90., 90.])


def get_orthorhombic_box():
    """Return an (arbitrarily defined) orthorhombic box."""
    box = np.zeros((3, 3))
    box[0][0] = 0.66
    box[1][1] = 0.75
    box[2][2] = 0.88
    return box


def minimal_driver():
    DUMP_DATA = bool(int(os.environ.get("DUMP_DATA", "0")))
    n_atoms = [32768]
    n_bins = 16384
    coords = util.generate_random_coordinate_set(n_atoms)

    box = None
    # box = get_triclinic_box()
    # box = get_orthorhombic_box()
    force_triclinic = False

    # box = get_orthorhombic_triclinic_box()
    # force_triclinic = True

    histo = pydh.histograms(coords, r_max, n_bins, box=box, precision="double",
                            force_triclinic=force_triclinic,
                            n_threads=util.get_n_cpu_cores(), verbose=True)

    if DUMP_DATA:
        file_name = sys._getframe().f_code.co_name + ".dat"
        util.dump_histograms(file_name, histo, r_max, n_bins)


if __name__ == "__main__":
    minimal_driver()
