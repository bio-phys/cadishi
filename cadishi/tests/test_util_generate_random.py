#!/usr/bin/env python2.7

"""A set of unit tests for the pydh CPU and cudh GPU histogram modules.

This file is part of the Cadishi package.  See README.rst,
LICENSE.txt, and the documentation for details.
"""


import os
import sys
import numpy as np
import glob
import math
import multiprocessing
import pytest
from cadishi import util

def test_generate_random_coordinate_set():
    n_atoms = [1024]
    coords = util.generate_random_coordinate_set(n_atoms=n_atoms)
    for coord_set in coords:
        # print coord_set
        assert(np.min(coord_set) >= 0.0)
        assert(np.max(coord_set) <= 1.0)

def test_generate_random_coordinate_set_min():
    n_atoms = [1024]
    coord_min = (-1.,-2.,-3.)
    coords = util.generate_random_coordinate_set(n_atoms=n_atoms, coord_min=coord_min)
    for coord_set in coords:
        # print coord_set
        assert(np.all(np.min(coord_set, axis=0) >= coord_min))
        assert(np.max(coord_set) <= 1.0)

def test_generate_random_coordinate_set_max():
    n_atoms = [1024]
    coord_max = (2.,3.,4.)
    coords = util.generate_random_coordinate_set(n_atoms=n_atoms, coord_max=coord_max)
    for coord_set in coords:
        # print coord_set
        assert(np.min(coord_set) >= 0.0)
        assert(np.all(np.max(coord_set, axis=0) <= coord_max))

def test_generate_random_coordinate_set_min_max():
    n_atoms = [1024]
    coord_min = (-1.,-2.,-3.)
    coord_max = (2.,3.,4.)
    coords = util.generate_random_coordinate_set(n_atoms=n_atoms, coord_min=coord_min, coord_max=coord_max)
    for coord_set in coords:
        # print coord_set
        assert(np.all(np.min(coord_set, axis=0) >= coord_min))
        assert(np.all(np.max(coord_set, axis=0) <= coord_max))
