# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4 fileencoding=utf-8
#
# Cadishi --- CAlculation of DIStance HIstograms
#
# Copyright (c) Klaus Reuter, Juergen Koefinger
# See the file AUTHORS.rst for the full list of contributors.
#
# Released under the MIT License, see the file LICENSE.txt.

"""A set of unit tests of h5pickle.
"""


import os
import sys
import numpy as np
import h5py
import copy

from .. import util
from .. import h5pickle


do_cleanup = True
h5name = util.scratch_dir() + "test_h5pickle.h5"
data_ref = {}


def test_h5py_save():
    global data_ref
    h5fp = h5py.File(h5name, "w")
    a = 3.14
    h5pickle.save(h5fp, 'a', a)
    s = "Hello World!"
    h5pickle.save(h5fp, 's', s)
    l = [1, 2, 3, 4, 5]
    h5pickle.save(h5fp, 'l', l)
    np_a = np.random.rand(3)
    h5pickle.save(h5fp, 'np_a', np_a)
    np_b = np.random.rand(2, 2)
    h5pickle.save(h5fp, 'np_b', np_b)
    d = {}
    d['a'] = a
    d['s'] = s
    d['l'] = l
    d['np_a'] = np_a
    d['np_b'] = np_b
    another_d = {}
    another_d['np_a'] = np_a
    another_d['np_b'] = np_b
    another_d['e'] = 2.7
    d['another_d'] = another_d
    h5pickle.save(h5fp, 'd', d)
    data_ref = copy.deepcopy(d)
    # print data_ref


def test_h5py_load():
    h5fp = h5py.File(h5name, 'r')
    data = h5pickle.load(h5fp)
    # print
    # print util.SEP
    # print data
    # print util.SEP
    # print data


if do_cleanup:
    def test_final_cleanup():
        util.rmrf(util.scratch_dir())
