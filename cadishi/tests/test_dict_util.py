# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4 fileencoding=utf-8
#
# Cadishi --- CAlculation of DIStance HIstograms
#
# Copyright (c) Klaus Reuter, Juergen Koefinger
# See the file AUTHORS.rst for the full list of contributors.
#
# Released under the MIT License, see the file LICENSE.txt.

"""A set of unit tests of dict_util.
"""


import os
import sys
import numpy as np
import pprint
from .. import dictfs
from .. import dict_util


data = {}


def test_dict_util_sum():
    global data
    # ---
    path1 = "some/weird/location"
    path2 = "some/other/location"
    obj = np.ones(10, dtype=np.int32)
    dictfs.save(data, path1 + '/data', obj)
    obj = np.ones(10, dtype=np.int32)
    dictfs.save(data, path2 + '/data', obj)
    # ---
    dict1 = dictfs.load(data, path1)
    dict2 = dictfs.load(data, path2)
    dict_util.sum_values(dict1, dict2)
    # ---
    assert np.all(dictfs.load(data, path1 + '/data') == 2)
    assert np.all(dictfs.load(data, path2 + '/data') == 1)


def test_dict_util_sum_create():
    global data
    # ---
    path1 = "some/empty/location"
    path2 = "some/strange/location"
    obj = {}
    dictfs.save(data, path1, obj)
    obj = np.ones(10, dtype=np.int32)
    dictfs.save(data, path2 + '/data', obj)
    # ---
    dict1 = dictfs.load(data, path1)
    dict2 = dictfs.load(data, path2)
    dict_util.sum_values(dict1, dict2)
    # ---
    assert np.all(dictfs.load(data, path1 + '/data') == 1)
    assert np.all(dictfs.load(data, path2 + '/data') == 1)


def test_dict_util_scale():
    global data
    # ---
    path2 = "some/other/location"
    dict2 = dictfs.load(data, path2)
    dict_util.scale_values(dict2, 4)
    # ---
    assert np.all(dictfs.load(data, path2 + '/data') == 4)
