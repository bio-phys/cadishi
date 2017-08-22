# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4 fileencoding=utf-8
#
# Cadishi --- CAlculation of DIStance HIstograms
#
# Copyright (c) Klaus Reuter, Juergen Koefinger
# See the file AUTHORS.rst for the full list of contributors.
#
# Released under the MIT License, see the file LICENSE.txt.

"""A set of unit tests of the dictfs code.
"""


import os
import sys
import numpy as np
import pprint
from .. import base


def test_base_container():
    obj = base.Container()
    obj.i = 1
    obj.put_meta({'foo': {'bar': True}})
    # ---
    a = np.random.rand(3, 3)
    path = "coordinates/Cl"
    obj.put_data(path, a)
    # ---
    aa = obj.get_data(path)
    assert np.all(a == aa)
    # ---
    path = '/'
    all_data = obj.get_data(path)
    # pprint.pprint(all_data)
