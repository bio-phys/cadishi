#!/usr/bin/env python2.7

"""A set of unit tests of the dictfs code.

This file is part of the Cadishi package.  See README.rst,
LICENSE.txt, and the documentation for details.
"""


__author__ = "Klaus Reuter"
__copyright__ = "Copyright (C) 2015-2016 Klaus Reuter"
__license__ = "license_tba"


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
