# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4 fileencoding=utf-8
#
# Cadishi --- CAlculation of DIStance HIstograms
#
# Copyright (c) Klaus Reuter, Juergen Koefinger
# See the file AUTHORS.rst for the full list of contributors.
#
# Released under the MIT License, see the file LICENSE.txt.

"""A set of unit tests of dictfs.
"""


import os
import sys
import numpy as np
import pprint
from nose.tools import *
from .. import dictfs


data = {}


def test_dictfs_save():
    global data
    path = "some/weird/location"
    obj = np.random.rand(4, 4)
    dictfs.save(data, path, obj)
    # print data


def test_dictfs_load():
    global data
    path = "some/weird/location"
    obj = dictfs.load(data, path)
    # print obj
    assert np.all(data['some']['weird']['location'] == obj)


def test_dictfs_query1():
    global data
    path = "some/weird/location"
    assert dictfs.exists(data, path)


def test_dictfs_delete():
    global data
    path = "some/weird/location"
    dictfs.delete(data, path)
    # print data


def test_dictfs_query2():
    global data
    path = "some/weird/location"
    assert not dictfs.exists(data, path)


def test_dictfs_replace_dict():
    global data
    path = "/"
    obj = {'foo': 'bar'}
    dictfs.save(data, path, obj)
    # pprint.pprint(data)


def test_dictfs_has_key():
    global data
    path = "foo"
    assert dictfs.exists(data, path)


def test_dictfs_has_key2():
    global data
    path = "foo2"
    assert not dictfs.exists(data, path)


@raises(KeyError)
def test_dictfs_load_fail():
    global data
    path = "some/other/location"
    obj = dictfs.load(data, path)
    # print obj
