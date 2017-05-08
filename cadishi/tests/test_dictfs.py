#!/usr/bin/env python2.7

"""A set of unit tests of dictfs.

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
