# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4 fileencoding=utf-8
#
# Cadishi --- CAlculation of DIStance HIstograms
#
# Copyright (c) Klaus Reuter, Juergen Koefinger
# See the file AUTHORS.rst for the full list of contributors.
#
# Released under the MIT License, see the file LICENSE.txt.

"""Python-to-HDF5 serialization.

h5pickle.py provides load() and save() routines to write Python data structures
into HDF5 files.  It works with NumPy arrays and basic Python data types.
Nested dictionaries are used to map HDF5 group hierarchies.

Note: The code is likely to fail with more complicated Python data types.

Working with the typical data sets used with Cadishi and Capriqorn, the HDF5
serialization implemented by h5pickle turns out to be a factor of 10 faster than
Python's native Pickle.
"""


import six
import numpy as np
# disable warnings intended to warn H5PY developers, but may confuse our users
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py

def save(h5_grp, key, data, compression=None):
    """Save commonly used Python data structures to a HDF5 file/group.
    For dictionaries, this function is called recursively, using the
    keys as labels to create sub-groups.
    """
    assert isinstance(key, six.string_types)
    if isinstance(data, dict):
        # --- save dictionary content into a subgroup
        sub_group = h5_grp.create_group(key)
        for key2 in list(data.keys()):
            save(sub_group, key2, data[key2], compression)
    elif isinstance(data, np.ndarray):
        # --- save NumPy arrays as HDF5 datasets
        maxshape = [None for _i in data.shape]
        h5_grp.create_dataset(key, data=data, maxshape=maxshape,
                              chunks=True, compression=compression)
    elif isinstance(data, list):
        dup = []
        for i in data:
            if isinstance(i, six.string_types):
                # encode into portable ASCII (http://docs.h5py.org/en/stable/strings.html)
                dup.append(np.string_(i))
            else:
                dup.append(i)
        h5_grp.attrs[key] = dup
    elif isinstance(data, six.string_types):
        # encode into portable ASCII (http://docs.h5py.org/en/stable/strings.html)
        h5_grp.attrs[key] = np.string_(data)
    else:
        # --- Attempt to save any other Python data structure as HDF5 attribute which may rise an exception!
        h5_grp.attrs[key] = data


def load(h5_grp):
    """Load a HDF5 group recursively into a Python dictionary,
    and return the dictionary.
    """
    data = {}
    for key in list(h5_grp.keys()):
        h5py_class = h5_grp.get(key, getclass=True)
        if h5py_class is h5py._hl.group.Group:
            # print h5py_class, "Group"
            subgrp = h5_grp[key]
            val = load(subgrp)
        elif h5py_class is h5py._hl.dataset.Dataset:
            # print h5py_class, "Data"
            val = (h5_grp[key])[()]
        else:
            # shouldn't be reached at all
            raise ValueError
        data[key] = val
    for key in h5_grp.attrs:
        data[key] = h5_grp.attrs[key]
    return data
