# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4 fileencoding=utf-8
#
# Cadishi --- CAlculation of DIStance HIstograms
#
# Copyright (c) Klaus Reuter, Juergen Koefinger
# See the file AUTHORS.rst for the full list of contributors.
#
# Released under the MIT License, see the file LICENSE.txt.

"""Various NumPy- and dictionary-related utilities.

Implements add, append, and scale operations for numerical data (ie. NumPy
arrays) stored in dictionaries.  In addition, an ASCII output routine is
provided.
"""


import copy
import numpy as np
import json

from . import util


def sum_values(X, Y, skip_keys=['radii', 'frame']):
    """Implement X += Y where X and Y are Python dictionaries (with string keys)
    that contain summable data types.
    The operation is applied to X for any value in Y, excluding keys that are in
    the list skip_keys.
    Typically, the values of X, Y are NumPy arrays (e.g. histograms) that are summed.

    Parameters
    ----------
    X : dict
        X is a dictionary with string keys that contains NumPy arrays.
    Y : dict
        Y is a dictionary with string keys that contains NumPy arrays.
    skip_keys : list of strings
        skip_keys is a list of strings for which the sum operation is skipped.

    Returns
    -------
    None
        The function sum_values operates on X directly
        and does not return anything.
    """
    assert isinstance(X, dict)
    assert isinstance(Y, dict)
    for key in list(Y.keys()):
        if key in skip_keys:
            continue
        if key not in X:
            X[key] = copy.deepcopy(Y[key])
        else:
            X[key] += Y[key]


def scale_values(X, C, skip_keys=['radii', 'frame']):
    """Implement X = X times C where X is a Python dictionary that contains supported
    data types.
    The operation is applied to any value in X, excluding keys that are in the
    list skip_keys.
    Typically, the values of X are NumPy arrays (histograms) that are rescaled
    after summation using a scalar C (e.g. to implement averaging operation).

    Parameters
    ----------
    X : dict
        X is a dictionary with string keys that contains NumPy arrays.
    C : scalar, NumPy array
        C is a multiplier, either a scalar of a NumPy array of size compatible
        with the contents of X.
    skip_keys : list of strings
        skip_keys is a list of strings for which the sum operation is skipped.

    Returns
    -------
    None
        The function scale_values operates on X directly
        and does not return anything.
    """
    assert isinstance(X, dict)
    for key in list(X.keys()):
        if key in skip_keys:
            continue
        X[key] *= C


def append_values(X, Y, skip_keys=['radii']):
    """Implement X.append(Y) where X and Y are Python dictionaries that contain
    NumPy data types.  The operation is applied to X for any value in Y,
    excluding keys that are in the list skip_keys.  Typically, the values of X,
    Y are NumPy arrays (e.g. particle numbers) that are appended.

    Parameters
    ----------
    X : dict
        X is a dictionary with string keys that contains NumPy arrays.
    Y : dict
        Y is a dictionary with string keys that contains NumPy arrays.
    skip_keys : list of strings
        skip_keys is a list of strings for which the append operation is skipped.

    Returns
    -------
    None
        The function scale_values operates on X directly
        and does not return anything.
    """
    assert isinstance(X, dict)
    assert isinstance(Y, dict)
    for key in list(Y.keys()):
        if key in skip_keys:
            continue
        if key not in X:
            X[key] = copy.deepcopy(Y[key])
        else:
            X[key] = np.append(X[key], Y[key])


def write_dict(dic, path, level=0):
    """Write a dictionary containing NumPy arrays or other Python data
    structures to text files.  In case the dictionary contains other
    dictionaries, the function is called recursively.  The keys should
    be strings to guarantee successful operation.

    Parameters
    ----------
    dic : dictionary
        A dictionary containing NumPy arrays or other Python data structures.
    path : string
        Path where the dictionary and its data shall be written to.
    level : int, optional
        Level in the nested-dictionary hierarchy during recursive operation.
        This parameter was added for debugging purposes and does not have any
        practical relevance.

    Returns
    -------
    None
        The function write_dict does not return anything.
    """
    np_keys = []
    py_keys = []
    for key in list(dic.keys()):
        val = dic[key]
        if isinstance(val, dict):
            _path = path + '/' + key
            _level = level + 1
            write_dict(val, _path, _level)
        else:
            if isinstance(val, np.ndarray):
                np_keys.append(key)
            else:
                py_keys.append(key)
    # ---
    np_keys.sort()
    py_keys.sort()
    # --- (1) save NumPy arrays to text files
    rad = 'radii'
    if rad in np_keys:
        np_keys.remove(rad)
        np_keys.insert(0, rad)
    # ---
    np_all_1d = True
    for key in np_keys:
        val = dic[key]
        if (len(val.shape) > 1):
            np_all_1d = False
            break
    if (len(np_keys) > 0):
        if np_all_1d:
            # --- concatenate arrays into a 2d array
            val = dic[np_keys[0]]
            n_row = val.shape[0]
            n_col = len(np_keys)
            arr = np.zeros([n_row, n_col])
            for idx, key in enumerate(np_keys):
                arr[:, idx] = (dic[key])[:]
            # --- build header
            if rad in np_keys:
                np_keys.remove(rad)
            header = '#'
            for key in np_keys:
                header = header + ' ' + key
            # --- dump data
            util.savetxtHeader(path + '.dat', header, arr)
        else:
            # --- we save arrays with more than one dimension separately
            for key in np_keys:
                arr = dic[key]
                # --- dump data
                util.savetxtHeader(path + '/' + key + '.dat', '# ' + key, arr)
    # --- (2) for robustness, save any other Python data to JSON text files
    if (len(py_keys) > 0):
        for key in py_keys:
            filename = path + '/' + key + '.json'
            util.md(filename)
            with open(filename, "w") as fp:
                json.dump(dic[key], fp, indent=4, sort_keys=True)
