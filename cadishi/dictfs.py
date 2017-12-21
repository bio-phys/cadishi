# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4 fileencoding=utf-8
#
# Cadishi --- CAlculation of DIStance HIstograms
#
# Copyright (c) Klaus Reuter, Juergen Koefinger
# See the file AUTHORS.rst for the full list of contributors.
#
# Released under the MIT License, see the file LICENSE.txt.

"""dictfs, the dictionary-based in-memory "file system".

Store and retrieve data from nested dictionaries in memory using path name
strings similar to a UNIX file system. Can be used in tandem with HDF5 IO.
"""


from past.builtins import basestring
import copy
from . import util


def _store_obj(node, subnodes, obj):
    """Walk through nested dictionaries recursively, create empty ones, if
    necessary, and store a deep copy of obj.
    """
    assert isinstance(node, dict)
    assert isinstance(subnodes, list)
    assert len(subnodes) > 0
    assert isinstance(subnodes[0], basestring)
    if len(subnodes) == 1:
        if subnodes[0] == '':
            # replace the dict at node itself
            assert isinstance(obj, dict)
            node.clear()
            for key in list(obj.keys()):
                node[key] = copy.deepcopy(obj[key])
        else:
            node[subnodes[0]] = copy.deepcopy(obj)
    else:
        if subnodes[0] not in node:
            node[subnodes[0]] = {}
        _store_obj(node[subnodes[0]], subnodes[1:], obj)


def _fetch_obj(node, subnodes):
    """Walk through nested dictionaries recursively,
    retrieve the object at the requested location.
    """
    assert isinstance(node, dict)
    assert isinstance(subnodes, list)
    assert len(subnodes) > 0
    assert isinstance(subnodes[0], basestring)
    if len(subnodes) == 1:
        if subnodes[0] == '':
            return node
        else:
            return node[subnodes[0]]
    else:
        return _fetch_obj(node[subnodes[0]], subnodes[1:])


def _delete_obj(node, subnodes):
    """Walk through nested dictionaries recursively,
    delete the object at the requested location.
    """
    assert isinstance(node, dict)
    assert isinstance(subnodes, list)
    assert len(subnodes) > 0
    assert isinstance(subnodes[0], basestring)
    if len(subnodes) == 1:
        if len(subnodes[0]) > 0:
            if subnodes[0] in node:
                del node[subnodes[0]]
    else:
        return _delete_obj(node[subnodes[0]], subnodes[1:])


def _query_obj(node, subnodes):
    """Walk through nested dictionaries recursively,
    inquire if the object exists at the requested location.
    """
    assert isinstance(node, dict)
    assert isinstance(subnodes, list)
    assert len(subnodes) > 0
    assert isinstance(subnodes[0], basestring)
    if len(subnodes) == 1:
        if subnodes[0] in node:
            return True
        else:
            return False
    else:
        if subnodes[0] in node:
            return _query_obj(node[subnodes[0]], subnodes[1:])
        else:
            return False


def _path_to_list(path):
    """Convert a Unix-Style location path to a list of its substrings.
    If the location path is already a list, do nothing.
    """
    if isinstance(path, basestring):
        return util.tokenize(path)
    elif isinstance(path, list):
        return path
    else:
        raise ValueError()


# --- API routines below ---

def save(node, path, obj):
    """Save a deepcopy of obj at path relative to node.
    """
    assert isinstance(node, dict)
    subnodes = _path_to_list(path)
    _store_obj(node, subnodes, obj)


def load(node, path):
    """Return the object at path relative to node.
    """
    assert isinstance(node, dict)
    subnodes = _path_to_list(path)
    return _fetch_obj(node, subnodes)


def delete(node, path):
    """Delete the object at path relative to node.
    """
    assert isinstance(node, dict)
    subnodes = _path_to_list(path)
    _delete_obj(node, subnodes)


def exists(node, path):
    """Query the object's existence at path relative to node.
    """
    assert isinstance(node, dict)
    subnodes = _path_to_list(path)
    return _query_obj(node, subnodes)
