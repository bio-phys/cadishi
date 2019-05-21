# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4 fileencoding=utf-8
#
# Cadishi --- CAlculation of DIStance HIstograms
#
# Copyright (c) Klaus Reuter, Juergen Koefinger
# See the file AUTHORS.rst for the full list of contributors.
#
# Released under the MIT License, see the file LICENSE.txt.

"""Cadishi ASCII IO library.

ASCII data writer for base.Container instances, mainly designed for debugging
purposes.  A reader is currently not implemented.
"""
from __future__ import print_function


from builtins import str
import numpy as np
import json

from .. import base
from .. import dict_util


class ASCIIReader(base.Reader):
    """ASCII data reader, currently not implemented."""

    def __init__(self):
        raise NotImplementedError()


class ASCIIWriter(base.Writer):
    """ASCII data writer for base.Container instances."""
    _depends = []
    _conflicts = []

    def __init__(self, directory='.', source=-1, verbose=False):
        self.directory = directory
        self.src = source
        self.verb = verbose
        self.counter = 0
        # ---
        self._depends.extend(super(base.Writer, self)._depends)
        self._conflicts.extend(super(base.Writer, self)._conflicts)

    def write_frame(self, frm):
        path = self.directory + '/' + str(frm.i)
        dict_util.write_dict(frm.data, path)

    def dump(self):
        for frm in next(self.src):
            if self.verb and (self.counter % 10 == 0):
                print("ASCIIWriter.dump() : ", frm.i)
            self.counter += 1
            self.write_frame(frm)
