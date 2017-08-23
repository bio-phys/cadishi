# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4 fileencoding=utf-8
#
# Cadishi --- CAlculation of DIStance HIstograms
#
# Copyright (c) Klaus Reuter, Juergen Koefinger
# See the file AUTHORS.rst for the full list of contributors.
#
# Released under the MIT License, see the file LICENSE.txt.

"""Cadishi dummy IO module, useful to test and develop Capriqorn pipelines.
"""

import numpy as np
from .. import base
from .. import util


class DummyReader(base.Reader):
    """Dummy reader, generates random coordinate data on the fly.
    Intended for development/testing purposes."""
    _depends = []
    _conflicts = []

    def __init__(self, n_frames, n_objects=[], verbose=False):
        self.n_frames = n_frames
        self.n_objects = n_objects
        self.verbose = verbose
        pass

    def get_frame(self, i):
        # internally, Cadishi and Capriqorn use base.Container() instances to handle data
        frm = base.Container()
        # generate list of (Nx3) NumPy coordinate arrays
        coord_set = util.generate_random_coordinate_set(self.n_objects)
        for idx, coords in enumerate(coord_set):
            # each species carries a label
            label = "El" + str(idx)
            # add coordinates to frame at the following standardized location
            location = base.loc_coordinates + '/' + label
            frm.put_data(location, coords.astype(np.float64))
        # finally, each frame is uniquely numbered
        frm.i = i
        return frm

    def next(self):
        for i in xrange(self.n_frames):
            yield self.get_frame(i)


class DummyWriter(base.Writer):
    """Dummy trajectory writer, provides a sink for a pipeline,
    discards frame by frame.  Mainly for development purposes.
    """
    _depends = []
    _conflicts = []

    def __init__(self, source, verbose=False):
        self.src = source
        self.verb = verbose
        # ---
        self._depends.extend(super(base.Writer, self)._depends)
        self._conflicts.extend(super(base.Writer, self)._conflicts)

    def dump(self):
        for frm in self.src.next():
            if self.verb:
                print "DummyWriter.dump() : ", frm.i
            pass
