#!/usr/bin/env python2.7

# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4 fileencoding=utf-8
#
# Cadishi --- CAlculation of DIStance HIstograms
#
# Copyright (c) Klaus Reuter, Juergen Koefinger
# See the file AUTHORS.rst for the full list of contributors.
#
# Released under the MIT License, see the file LICENSE.txt.

"""Example code demonstrating how-to feed coordinate data into Cadishi."""

import numpy as np
from cadishi import base
from cadishi import util
from cadishi.io import hdf5


class CustomReader(base.Reader):
    """A reader example, simply generates random coordinate data on the fly.
    This class can easily be extended with file IO to read real data into
    Cadishi.
    """
    _depends = []
    _conflicts = []

    def __init__(self, n_frames, n_objects=[], verbose=False):
        """Set internal parameters at instantiation."""
        self.n_frames = n_frames
        self.n_objects = n_objects
        self.verbose = verbose

    def get_frame(self, i):
        """Create coordinate data sets for a single frame."""
        # Cadishi and Capriqorn use base.Container() instances to handle data
        frm = base.Container()
        # Generate list of (Nx3) NumPy coordinate arrays.
        # NOTE: This is the location where you would want to feed your own data in!
        # We use random data in this simple example.
        coord_set = util.generate_random_coordinate_set(self.n_objects)
        for idx, coords in enumerate(coord_set):
            # Each species carries a label.
            # NOTE: You need to take care about labels fitting to your data!
            label = "species_" + str(idx)
            # add coordinates to frame at the following standardized location
            location = base.loc_coordinates + '/' + label
            frm.put_data(location, coords.astype(np.float64))
        # finally, each frame is uniquely numbered
        frm.i = i
        return frm

    def next(self):
        """Generator function, yields frame by frame. Called by subsequent
        pipeline functions."""
        for i in xrange(self.n_frames):
            yield self.get_frame(i)


# parameters for CustomReader
n_frames = 10
ensemble_size = [1024, 2048, 8192]
# HDF5 output file
h5_file_name = "trajectory.h5"

# set up a Python generator pipeline consisting of a reader and a writer
reader = CustomReader(n_frames=n_frames, n_objects=ensemble_size)
# reader serves as the source for the HDF5 writer below
writer = hdf5.H5Writer(source=reader, file=h5_file_name, compression=None, verbose=True)
# start the pipeline by calling the dump() function of the writer
writer.dump()
