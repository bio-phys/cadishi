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

"""
Example on how-to access HDF5 data saved by Cadishi or Capriqorn.
We open a data file, find out the number of frames and the frame
indices, select the last frame, pick two datasets and plot them
using matplotlib.
"""


import numpy as np
import matplotlib.pyplot as plt
from cadishi.io.hdf5 import H5Reader


file_name="histograms.h5"

# open the HDF5 file via the Cadishi HDF5 reader
reader = H5Reader(file=file_name)
# obtain information about the trajectory
ti = reader.get_trajectory_information()
# get the last available frame index
idx = ti.frame_numbers[-1]
# load the frame into memory as a base.Container() object (nested structure of
# dicts and NumPy arrays)
frm = reader.get_frame(idx)

# optional during development: explore the base.Container() object
#print frm.get_keys('/')
#print frm.get_keys('/histograms')

# access the radial distance histogram of the C atoms via a path-like string,
# default locations used by Cadishi and Capriqorn are defined in cadihi.base
radii = frm.get_data('/histograms/radii')
histo_cc = frm.get_data('/histograms/C,C')


# let us plot the histogram via matplotlib
plt.plot(radii, histo_cc)
plt.xlabel('radius')
plt.ylabel('count')
plt.title('Cadishi distance histogram of C,C')
plt.grid(True)
# save the plot to an image file
plt.savefig("histo_cc.svg")
# finally, display the plot on the screen
plt.show()
