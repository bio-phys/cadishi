# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4 fileencoding=utf-8
#
# Cadishi --- CAlculation of DIStance HIstograms
#
# Copyright (c) Klaus Reuter, Juergen Koefinger
# See the file AUTHORS.rst for the full list of contributors.
#
# Released under the MIT License, see the file LICENSE.txt.

"""Generate a HDF5 input data file compatible with the `cadishi histo`
histograms calculation.

This program is not intended to be invoked directly. It is launched via cli.py
which in turn is called as the `cadishi` command via an entry_point in setup.py.
"""
from __future__ import print_function


import os
import sys
from .. import util
from ..io import hdf5
from ..io import dummy


# default_h5file = "preprocessor_output/trajectory.h5"
default_h5file = "random.h5"


def configure_cli(subparsers):
    """Attach a parser (specifying command name and flags) to the argparse subparsers object."""
    parser = subparsers.add_parser('random', help='generate a input data file with random coordinates')
    parser.add_argument('--size', '-s', help='specify the number of objects per species',
                        type=str, metavar='N1,N2,N3,...')
    parser.add_argument('--frames', '-f', help='number of frames', type=int, metavar='n_frames')
    parser.add_argument('--output', '-o', help='output path and file name', type=str, metavar=default_h5file)
    parser.set_defaults(func=main)


def main(pargs):
    if (pargs.size):
        print(pargs.size)
        size = [int(x) for x in (pargs.size).split(',')]
    else:
        size = [512, 768, 1024]

    if (pargs.frames):
        n_frames = int(pargs.frames)
    else:
        n_frames = 10

    if (pargs.output):
        h5file = pargs.output
    else:
        h5file = default_h5file
    util.md(h5file)

    reader = dummy.DummyReader(n_frames=n_frames, n_objects=size)
    # given that we have random float data compression is not beneficial here
    writer = hdf5.H5Writer(source=reader, file=h5file)
    writer.dump()
    print(util.SEP)
    print(" Created random trajectory file <" + h5file + ">.")
    print(" Next, run `" + util.get_executable_name() + " example` to generate a parameter file.")
    print(util.SEP)
