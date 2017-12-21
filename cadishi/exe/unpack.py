# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4 fileencoding=utf-8
#
# Cadishi --- CAlculation of DIStance HIstograms
#
# Copyright (c) Klaus Reuter, Juergen Koefinger
# See the file AUTHORS.rst for the full list of contributors.
#
# Released under the MIT License, see the file LICENSE.txt.

"""Extract data from HDF5 to legacy (NumPy, JSON) text data.

unpack creates a directory structure from the HDF5 file's group structure and
writes the HDF5 datasets to text files located in the correct directories.

The main() function is to be called via cli.py.
"""
from __future__ import print_function


import os
import sys
import argparse
from .. import util
from ..io import hdf5
from ..io import ascii


def configure_cli(subparsers):
    """Attach a parser (specifying command name and flags) to the argparse subparsers object."""
    parser = subparsers.add_parser('unpack', help='unpack HDF5 file')
    parser.add_argument('--force', '-f', help='write into non-empty directories', action='store_true')
    parser.add_argument('--output', '-o', type=str, help='output directory', metavar='output_directory')
    parser.add_argument('file', nargs=argparse.REMAINDER, help='HDF5 file', metavar='file.h5')
    parser.set_defaults(func=main)


def main(pargs):
    args = vars(pargs)
    file_list = args['file']
    print(util.SEP)
    try:
        assert(len(file_list) > 0)
        file_name = file_list[0]
    except:
        print(" Error: Need to specify a file to be unpacked.")
        print(util.SEP)
        sys.exit(1)
    if (pargs.output):
        output_dir = pargs.output
    else:
        output_dir = os.path.splitext(os.path.basename(file_name))[0]

    if (not pargs.force) and os.path.isdir(output_dir):
        if (len(os.listdir(output_dir)) > 0):
            print(" Error: Output directory '" + output_dir + "' is not empty.")
            print(" Use the switch '-f' to overwrite existing files.")
            print(util.SEP)
            sys.exit(1)

    if not os.path.isfile(file_name):
        print(" Error: File does not exist: " + file_name)
        print(util.SEP)
        sys.exit(1)

    reader = hdf5.H5Reader(file=file_name)
    writer = ascii.ASCIIWriter(source=reader, directory=output_dir)
    writer.dump()
    print(" Unpacked '" + file_name + "' into the directory '" + output_dir + "'.")
    print(util.SEP)
