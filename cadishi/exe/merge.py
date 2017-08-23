# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4 fileencoding=utf-8
#
# Cadishi --- CAlculation of DIStance HIstograms
#
# Copyright (c) Klaus Reuter, Juergen Koefinger
# See the file AUTHORS.rst for the full list of contributors.
#
# Released under the MIT License, see the file LICENSE.txt.

"""Merge several HDF5 files into a single one.

The main() function is to be called via cli.py.
"""


import os
import sys
import argparse
from .. import util
from ..io import hdf5


def configure_cli(subparsers):
    """Attach a parser (specifying command name and flags) to the argparse subparsers object."""
    parser = subparsers.add_parser('merge', help='merge HDF5 files')
    parser.add_argument('--force', '-f', help='overwrite an existing file', action='store_true')
    parser.add_argument('--output', '-o', type=str, help='output file', metavar='file.h5')
    parser.add_argument('--compression', '-c', type=str, help='output file', metavar='none|gzip|lzf')
    parser.add_argument('files', nargs=argparse.REMAINDER, help='HDF5 files', metavar='file1.h5 file2.h5 ...')
    parser.set_defaults(func=main)


def main(pargs):
    if (pargs.output):
        output_file = pargs.output
    else:
        output_file = "merged.h5"
    if (pargs.compression):
        compression = pargs.compression
    else:
        compression = "lzf"
    args = vars(pargs)
    file_list = args['files']

    print(util.SEP)
    try:
        if (compression.lower() == "none"):
            compression = None
        assert(compression in hdf5.H5Writer.valid_compression)
    except:
        print(" Error: Invalid compression method requested: " + compression)
        print(util.SEP)
        sys.exit(1)

    if (not pargs.force) and os.path.isfile(output_file):
        print(" Error: Output file '" + output_file + "' exists.")
        print(" Use the switch '--force' or '-f' to overwrite it.")
        print(util.SEP)
        sys.exit(1)

    for file_name in file_list:
        if not os.path.isfile(file_name):
            print(" Error: File does not exist: " + file_name)
            print(util.SEP)
            sys.exit(1)

    util.md(output_file)

    if (len(file_list) > 0):
        reader = hdf5.H5Reader(file=file_list)
        writer = hdf5.H5Writer(source=reader, compression=compression, file=output_file)
        print(" Merged files into '" + output_file + "'.")
        writer.dump()
        print(util.SEP)
    else:
        print(" Error: At least one file to be merged must be specified.")
        print(util.SEP)
        sys.exit(1)
