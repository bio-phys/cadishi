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

"""Convert data from MD simulation codes to a Cadishi-compatible HDF5 input file.
"""


from __future__ import print_function
import os
import sys
import argparse
from cadishi import util
from cadishi.io import md
from cadishi.io import hdf5


def cliarg_get():
    """Define and handle command line arguments, return them as argparse args object."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pdb", help="specify pdb file", type=str, metavar="pdb_file", required=True)
    parser.add_argument("-t", "--trajectory", help="specify trajectory file", type=str, metavar="trajectory_file", required=True)
    parser.add_argument("-a", "--alias", help="specify alias file", type=str, metavar="alias_file", required=True)
    parser.add_argument("-o", "--output", help="specify output file", type=str, metavar="output_file", default="trajectory.h5")
    args = parser.parse_args()
    return args


def main(args):
    pdb_file = args.pdb
    trajectory_file = args.trajectory
    alias_file = args.alias
    h5_file = args.output

    verbose = True
    print(util.SEP)
    reader = md.MDReader(pdb_file=pdb_file, alias_file=alias_file, trajectory_file=trajectory_file, verbose=verbose)
    writer = hdf5.H5Writer(source=reader, file=h5_file, compression="lzf", verbose=verbose)
    util.md(h5_file)
    writer.dump()
    print(" Created trajectory file <" + h5_file + ">.")
    print(util.SEP)


if __name__ == "__main__":
    args = cliarg_get()
    main(args)
