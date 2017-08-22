# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4 fileencoding=utf-8
#
# Cadishi --- CAlculation of DIStance HIstograms
#
# Copyright (c) Klaus Reuter, Juergen Koefinger
# See the file AUTHORS.rst for the full list of contributors.
#
# Released under the MIT License, see the file LICENSE.txt.

"""Cadishi main executable.  Uses argparse to call further code.

This program is not intended to be called directly, rather
a user executable `cadishi` is created via an entry_point in setup.py.
"""


import sys
import argparse


def parse_args():
    """"Set up the cadishi command line interface using argparse.

    Individual cadishi commands and their arguments are set up
    next to their implementation via the configure_cli() functions.
    """
    from . import histograms
    from . import histograms_example
    from . import check_parameter_file
    from . import random_trajectory
    from . import merge
    from . import unpack
    from .. import version

    version_string = "Cadishi " + version.get_version_string()
    try:
        from .. import githash
    except:
        pass
    else:
        version_string += " (git: " + githash.human_readable + ")"

    parser = argparse.ArgumentParser()

    parser.add_argument('-v', '--version', help='print version information',
                        action='version', version=version_string)

    subparsers = parser.add_subparsers(help='Commands')

    histograms.configure_cli(subparsers)
    histograms_example.configure_cli(subparsers)
    merge.configure_cli(subparsers)
    unpack.configure_cli(subparsers)
    # --- the following two commands are considered "secret", argparse.SUPPRESS does unfortunately not work ---
    if ('check' in '\t'.join(sys.argv)):
        check_parameter_file.configure_cli(subparsers)
    if ('random' in '\t'.join(sys.argv)):
        random_trajectory.configure_cli(subparsers)

    return parser.parse_args()


def main():
    args = parse_args()
    args.func(args)
