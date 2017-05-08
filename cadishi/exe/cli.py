#!/usr/bin/env python2.7
# vim:fileencoding=utf-8
"""Cadishi main executable.  Uses argparse to call further code.

This program is not intended to be called directly, rather
a user executable `cadishi` is created via an entry_point in setup.py.
"""
# This file is part of the Cadishi package.  See README.rst,
# LICENSE.txt, and the documentation for details.

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

    parser = argparse.ArgumentParser()
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
