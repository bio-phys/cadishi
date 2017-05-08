#!/usr/bin/env python2.7
# vim:fileencoding=utf-8
"""Check parameter file if it is valid for Cadishi."""
# This file is part of the Cadishi package.  See README.rst,
# LICENSE.txt, and the documentation for details.


import os
import sys
import argparse
from .. import util
from . import histograms


def configure_cli(subparsers):
    """Attach a parser (specifying command name and flags) to the argparse subparsers object."""
    parser = subparsers.add_parser('check', help='check validiy of parameter file')
    parser.add_argument('--dump', '-d', help='write checked parameter set to file', action='store_true')
    parser.add_argument('file', nargs=argparse.REMAINDER, help='parameter file to be checked', metavar='histograms.yml')
    parser.set_defaults(func=main)


def main(pargs):
    args = vars(pargs)
    file_list = args['file']

    print(util.SEP)
    template_file = os.path.abspath(os.path.dirname(os.path.abspath(__file__))
                                    + "/../data/histograms_template.yaml")

    if (len(file_list) > 0):
        file = file_list[0]
    else:
        file = "histograms.yaml"

    try:
        valid_parameters = util.load_parameter_file(template_file)
        pending_parameters = util.load_parameter_file(file)
        print(" Checking file '" + file + "' ...")
        # high-level checks if valid keys are used in the parameter file
        util.check_parameter_labels(pending_parameters, valid_parameters)
        # low-level checks if values are OK, files exist, etc.
        histograms_par.check_parameters(pending_parameters)
    except Exception as e:
        print(" Error: " + e.message + ".")
        print(util.SEP)
        sys.exit(1)
    else:
        if pargs.dump:
            file_base, file_ext = os.path.splitext(file)
            file_dump = file_base + "_checked" + file_ext
            util.save_yaml(pending_parameters, file_dump)
            print(" Wrote file '" + file_dump + "'.")
        print(" OK!")
        print(util.SEP)
