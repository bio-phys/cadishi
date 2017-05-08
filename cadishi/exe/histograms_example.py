#!/usr/bin/env python2.7
# vim:fileencoding=utf-8
"""Parameter file generator for cadishi.

Generates a parameter file in YAML or JSON format for `cadishi histo`,
demonstrating the full spectrum of options. After generating the example
parameter file, the user may edit the parameter file, adapt the switches and
values to his/her needs, before running the calculation via `cadishi histo`.

By default, the parameter file generator creates an input file based on the
testcase that is included with the Cadishi package.

This program is not intended to be invoked directly. It is launched via cli.py
which in turn is called as the `cadishi` command via an entry_point in setup.py.
"""
# This file is part of the Cadishi package.  See README.rst,
# LICENSE.txt, and the documentation for details.


import os
import sys
from .. import util


def configure_cli(subparsers):
    """Attach a parser (specifying command name and flags) to the argparse subparsers object."""
    parser = subparsers.add_parser('example', help='generate example parameter file')
    parser.add_argument('--expert', '-e', help='show expert parameters', action='store_true')
    parser.set_defaults(func=main)


def main(argparse_args):
    if argparse_args.expert:
        expert_flag = True
    else:
        expert_flag = False

    print(util.SEP)

    template_file = os.path.abspath(os.path.dirname(os.path.abspath(__file__))
                                    + "/../data/histograms_template.yaml")

    yaml_file = "histograms.yaml"

    trajectory_file = "./preprocessor_output/trajectory.h5"
    if not os.path.exists("preprocessor.yaml"):
        trajectory_file = util.testcase() + "trajectory.h5"

    with open(template_file, 'r') as fp_in, open(yaml_file, 'w') as fp_out:
        template_lines = fp_in.readlines()
        for line in template_lines:
            # apply substitutions
            substring = "__TRAJECTORY_FILE__"
            if substring in line:
                line = line.replace(substring, trajectory_file)
            # skip expert parameters if not requested explicitly
            substring = "expert"
            if (not expert_flag) and (substring in line):
                continue
            fp_out.write(line)

    print(" Histogram calculation example input file was written to <" + yaml_file + ">.")
    print(" Inspect and adapt this file to your needs.  Finally, run the histogram")
    print(" calculation using the command `" + os.path.basename(sys.argv[0]) + " histo`.")
    print(util.SEP)
