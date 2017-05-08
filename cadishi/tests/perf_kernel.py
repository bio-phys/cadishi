#!/usr/bin/env python2.7

"""Measure the performance of cudh and pydh using conveniently.

This file is part of the Cadishi package.  See README.rst,
LICENSE.txt, and the documentation for details.
"""

__author__ = "Klaus Reuter"
__copyright__ = "Copyright (C) 2015-2016 Klaus Reuter"
__license__ = "license_tba"


import os
import sys
import numpy as np
import time
import datetime
import math
import yaml
import argparse
from six.moves import range
from cadishi import util
from cadishi import version


# --- set up and parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--cpu', help='run pydh CPU kernel (default)', action="store_true")
parser.add_argument('--gpu', help='run cudh GPU kernel instead of pydh CPU kernel', action="store_true")
parser.add_argument('--size', help='specify number of coordinate tuples to be used', type=str, metavar='N1,N2,N3,...')
parser.add_argument('--bins', help='number of histogram bins', type=int, metavar='bins')
parser.add_argument('--box', help='select orthorhombic or triclinic box', type=str, metavar='box')
parser.add_argument('--threads', help='number of CPU threads', type=int, metavar='threads')
parser.add_argument('--check-input', help='activate input check in kernels', action="store_true")
parser.add_argument('--double-precision', help='use double precision coordinates', action="store_true")
parser.add_argument('--numa', help='use numa process pinning', action="store_true")
parser.add_argument('--timestamp', help='add timestamp to output file name', action="store_true")
parser.add_argument('--output', help='specify output file name', type=str, metavar='file_name')
parser.add_argument('--verbose', help='print results to stdout', action="store_true")
p_args = parser.parse_args()


run_values = {}


if p_args.gpu:
    run_values['kernel'] = "cudh"
else:
    run_values['kernel'] = "pydh"

if p_args.size:
    run_values['size'] = [int(x) for x in (p_args.size).split(',')]
else:
    run_values['size'] = [50000, 50000]

if p_args.bins:
    run_values['bins'] = p_args.bins
else:
    run_values['bins'] = 8000

if p_args.box:
    run_values['box'] = p_args.box
else:
    run_values['box'] = "None"

if p_args.threads:
    run_values['threads'] = p_args.threads
else:
    run_values['threads'] = 1

if p_args.check_input:
    run_values['check_input'] = p_args.check_input
else:
    run_values['check_input'] = False

if p_args.double_precision:
    run_values['precision'] = "double"
else:
    run_values['precision'] = "single"

if p_args.numa:
    run_values['numa'] = True
else:
    run_values['numa'] = False

if p_args.verbose:
    run_values['verbose'] = True
else:
    run_values['verbose'] = False

if p_args.timestamp:
    t = time.time()
    s = datetime.datetime.fromtimestamp(t).strftime('%Y-%m-%d-%H-%M-%S')
    output_suffix = "_" + s + ".yml"
else:
    output_suffix = ".yml"

if p_args.output:
    run_values['output'] = p_args.output
else:
    output_prefix = "perf/"
    run_values['output'] = output_prefix + "perf_kernel" + output_suffix

run_values['version'] = version


# ---


if run_values['kernel'] == "cudh":
    from cadishi.kernel import cudh
else:
    from cadishi.kernel import pydh


# def get_random_coordinate_set(n_atoms=[512, 1024, 2048]):
#     """return random coordinate sets in a unit box"""
#     coords = []
#     for n in n_atoms:
#         c = np.random.rand(n, 3)
#         coords.append(c)
#     return coords


def get_bap():
    """return the problem size in billion atom pairs"""
    n_atoms = run_values['size']
    n_el = len(n_atoms)
    bap = 0.0
    for i in range(n_el):
        for j in range(i, n_el):
            if (i != j):
                bap += float(n_atoms[i] * n_atoms[j]) / 1.e9
            else:
                bap += float((n_atoms[j] * (n_atoms[j] - 1)) / 2) / 1.e9
    return bap


# ------------------------------------------------------------------------------


if run_values['numa']:
    numa_topology = util.get_numa_domains()
    util.set_numa_domain(0, numa_topology)

coords = util.generate_random_coordinate_set(run_values['size'])

n_bins = run_values['bins']

if run_values['box'].startswith('t'):
    # triclinic dummy box
    box = np.asarray([0.30, 0.32, 0.34, 60., 60., 90.])
elif run_values['box'].startswith('o'):
    # orthorhombic dummy box
    box = np.zeros((3, 3))
    box[0][0] = 0.66
    box[1][1] = 0.68
    box[2][2] = 0.70
else:
    # no box
    box = []

# --- r_max, valid when coordinates are in a unit box
r_max = math.sqrt(3.0)

bap = get_bap()

if run_values['verbose']:
    print("Running " + run_values['kernel'] + " ...")

t0 = time.time()
if run_values['kernel'] == "cudh":
    cudh = cudh.histograms(coords,
                           r_max,
                           run_values['bins'],
                           run_values['precision'],
                           check_input=run_values['check_input'],
                           box=box)
else:
    pydh = pydh.histograms(coords,
                           r_max,
                           run_values['bins'],
                           run_values['precision'],
                           pydh_threads=run_values['threads'],
                           check_input=run_values['check_input'],
                           box=box)
t1 = time.time()
dt = (t1 - t0)

bapps = bap / dt

run_values['bap'] = bap
run_values['time'] = dt
run_values['bapps'] = bapps

if run_values['verbose']:
    keys = ['size', 'bins', 'box', 'kernel', 'precision', 'threads',
            'check_input', 'bap', 'time', 'bapps']
    vals = [x + ":" + str(run_values[x]) for x in keys]
    line = ' '.join(vals)
    print(line)

util.md(run_values['output'])

with open(run_values['output'], 'w') as fp:
    del(run_values['output'])
    yaml.dump(run_values, fp, default_flow_style=False)
