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
from collections import OrderedDict


# --- set up and parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--size', help='specify number of coordinate tuples to be used', type=str, metavar='N1,N2,N3,...')
parser.add_argument('--bins', help='number of histogram bins', type=int, metavar='N')
parser.add_argument('--box', help='select orthorhombic or triclinic box', type=str, metavar='box')
parser.add_argument('--check-input', help='activate input check in kernels', action="store_true")
parser.add_argument('--double-precision', help='use double precision coordinates', action="store_true")
parser.add_argument('--cpu', help='run pydh CPU kernel (default)', action="store_true")
parser.add_argument('--threads', help='number of CPU threads', type=int, metavar='N')
parser.add_argument('--numa', help='use numa process pinning', action="store_true")
parser.add_argument('--gpu', help='run cudh GPU kernel (optionally on GPU N, default 0)', nargs='?',
                    const=0, type=int, metavar='N')
parser.add_argument('--thread-block-x', help='set thread block size for the first dimension of GPU algorithms 1, 2', type=int, metavar='N')
parser.add_argument('--gpu-algorithm', help='select GPU algorithm (1 advanced, 2 global, 3 simple)', type=int, metavar='N')
parser.add_argument('--histo2', help='only run mixed species histogram computation', action="store_true")
parser.add_argument('--silent', help='do not print results to stdout', action="store_true")
parser.add_argument('--sqlite', help='write results to sqlite table', action="store_true")
p_args = parser.parse_args()


run_values = OrderedDict()

gpu_id = 0
if p_args.gpu is not None:
    gpu_id = p_args.gpu
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

if (p_args.gpu_algorithm >= 0):
    run_values['gpu_algorithm'] = p_args.gpu_algorithm
else:
    run_values['gpu_algorithm'] = -1

if (p_args.thread_block_x > 0):
    run_values['thread_block_x'] = p_args.thread_block_x
else:
    run_values['thread_block_x'] = 0

if p_args.histo2:
    run_values['histo2'] = p_args.histo2
else:
    run_values['histo2'] = False

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

if p_args.silent:
    run_values['silent'] = True
else:
    run_values['silent'] = False


run_values['timestamp'] = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')

run_values['version'] = version.get_version_string()


# data types for sqlite database
output_keys = OrderedDict()
output_keys['kernel'] = 'text'
output_keys['version'] = 'text'
output_keys['timestamp'] = 'text'
output_keys['threads'] = 'integer'
output_keys['precision'] = 'text'
output_keys['size'] = 'text'
output_keys['bins'] = 'integer'
output_keys['box'] = 'text'
output_keys['check_input'] = 'text'
output_keys['gpu_algorithm'] = 'integer'
output_keys['thread_block_x'] = 'integer'
output_keys['histo2'] = 'text'
output_keys['bap'] = 'real'
output_keys['time'] = 'real'
output_keys['bapps'] = 'real'


database = 'perf.db'
if p_args.sqlite:
    import sqlite3 as sq3  # on some systems sqlite is not available by default
    if not os.path.isfile(database):
        with sq3.connect(database) as conn:
            keys = []
            for key in output_keys:
                key_sql = key + " " + str(output_keys[key])
                keys.append(key_sql)
            sql = "CREATE TABLE IF NOT EXISTS cadishi (" + ', '.join(keys) + ")"
            conn.cursor().execute(sql)
            conn.commit()


# ---


if run_values['kernel'] == "cudh":
    from cadishi.kernel import cudh
else:
    from cadishi.kernel import pydh


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
                if not run_values['histo2']:
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

t0 = time.time()
if run_values['kernel'] == "cudh":
    cudh = cudh.histograms(coords,
                           r_max,
                           run_values['bins'],
                           run_values['precision'],
                           gpu_id=gpu_id,
                           thread_block_x=run_values['thread_block_x'],
                           check_input=run_values['check_input'],
                           box=box,
                           do_histo2_only=run_values['histo2'],
                           algorithm=run_values['gpu_algorithm'])
else:
    pydh = pydh.histograms(coords,
                           r_max,
                           run_values['bins'],
                           run_values['precision'],
                           pydh_threads=run_values['threads'],
                           check_input=run_values['check_input'],
                           box=box,
                           do_histo2_only=run_values['histo2'])
t1 = time.time()
dt = (t1 - t0)

bapps = bap / dt

run_values['bap'] = bap
run_values['time'] = dt
run_values['bapps'] = bapps


if not run_values['silent']:
    vals = [x + ":" + str(run_values[x]) for x in output_keys]
    line = ' '.join(vals)
    print(line)


if p_args.sqlite:
    with sq3.connect(database) as conn:
        data_str_lst = []
        for key in output_keys:
            val = run_values[key]
            if (output_keys[key] == 'text'):
                val = util.quote(str(val))
            data_str_lst.append(str(val))
        sql = "INSERT INTO cadishi VALUES (" + ', '.join(data_str_lst) + ")"
        # print sql
        conn.cursor().execute(sql)
        conn.commit()
