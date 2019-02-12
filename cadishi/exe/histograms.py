# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4 fileencoding=utf-8
#
# Cadishi --- CAlculation of DIStance HIstograms
#
# Copyright (c) Klaus Reuter, Juergen Koefinger
# See the file AUTHORS.rst for the full list of contributors.
#
# Released under the MIT License, see the file LICENSE.txt.

"""Distance histogram calculation driver, task parallel version.

Performs distance histogram calculations using cudh and pydh (or dist).
The configuration is read from the parameter file histograms.{json,yaml}.

This program is not intended to be invoked directly. It is launched via cli.py
which in turn is called as the `cadishi` command via an entry_point in setup.py.
"""


from builtins import next
from builtins import str
from builtins import range
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from past.builtins import basestring
    from past.utils import old_div
import sys
import os
import math
import glob
import numpy as np
import time
import datetime
import multiprocessing
import json
import ctypes
import signal
import cProfile
import io
import pstats
import argparse
from .. import base
from .. import util
from .. import pbc
from .. import version
from ..io import hdf5
from .. import worker
from ..kernel import cudh


def configure_cli(subparsers):
    """Attach a parser (specifying command name and flags) to the argparse subparsers object."""
    parser = subparsers.add_parser('histo', help='run distance histogram calculation')
#     parser.add_argument('--input', '-i', type=str, help='input parameter file', metavar='par.yml')
    parser.add_argument('input', nargs=argparse.REMAINDER,
                        help='histograms parameter file (optional)', metavar='histograms.yaml')
    parser.set_defaults(func=main)


def check_parameters(histoparam):
    """Check and sanitize input parameters for the histogram computation.

    Together with the <histograms_template.yaml> file, this function needs
    to be extended with each new parameter that is introduced."""
    import os
    from .. import util

    myNoneType = type(None)

    for category in ['general', 'histogram', 'cpu', 'gpu', 'input', 'output']:
        if not category in histoparam:
            histoparam[category] = {}

    util.check_parameter(histoparam, 'cpu:module', basestring, 'pydh', valid_values=['pydh', 'dist'])
    util.check_parameter(histoparam, 'gpu:module', basestring, 'cudh', valid_values=['cudh'])

    # auto-detect the number of CPU workers, if not set explicitly
    util.check_parameter(histoparam, 'cpu:workers', int, -1)
    if (histoparam['cpu']['workers'] < 0):
        if 'threads' in histoparam['cpu']:
            del histoparam['cpu']['threads']  # autodetect below
        try:
            n_workers = util.get_n_cpu_sockets()
        except:
            n_workers = 1
        histoparam['cpu']['workers'] = n_workers

    # auto-detect the number of GPU workers, if not set explicitly
    util.check_parameter(histoparam, 'gpu:workers', int, -1)
    if (histoparam['gpu']['workers'] < 0 and cudh.have_c_cudh):
        histoparam['gpu']['workers'] = cudh.get_num_devices()

    if (histoparam['cpu']['workers'] == 0) and (histoparam['gpu']['workers'] == 0):
        raise ValueError("At least one worker (CPU and/or GPU) must be used")

    # now that we know the number of workers we can assign threads to the CPU workers
    if (not 'threads' in histoparam['cpu']) or (histoparam['cpu']['threads'] < 0):
        try:
            n_cores = util.get_n_cpu_cores()
        except Exception as e:
            n_cores = 1
        n_cores = n_cores - histoparam['gpu']['workers'] - 2  # 2: main task, sum task
        n_cores = old_div(n_cores, histoparam['cpu']['workers'])
        if (n_cores <= 0):
            n_cores = 1
        histoparam['cpu']['threads'] = n_cores

    util.check_parameter(histoparam, 'general:profile', bool, False)
    util.check_parameter(histoparam, 'general:numa_aware', bool, False)
    util.check_parameter(histoparam, 'general:redirect_output', bool, True)
    util.check_parameter(histoparam, 'general:verbose', bool, True)
    util.check_parameter(histoparam, 'general:queue_maxsize', int, 256)
    util.check_parameter(histoparam, 'general:queue_timeout', int, 3600)

    # define the maximum distance for the histograms
    # default '-1': set r_max to be set by the capriqorn preprocessor pipeline log
    util.check_parameter(histoparam, 'histogram:r_max', (float, int), -1)
    util.check_parameter(histoparam, 'histogram:sum', int, 1, min_value=1)
    util.check_parameter(histoparam, 'histogram:dr', float, 0.01, min_value=0.0)

    util.check_parameter(histoparam, 'cpu:check_input', bool, True)
    util.check_parameter(histoparam, 'cpu:precision', basestring, 'single', valid_values=['single', 'double'])

    util.check_parameter(histoparam, 'gpu:check_input', bool, True)
    util.check_parameter(histoparam, 'gpu:precision', basestring, 'single', valid_values=['single', 'double'])

    if not 'file' in histoparam['input']:
        raise ValueError("input:file is not set")
    else:
        util.check_parameter(histoparam, 'input:file', basestring, None, file_existence=True)
    util.check_parameter(histoparam, 'input:first', (myNoneType, int), None)
    util.check_parameter(histoparam, 'input:last', (myNoneType, int), None)
    util.check_parameter(histoparam, 'input:step', int, 1, min_value=1)
    util.check_parameter(histoparam, 'input:periodic_box', (myNoneType, list), None)

    util.check_parameter(histoparam, 'output:directory', basestring, './histograms_output/')
    util.check_parameter(histoparam, 'output:file', basestring, 'histograms.h5')
    util.check_parameter(histoparam, 'output:compression', (myNoneType, basestring),
                         None, valid_values=[None, 'gzip', 'lzf'])

    util.check_parameter(histoparam, 'output:write_h5', bool, True)
    util.check_parameter(histoparam, 'output:write_npx', bool, False)
    util.check_parameter(histoparam, 'output:write_npy', bool, False)
    util.check_parameter(histoparam, 'output:write_xyz', bool, False)

    _odir = histoparam['output']['directory']
    if (_odir[-1] != '/'):
        histoparam['output']['directory'] = _odir + '/'
    # stdout flush frequency, trigger flush after n frames.
    # Be careful on parallel file systems with many parallel jobs!
    util.check_parameter(histoparam, 'output:flush_interval', int, 100, min_value=1)


# global list of all the worker processes, needed by the shutdown signal handler
mp_all_workers_list = []


def unexpectedShutdownHandler(signum, frame):
    """Singnal handler, to catch SIGUSR1 sent by child processes, and SIGTERM."""
    print(util.SEP)
    print(" %s Shutdown signal received!" % util.timeStamp(dateAndTime=True))
    print(" Killing all child processes ...")
    for mp_worker in mp_all_workers_list:
        mp_worker.terminate()
    print(" Killing master process.  Goodbye.")
    print(util.SEP)
    os.kill(os.getpid(), signal.SIGTERM)
    time.sleep(3.0)
    os.kill(os.getpid(), signal.SIGKILL)


def main(argparse_args):
    print_it = util.PrintWrapper()

    print(util.SEP)

    if (argparse_args.input):
        parameter_file = argparse_args.input[0]
    else:
        if util.have_yaml:
            parameter_file = 'histograms.yaml'
        else:
            parameter_file = 'histograms.json'

    if not os.path.exists(parameter_file):
        print(" Could not find histogram input file '" + parameter_file + "'.")
        print(" Run `" + util.get_executable_name() + " example` to generate an example input file.")
        print(util.SEP)
        exit(1)

    histoparam = {}
    try:
        histoparam = util.load_parameter_file(parameter_file)
    except:
        print(" Error: Could not read input file '" + parameter_file + "'.")
        exit(1)

    try:
        template_file = os.path.abspath(os.path.dirname(os.path.abspath(__file__))
                                        + "/../data/histograms_template.yaml")
        valid_parameters = util.load_parameter_file(template_file)
        util.check_parameter_labels(histoparam, valid_parameters)
        check_parameters(histoparam)
    except Exception as e:
        print(" Error: " + e.message + ".")
        print(util.SEP)
        sys.exit(1)

    # --- END OF CONFIGURATION SECTION ---

    util.md(histoparam['output']['directory'])

    reader = hdf5.H5Reader(file=histoparam['input']['file'])
    ti = reader.get_trajectory_information()
    del reader

    # --- correct input parameters based on trajectory ---
    if (histoparam['input']['first'] == None):
        histoparam['input']['first'] = ti.frame_numbers[0]
    if (histoparam['input']['last'] == None):
        histoparam['input']['last'] = ti.frame_numbers[-1]
    if (histoparam['input']['step'] == None):
        histoparam['input']['step'] = 1
    assert (histoparam['input']['first'] >= ti.frame_numbers[0])
    assert (histoparam['input']['last'] <= ti.frame_numbers[-1])
    assert (histoparam['input']['first'] <= histoparam['input']['last'])

    n_frames = len(range(histoparam['input']['first'], \
                         histoparam['input']['last'] + 1, \
                         histoparam['input']['step']))

    if histoparam['histogram']['r_max'] < 0:
        r_max = ti.get_pipeline_parameter('r_max')
    else:
        r_max = histoparam['histogram']['r_max']

    dr = histoparam['histogram']['dr']
    nbins = int(math.ceil(old_div(r_max, dr)))

    print(version.get_printable_version_string())
    print(util.SEP)
    print(" parameter file:       " + parameter_file)
    print(" trajectory file:      " + str(histoparam['input']['file']))
    print(" output directory:     " + histoparam['output']['directory'])
    print(" r_max:                " + str(r_max))
    print(" nbins:                " + str(nbins))
    print(" cpu workers:          " + str(histoparam['cpu']['workers']))
    if (histoparam['cpu']['workers'] > 0):
        print(" cpu threads:          " + str(histoparam['cpu']['threads']))
    print(" gpu workers:          " + str(histoparam['gpu']['workers']))
    print(util.SEP)

    elements = ti.species

    nEl = len(elements)
    header_str = "# "
    counter = 0
    for i in range(nEl):
        for j in range(i, nEl):
            header_str += "%s,%s " % (elements[i], elements[j])
            counter += 1
    header_str = header_str[:-1] + "\n"

    hfp = open(histoparam['output']['directory'] + "header.dat", 'w')
    hfp.write(header_str)
    hfp.close()

    # initialize a time mark for relative timing information
    t0 = time.time()
    print(" %s proceeding to distance histogram computation" % util.timeStamp(dateAndTime=True))

    # ------ set up the multiprocessing environment ------
    # use blocking queues for job handling and synchronization
    task_queue = multiprocessing.JoinableQueue(histoparam['general']['queue_maxsize'])
    result_queue = multiprocessing.JoinableQueue(histoparam['general']['queue_maxsize'])
    # set up and launch worker processes
    pool = []
    # set up processes for the calculation of the individual histograms
    for i in range(histoparam['cpu']['workers']):
        mp_worker = multiprocessing.Process(target=worker.compute,
                                            args=(histoparam, i, 'cpu', task_queue, result_queue, r_max, nbins, t0))
        pool.append(mp_worker)
    for i in range(histoparam['gpu']['workers']):
        mp_worker = multiprocessing.Process(target=worker.compute,
                                            args=(histoparam, i, 'gpu', task_queue, result_queue, r_max, nbins, t0))
        pool.append(mp_worker)
    n_workers = len(pool)

    # set up process for summing-up and writing the histograms
    sum_worker = multiprocessing.Process(target=worker.sum,
                                         args=(histoparam, result_queue, nEl, nbins, dr, header_str, t0, n_frames, n_workers))

    # build list of all child processes to be used by the signal handler
    mp_all_workers_list = pool + [sum_worker]


    if (histoparam['general']['verbose']):
        print(util.SEP)
        print(" %s spawning processes: %d CPU worker, %d GPU worker, 1 writer" % \
            (util.timeStamp(t0=t0), histoparam['cpu']['workers'], histoparam['gpu']['workers']))
        sys.stdout.flush()


    for mp_worker in mp_all_workers_list:
        mp_worker.start()

    for mp_worker in mp_all_workers_list:
        assert(mp_worker.is_alive())

    # install the shutdown handler for SIGUSR1 events received from child processes
    signal.signal(signal.SIGUSR1, unexpectedShutdownHandler)
    # install the shutdown handler for Ctrl-c
    if not histoparam['general']['profile']:
        signal.signal(signal.SIGINT, unexpectedShutdownHandler)
    # ------

    # --- save histogram parameters (that may have been altered/corrected within the program)
    util.save_yaml(histoparam, histoparam['output']['directory'] + "histograms.yaml")

    # --- save particle numbers of each frame to calculate densities and fluctuations
    nr_part_fp = open(histoparam['output']['directory'] + "nrPart.%d.%d.dat" %
                      (histoparam['input']['first'], histoparam['input']['last']), 'w')
    nr_header = "# " + "%s " * len(elements) % tuple(elements) + "\n"
    nr_part_fp.write(nr_header)

    termination_msg = "done"

    reader = hdf5.H5Reader(file=histoparam['input']['file'],
                           first=histoparam['input']['first'],
                           last=histoparam['input']['last'],
                           step=histoparam['input']['step'],
                           verbose=False)
    # --- fetch data from reader and put it into the task queue
    for frm in next(reader):
        assert isinstance(frm, base.Container)

        # (1) --- create a list containing per-species numpy arrays with coordinate triples
        particleNrs = []
        for el in elements:
            coord_set = frm.get_data(base.loc_coordinates + '/' + el)
            n_part = coord_set.shape[0]
            particleNrs.append(n_part)
            # ---
            frm.put_data(base.loc_nr_particles + '/' + el, np.array([n_part]))

        # (2) --- create a list containing all the coordinate triples
        # in a linearly concatenated manner for xyz output
        # TODO: check if the sorting by species is OK
        # WARNING: severely degrades performance, needs C implementation
        if histoparam['output']['write_xyz']:
            volCrds = []
            volSpecies = []
            for el in elements:
                coord_set = frm.get_data(base.loc_coordinates + '/' + el)
                for ij in range(coord_set.shape[0]):
                    triple = coord_set[ij, :]
                    volCrds.append(triple)
                    volSpecies.append(el)
            util.write_xyzFile(volCrds, volSpecies, histoparam['output']['directory'] +
                               "volume.fr%d.xyz" % frm.i)
            volCrds = []
            volSpecies = []

        # --- handle miscellaneous information attached to the frame ---

        # assure correct handling of the periodic boxes
        if (frm.get_geometry() is not None):
            frm.put_data(base.loc_dimensions, [])
            msg = "geometry filter detected, disabling box."
        else:
            if histoparam['input']['periodic_box'] is None:
                if frm.contains_key(base.loc_dimensions):
                    msg = "using box information provided by frame."
                    pass
                else:
                    msg = "frame does not provide any box information, disabling box."
                    frm.put_data(base.loc_dimensions, [])
            else:
                periodic_box = histoparam['input']['periodic_box']
                msg = "using periodic box information provided by parameter file."
                frm.put_data(base.loc_dimensions, periodic_box)
        print_it.once("periodic box", msg, time_stamp=util.timeStamp(t0=t0))

        dimensions = frm.get_data(base.loc_dimensions)
        box_volume = pbc.get_box_volume(dimensions)
        if box_volume is not None:
            frm.put_data(base.loc_volumes, {'box': box_volume})

        if not frm.contains_key(base.loc_histogram_scale_factors):
            frm.put_data(base.loc_histogram_scale_factors, [])

        if not frm.contains_key(base.loc_histogram_mask):
            frm.put_data(base.loc_histogram_mask, [])

        bap = 0.0  # billion atom pairs
        for i in range(nEl):
            for j in range(i, nEl):
                n_at1 = particleNrs[i]
                n_at2 = particleNrs[j]
                if (i == j):
                    bap += old_div(float(old_div((n_at1 * (n_at1 - 1)), 2)), 1.e9)
                else:
                    bap += old_div(float(n_at1 * n_at2), 1.e9)

        # --- write out particle numbers
        nr_part_fp.write("%d " * (nEl + 1) % tuple([frm.i] + particleNrs) + "\n")

        if (histoparam['general']['verbose']):
            print(" %s enqueueing frame %d ..." % (util.timeStamp(t0=t0), frm.i))
            if (frm.i % histoparam['output']['flush_interval'] == 0):
                sys.stdout.flush()

        work_package = (frm, bap)
        task_queue.put(work_package, histoparam['general']['queue_timeout'])

        # cadishi cat be stopped by creating a file "stop" in the same directory
        if os.path.isfile("stop"):
            termination_msg = "stop"
            if (histoparam['general']['verbose']):
                print(" %s `stop' file encountered" % util.timeStamp(t0=t0))
                sys.stdout.flush()
            break

    nr_part_fp.close()

    # Trigger the workers to shut down in a controlled way by sending "None".
    for mp_worker in pool:
        task_queue.put(termination_msg, histoparam['general']['queue_timeout'])

    if (histoparam['general']['verbose']):
        print(" %s waiting for worker processes ..." % util.timeStamp(t0=t0))
        sys.stdout.flush()

    task_queue.join()
    if (histoparam['general']['verbose']):
        print(" %s joined task_queue" % util.timeStamp(t0=t0))
        sys.stdout.flush()

    result_queue.join()
    if (histoparam['general']['verbose']):
        print(" %s joined result_queue" % util.timeStamp(t0=t0))
        sys.stdout.flush()

    t1 = time.time()
    wallclock = float(t1 - t0)
    ntotal = float(histoparam['input']['last'] + 1 - histoparam['input']['first'])
    fps = old_div(ntotal, wallclock)

    # Save preprocessor meta information (that was fetched from the HDF5 input file)
    # to a JSON text file for potential evaluation by the postprocessor.
    with open(histoparam['output']['directory'] + "preprocessor_log.json", 'w') as fp:
        json.dump(ti.pipeline_log, fp, indent=4, sort_keys=True)

    print(util.SEP)
    print(" %s distance histogram computation finished" % util.timeStamp(dateAndTime=True))
    print("   frames:             %d" % ntotal)
    print("   wallclock time:     %.3f" % wallclock)
    print("   frames per second:  %.3f" % fps)
    print("   output directory:   " + histoparam['output']['directory'])

    sys.stdout.flush()

    # Join all worker processes which should have already terminated themselves.
    # Update: terminate() should work as well fine here, but join() is cleaner.
    for mp_worker in pool:
        mp_worker.join()
    # Important: To guarantee proper writeout of the HDF5 file we must
    # join the process (i.e. wait until it terminates itself)!
    # Using terminate() here does cause HDF5 errors!
    sum_worker.join()
    print(util.SEP)
    sys.exit(0)
