# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4 fileencoding=utf-8
#
# Cadishi --- CAlculation of DIStance HIstograms
#
# Copyright (c) Klaus Reuter, Juergen Koefinger
# See the file AUTHORS.rst for the full list of contributors.
#
# Released under the MIT License, see the file LICENSE.txt.

"""Functions to be used as multiprocessing-workers by the <histograms.py>
executable.
"""
from __future__ import print_function
from __future__ import division


from future import standard_library
standard_library.install_aliases()
from builtins import str
from builtins import range
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
from . import base
from . import util
from . import pbc
from . import version
from .io import hdf5


def _cProfile_Exit(signum, stack):
    """Signal handler to dump profile data when a child process receives SIGTERM."""
    global cProfile_handle
    cProfile_handle.disable()
    s = io.StringIO()
    # sortby = 'cumulative'
    sortby = 'tottime'
    ps = pstats.Stats(cProfile_handle, stream=s).sort_stats(sortby)
    ps.print_stats(10)
    print(util.SEP)
    print(" Information generated by cProfile:")
    print(s.getvalue())
    print(util.SEP)
    sys.exit()


def compute(histoparam, worker_id, worker_type, taskQueue, resultQueue, r_max, n_bins, t0):
    """Compute-worker wrapper to handle output redirection, numa pinning, and profiling.

    To be used as the entry function for a multiprocessing subprocess.

    Calls the _compute() function which does the real work."""
    if (histoparam['general']['redirect_output']):
        util.redirectOutput("%shisto_%s_worker_%02d.log" % (histoparam['output']['directory'], worker_type, worker_id))
    if (histoparam['general']['numa_aware']):
        n_numa_domains = len(numa_topology)
        if (n_numa_domains > 0):
            numa_id = worker_id % n_numa_domains
            util.set_numa_domain(numa_id, numa_topology)
    if (histoparam['general']['profile']):
        global cProfile_handle
        cProfile_handle = cProfile.Profile()
        signal.signal(signal.SIGTERM, _cProfile_Exit)
        cProfile_handle.enable()
    # ---
    _compute(histoparam, worker_id, worker_type, taskQueue, resultQueue, r_max, n_bins, t0)


def _compute(histoparam, worker_id, worker_type, taskQueue, resultQueue, r_max, n_bins, t0):
    """Histogram-computation worker, running the pydh and cudh kernels.

    Pulls work packages from the taskQueue, puts results into the resultQueue.

    To be called via the compute() wrapper.
    """

    if (worker_type == "cpu"):
        if (histoparam['cpu']['module'] == 'pydh'):
            from .kernel import pydh
            if not pydh.have_c_pydh:
                from .kernel import common
                raise RuntimeError(common.import_pydh_error_msg)
        elif (histoparam['cpu']['module'] == 'dist'):
            from .kernel import dist
        else:
            raise RuntimeError("unsupported CPU histogram kernel requested: " + str(histoparam['cpu']['module']))
    elif (worker_type == "gpu"):
        from .kernel import cudh
        if not cudh.have_c_cudh:
            from .kernel import common
            raise RuntimeError(common.import_cudh_error_msg)
        if (cudh.get_num_cuda_devices() == 0):
            raise RuntimeError("no usable CUDA-enabled GPU detected")
    else:
        raise RuntimeError("unsupported worker type requested: " + str(worker_type))

    worker_str = "%s worker %02d" % (worker_type, worker_id)
    if (histoparam['general']['verbose']):
        print(util.SEP)
        if (worker_type == "cpu"):
            threads_str = "(%d threads)" % histoparam['cpu']['threads']
        else:
            threads_str = ""
        print(" %s %s: worker started %s" % (util.timeStamp(dateAndTime=True), worker_str, threads_str))
        print(util.SEP)
    notify_master = False
    icount = 0
    termination_msg = ""
    t1 = time.time()

    try:
        while True:
            work_item = taskQueue.get()
            if work_item in ["done", "stop"]:
                resultQueue.put(work_item)
                taskQueue.task_done()
                termination_msg = "(" + work_item + ")"
                break
            else:
                t2 = time.time()
                wait_time = t2 - t1
                #
                frm = work_item[0]
                species_Crds = []
                for el in frm.get_keys(base.loc_coordinates):
                    coord_set = frm.get_data(base.loc_coordinates + '/' + el)
                    species_Crds.append(coord_set)
                bap = work_item[1]
                #
                histogram_scale_factors = frm.get_data(base.loc_histogram_scale_factors)
                histogram_mask = frm.get_data(base.loc_histogram_mask)
                periodic_box = frm.get_data(base.loc_dimensions)
                #
                if (histoparam['general']['verbose']):
                    print(" %s %s: processing frame %d ..." % (util.timeStamp(t0=t0), worker_str, frm.i))
                #
                try:
                    if (worker_type == "cpu"):
                        if (histoparam['cpu']['module'] == 'pydh'):
                            histograms = pydh.histograms(species_Crds, r_max, n_bins,
                                                         histoparam['cpu']['precision'],
                                                         histoparam['cpu']['threads'],
                                                         scale_factors=histogram_scale_factors,
                                                         mask_array=histogram_mask,
                                                         check_input=histoparam['cpu']['check_input'],
                                                         box=periodic_box)
                        elif (histoparam['cpu']['module'] == 'dist'):
                            histograms = dist.histograms(species_Crds, r_max, n_bins)
                        else:
                            raise RuntimeError("unsupported CPU histogram kernel requested: " +
                                               str(histoparam['cpu']['module']))
                    elif (worker_type == "gpu"):
                        histograms = cudh.histograms(species_Crds, r_max, n_bins,
                                                     histoparam['gpu']['precision'],
                                                     gpu_id=worker_id,
                                                     scale_factors=histogram_scale_factors,
                                                     mask_array=histogram_mask,
                                                     check_input=histoparam['gpu']['check_input'],
                                                     box=periodic_box)
                    else:
                        raise RuntimeError("unsupported worker type requested: " + str(worker_type))
                except ValueError as error:
                    print(" %s %s: value error: pair distance > r_max" % (util.timeStamp(t0=t0), worker_str))
                    notify_master = True
                    break
                except RuntimeError as error:
                    print(" %s %s: runtime error: %s" % (util.timeStamp(t0=t0), worker_str, error.message))
                    notify_master = True
                    break
                except Exception as error:
                    print(" %s %s: general error: %s" % (util.timeStamp(t0=t0), worker_str, error.message))
                    notify_master = True
                    break
                #
                t1 = time.time()
                comp_time = t1 - t2
                #
                # --- temporarily pack the 2D histograms array into the Container instance
                frm.put_data('tmp/histograms', histograms)
                # --- delete the coordinate data
                frm.del_data(base.loc_coordinates)
                #
                result_item = (frm, comp_time, wait_time, worker_type, bap)
                #
                resultQueue.put(result_item)
                #
                icount += 1
                if (icount % histoparam['output']['flush_interval'] == 0):
                    sys.stdout.flush()
                taskQueue.task_done()
    except Exception as error:
        print(" %s %s: general error: %s" % (util.timeStamp(t0=t0), worker_str, error.message))
        notify_master = True
    if notify_master:
        print(" %s %s: sending shutdown signal to master process" % (util.timeStamp(t0=t0), worker_str))
        os.kill(os.getppid(), signal.SIGUSR1)
    if (histoparam['general']['verbose']):
        print(util.SEP)
        print(" %s %s: shutting down %s" % (util.timeStamp(t0=t0), worker_str, termination_msg))
        print(util.SEP)
    sys.stdout.flush()
    sys.exit(0)


def sum(histoparam, resultQueue, n_El, n_bins, dr, header_str, t0, n_frames, n_workers):
    """Worker function: Fetch histograms from resultQueue, order, sum up, and
    write results out to HDF5.

    To be used as the entry function for the multiprocessing summation subprocess.
    """

    worker_str = "sum worker"

    if (histoparam['general']['redirect_output']):
        util.redirectOutput(histoparam['output']['directory'] + "histo_sum_worker.log")
        print(util.SEP)
        print(" %s %s: starting" % (util.timeStamp(dateAndTime=True), worker_str))
        print(util.SEP)
    #
    # --- prepare and truncate the histogram file list
    if histoparam['output']['write_npy']:
        with open(histoparam['output']['directory'] + "distHisto.list", 'w') as fp:
            pass
    #
    iframe = histoparam['input']['first']
    #
    nHij = int(n_El) * (int(n_El) + 1) // 2
    histo_0 = np.zeros((n_bins, nHij + 1), order='F')
    for i in range(n_bins):
        histo_0[i, 0] = dr * (i + 0.5)
    #
    frm = base.Container()
    #
    histo = histo_0.copy()
    time_cpu = 0.0
    time_gpu = 0.0
    wait_cpu = 0.0
    wait_gpu = 0.0
    bap_cpu = 0.0
    bap_gpu = 0.0
    n_cpu = 0
    n_gpu = 0
    n_done = 0
    # counter starting at zero, independent of the actual frame numbers
    icount = 0
    termination_msg = ""
    #
    # --- use a hash map as a buffer to recover the original frame order
    buf = {}
    finished = False
    #
    with hdf5.H5Writer(file=histoparam['output']['directory'] + histoparam['output']['file'],
                       compression=histoparam['output']['compression'], mode='w') as h5writer:
        while True:
            if not finished:
                try:
                    result_item = resultQueue.get(False, 0.5)
                    if result_item in ["stop", "done"]:
                        resultQueue.task_done()
                        termination_msg = "(" + result_item + ")"
                        n_done += 1
                    else:
                        frame_number = (result_item[0]).i
                        buf[frame_number] = result_item
                        if (histoparam['general']['verbose']):
                            print(" %s %s: buffering frame %d ..." % (util.timeStamp(t0=t0), worker_str, frame_number))
                except Exception as e:
                    pass
            else:
                break  # will implicitly close h5writer
            #
            if (iframe in buf):
                icount = icount + 1
                #
                result_item = buf[iframe]
                frm_in = result_item[0]
                frame_time = result_item[1]
                frame_wait = result_item[2]
                processor_type = result_item[3]
                bap = result_item[4]
                del buf[iframe]
                #
                if (histoparam['general']['verbose']):
                    print(" %s %s: processing frame %d ..." % (util.timeStamp(t0=t0), worker_str, iframe))

                # --- sum distance histograms
                frame_histo = frm_in.get_data('tmp/histograms')
                histo += frame_histo
                frm_in.del_data('tmp/histograms')

                # --- sum length histograms
                if frm_in.contains_key(base.loc_len_histograms):
                    if not frm.contains_key(base.loc_len_histograms + '/radii'):
                        frm.put_data(base.loc_len_histograms + '/radii',
                                     frm_in.get_data(base.loc_len_histograms + '/radii'))
                    frm.sum_data(frm_in, base.loc_len_histograms)

                # --- append particle numbers
                if frm_in.contains_key(base.loc_nr_particles):
                    frm.append_data(frm_in, base.loc_nr_particles)

                # --- append volume of the periodic box
                if frm_in.contains_key(base.loc_volumes):
                    frm.append_data(frm_in, base.loc_volumes)

                if (processor_type == "cpu"):
                    time_cpu += frame_time
                    wait_cpu += frame_wait
                    n_cpu += 1
                    bap_cpu += bap
                elif (processor_type == "gpu"):
                    time_gpu += frame_time
                    wait_gpu += frame_wait
                    n_gpu += 1
                    bap_gpu += bap
                else:
                    pass
                #
                if (icount % histoparam['histogram']['sum'] == 0):
                    t1 = time.time()
                    wallclock = float(t1 - t0)
                    print(util.SEP)
                    print(" %s %s: writeout at frame %d (sum of %d frames)" % \
                        (util.timeStamp(t0=t0), worker_str, iframe, histoparam['histogram']['sum']))

                    # NOTE: We do not perform any averaging operation on the histogram here.
                    # Averaging is done consistently in the postprocessing averaging filter,
                    # where the histoparam['histogram']['sum'] is read from the pipeline.

                    # --- add and update metadata
                    frm.i = iframe
                    frm.put_data('log', frm_in.get_data('log'))
                    log_entry = {'histograms': histoparam}
                    frm.put_meta(log_entry)

                    # --- NumPy native output ---
                    oname = "distHisto.%d" % iframe
                    if histoparam['output']['write_npx']:
                        util.savetxtHeader(histoparam['output']['directory'] + oname + '.dat', header_str, histo)
                    if histoparam['output']['write_npy']:
                        np.save(histoparam['output']['directory'] + oname + ".npy", histo)
                        util.appendLineToFile(histoparam['output']['directory'] + "distHisto.list", oname + ".npy")

                    # --- HDF5 output ---
                    if histoparam['output']['write_h5']:
                        # convert histo[:,:] into hashed ('el2,el2') histo[:] arrays
                        for (idx, key) in enumerate(header_str.split()):
                            if key == '#':
                                key = 'radii'
                            frm.put_data(base.loc_histograms + '/' + key, histo[:, idx])
                        # --- write frame to HDF5 file
                        h5writer.put_frame(frm)

                    # --- reinitialize frame
                    del frm
                    frm = base.Container()
                    # --- reinitialize the 2D histogram array
                    histo = histo_0.copy()
                    # ---
                    if (n_cpu > 0):
                        print("   CPU: %d frames, %.3f (%.3f) avg comp (io) time [s], %.3f bapps"\
                            % (n_cpu, old_div(time_cpu, float(n_cpu)), old_div(wait_cpu, float(n_cpu)), old_div(bap_cpu, time_cpu)))
                    if (n_gpu > 0):
                        print("   GPU: %d frames, %.3f (%.3f) avg comp (io) time [s], %.3f bapps"\
                            % (n_gpu, old_div(time_gpu, float(n_gpu)), old_div(wait_gpu, float(n_gpu)), old_div(bap_gpu, time_gpu)))
                    if (n_frames > 0):
                        frac_done = float(icount) / float(n_frames)
                        time_est = wallclock / frac_done
                        print("   RUN: {:.2f} % done, estimated total run time {:.2f} s.".format(100. * frac_done, time_est))
                    print(util.SEP)
                #
                if (icount % histoparam['output']['flush_interval'] == 0):
                    sys.stdout.flush()
                    if histoparam['output']['write_h5']:
                        h5writer.hard_flush()
                # if (iframe == histoparam['input']['last']):
                    # finished = True
                iframe += 1
                resultQueue.task_done()
            elif (n_done == n_workers):
                finished = True
        # end while
        if (histoparam['general']['verbose']):
            print(util.SEP)
            print(" %s %s: shutting down %s" % (util.timeStamp(dateAndTime=True), worker_str, termination_msg))
            util.rm("stop")
            print(util.SEP)
            sys.stdout.flush()
    # end with ... h5writer
    del h5writer
    sys.exit(0)
