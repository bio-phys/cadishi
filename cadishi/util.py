# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4 fileencoding=utf-8
#
# Cadishi --- CAlculation of DIStance HIstograms
#
# Copyright (c) Klaus Reuter, Juergen Koefinger
# See the file AUTHORS.rst for the full list of contributors.
#
# Released under the MIT License, see the file LICENSE.txt.

"""Miscellaneous useful and convenient functions used by Cadishi and Capriqorn,
of potential general use.
"""
from __future__ import print_function
from __future__ import division


from builtins import str
from builtins import range
from past.builtins import basestring
from builtins import object
from past.utils import old_div
import importlib
import os
import sys
import re
import subprocess as sub
import gzip
import json
import getpass
import shutil
import numpy as np
from scipy import stats
import time
import datetime
import cProfile
import multiprocessing as mp


# in case we don't find PyYAML, we fall back to JSON that comes with Python
have_yaml = True
try:
    import yaml
except ImportError:
    have_yaml = False


SEP = "-----------------------------------------------------------------------------"


# inspired by https://zapier.com/engineering/profiling-python-boss/
def do_cprofile(func):
    """Decorator to run a function through cProfile."""
    def profiled_func(*args, **kwargs):
        profile = cProfile.Profile()
        try:
            profile.enable()
            result = func(*args, **kwargs)
            profile.disable()
            return result
        finally:
            profile.print_stats(sort='time')
    return profiled_func


# inspired by https://zapier.com/engineering/profiling-python-boss/
def timefunc(f):
    """Decorator to run simple timer on a function."""
    def f_timer(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        dt = end - start
        print(" Wall clock time: {:.3f} s".format((dt)))
        print(SEP)
        return result
    return f_timer


# inspired by https://zapier.com/engineering/profiling-python-boss/
try:
    from line_profiler import LineProfiler

    def do_lprofile(follow=[]):
        """Decorator to run the line profiler on a function."""
        def inner(func):
            def profiled_func(*args, **kwargs):
                try:
                    profiler = LineProfiler()
                    profiler.add_function(func)
                    for f in follow:
                        profiler.add_function(f)
                    profiler.enable_by_count()
                    return func(*args, **kwargs)
                finally:
                    profiler.print_stats()
            return profiled_func
        return inner
except ImportError:
    def do_lprofile(follow=[]):
        "Helpful if you accidentally leave in production!"
        def inner(func):
            def nothing(*args, **kwargs):
                return func(*args, **kwargs)
            return nothing
        return inner


def get_numa_domains():
    """Parse and return the output of the <numactl --hardware> command."""
    numa_topology = []
    regex = re.compile('node [0-9]+ cpus')
    cmd = ['numactl', '--hardware']
    try:
        raw = sub.check_output(cmd).split('\n')
        for line in raw:
            if re.match(regex, line):
                line_tokens = line.split()
                node_id = line_tokens[1]
                # ---
                del line_tokens[0]
                del line_tokens[0]
                del line_tokens[0]
                # ---
                node_cpus = ','.join(line_tokens)
                # ---
                numa_topology.append((node_id, node_cpus))
    except:
        del numa_topology[:]
    # ---
    return numa_topology


def set_numa_domain(numa_id, numa_topology):
    """Pin the current process onto a numa domain."""
    try:
        (numa_node, numa_cpus) = numa_topology[numa_id]
        pid = "%d" % os.getpid()
        cmd = ['taskset', '-pc', numa_cpus, pid]
        raw = sub.check_output(cmd).split('\n')
        print(SEP)
        print(" " + raw[0])
        print(" " + raw[1])
        print(SEP)
        exit_status = True
    except:
        exit_status = False
    return exit_status


def _cat_proc_cpuinfo_grep_query_sort_uniq(query):
    """Determine the number of unique lines in /proc/cpuinfo

    Parameters
    ----------
    string : query
        string the lines to be searched for shall begin with

    Returns
    -------
    set
        unique lines in /proc/cpuinfo that begin with query

    May throw an IOError exception in case /proc/cpuinfo does not exist.
    """
    items_seen = set()
    with open("/proc/cpuinfo") as fp:
        for line_raw in fp:
            if line_raw.startswith(query):
                line = line_raw.replace('\t', '').strip('\n')
                items_seen.add(line)
    return items_seen


def get_n_cpu_sockets():
    """Determine the number of CPU sockets on a Linux host.

    Returns
    -------
    int
        number of CPU sockets

    May throw an IOError exception in case /proc/cpuinfo does not exist.
    """
    return len(_cat_proc_cpuinfo_grep_query_sort_uniq("physical id"))


def get_n_cpu_cores():
    """Determine the number of CPU cores on a Linux host.

    Returns
    -------
    int
        number of CPU cores
    """
    # May throw an IOError exception in case /proc/cpuinfo does not exist.
    # return len(_cat_proc_cpuinfo_grep_query_sort_uniq("processor"))
    return mp.cpu_count()  # more-portable solution


def rm(resource):
    """Remove a file. If the file does not exist, no error is raised."""
    try:
        os.remove(resource)
    except OSError:
        pass


def rmrf(resource):
    """Remove file or directory tree. No error is raised if the target does not exist."""
    try:
        shutil.rmtree(resource)
    except OSError:
        pass


def md(resource):
    """Create a directory (for a file), if necessary.  The behaviour highly
    depends on the resource (string) parameter.

    If resource is of the form "string",
        nothing is done.
    If resource is of the form "string/",
        a directory labeled "string" is created.
    If resource is of the form "string/foo",
        a directory labeled "string" is created.
    If resource is of the form "string/foo/",
        a directory structure "string/foo" is created.
    """
    folder = os.path.dirname(resource)
    if (len(folder) == 0):
        return
    if not os.path.exists(folder):
        # Despite the 'if' there may be a race condition when a parallel
        # pipeline with many workers is used, so we catch the exception thrown
        # if the directory does exist.
        try:
            os.makedirs(folder)
        except OSError:
            pass


def ls(resource, files=True, directories=False):
    """Return a list of files and optionally directories located at resource."""
    file_list = []
    for (_dirpath, _dirnames, _filenames) in os.walk(resource):
        if files:
            file_list.extend(_filenames)
        if directories:
            file_list.extend(_dirnames)
        break
    return file_list


def testcase():
    """Try to locate the test case that comes with cadishi.  Works for a
    check-out (or tarball) of the source files as well as for an installation.
    Returns the full path to the testcase including a trailing slash."""
    file_path = os.path.dirname(os.path.abspath(__file__))
    testcase_path = os.path.abspath(file_path + "/tests/data")
    return testcase_path + "/"


def scratch_dir():
    """Return and create a per-user scratch directory for unit test data.
    """
    uid = getpass.getuser()
    dir = "/tmp/" + uid + "/py.test/"
    md(dir)
    return dir


def load_class(module_name, class_name):
    """Load a class from a module, where class and module are specified as
    strings. Useful to dynamically build Capriqorn pipelines.
    """
    m = importlib.import_module(module_name)
    c = getattr(m, class_name)
    return c


def tokenize(path, sep='/'):
    """Remove any separators sep from path and return a list of the strings in
    between. If path is empty or '/', the list has the empty string as a single
    entry.
    """
    assert isinstance(path, basestring)
    val = path.rstrip(sep).lstrip(sep).split(sep)
    return val


def pipeline_entry(label, param):
    """Return a dictionary with a single key-value pair, in particular label
    (key) being a string label, and param being a dictionary with string keys
    and arbitrary values. The return value may be appended eg. to a pipeline_log
    list.
    """
    assert isinstance(label, basestring)
    assert isinstance(param, dict)
    entry = {}
    entry[label] = param
    return entry


def search_pipeline(label, pipeline):
    """Iterate through the pipeline list backwards in order to find the entry
    (dict) identified by label (string)."""
    assert isinstance(label, basestring)
    assert isinstance(pipeline, list)
    # ---
    for entry in reversed(pipeline):
        parameters = {}
        for (key, parameters) in entry.items():
            if (key == label):
                assert isinstance(parameters, dict)
                return parameters
    # ---
    return None


def get_elements(header):
    """Return a complete list of all the chemical elements present in a header.
    Header may be a string or a list containing single element IDs or pair
    combinations thereof.  "#" and "radii" are skipped automatically.
    """
    if isinstance(header, basestring):
        header = header.rstrip().split()
    assert isinstance(header, list)
    el_set = set([])
    for item in header:
        if (item == '#') or (item == 'radii'):
            continue
        pair = item.split(',')
        el_set.add(pair[0])
        if (len(pair) == 2):
            el_set.add(pair[1])
    return sorted(list(el_set))


def open_r(filename):
    """Open an uncompressed or GZIP-compressed text file for reading. Return the
    file pointer."""
    assert (os.path.exists(filename))
    if filename.endswith('.gz'):
        fp = gzip.open(filename, mode='rb')
    else:
        fp = open(filename, 'r')
    return fp


def appendLineToFile(filename, string):
    """Append string as a line to the end of the file identified by filename."""
    with open(filename, 'a') as fp:
        if not string.endswith('\n'):
            string += '\n'
        fp.write(string)


def savetxtHeader(name, header, array):
    """Save data including its header.
    Legacy routine from the initial histograms implementation."""
    md(name)
    fp = open(name, 'w')
    if header[-1] is not '\n':
        header += '\n'
    fp.write(header)
    fp.close()
    fp = open(name, 'a')
    np.savetxt(fp, array)
    fp.close()


def write_xyzFile(coords, names, filename):
    """Write coordinates in xyz format to the file labeled filename."""
    fp = open(filename, 'w')
    fp.write("%d \n generated with histograms.py\n" % len(coords))
    for i in range(len(coords)):
        fp.write("%s %8.3f %8.3f %8.3f\n" %
                 tuple([names[i]] + list(coords[i])))
    fp.close()
    return


if have_yaml:
    def load_yaml(filename):
        with open(filename, "r") as fp:
            return yaml.safe_load(fp)

    def save_yaml(data, filename):
        with open(filename, "w") as fp:
            yaml.safe_dump(data, fp, default_flow_style=False)


def load_json(filename):
    with open(filename, "r") as fp:
        return json.load(fp)


def save_json(data, filename):
    with open(filename, "w") as fp:
        json.dump(data, fp, indent=4, sort_keys=True)


def load_parameter_file(filename):
    """Load parameters from a JSON or YAML file and return it as a nested
    structure of dictionaries."""
    if not os.path.exists(filename):
        raise IOError("File '" + filename + "' does not exist")
    if filename.endswith('.json'):
        return load_json(filename)
    else:
        if have_yaml:
            return load_yaml(filename)
        else:
            raise RuntimeError("PyYAML is not available.")


def compare(histo1, histo2):
    """Compare two NumPy arrays (e.g. histograms) if they are identical."""
    if (histo1.dtype == histo2.dtype) and (histo1.dtype == np.float64):
        compare_strictly(histo1, histo2)
    else:
        compare_approximately(histo1, histo2)


def compare_strictly(histo1, histo2):
    """Check if two histograms (1D numpy arrays) or two sets of histograms (2D
    numpy arrays) are identical.

    Only suitable to check the results of double precision computations.
    """
    assert (histo1 == histo2).all()


def compare_approximately(histo1, histo2, ks_stat_max=0.01, p_value_min=0.99):
    """Check if two histograms (1D numpy arrays) or two sets of histograms (2D
    numpy arrays) are reasonably similar.

    The routine can be used to check the results of single precision computations
    against a reference file.  Computes the Kolmogorov-Smirnov statistic on 2
    samples using
    http://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.stats.ks_2samp.html
    """
    assert((histo1.shape == histo2.shape))
    # --- remove first column (is not a histogram)
    if (histo1.ndim == 2):
        hist1 = histo1[:, 1:]
        hist2 = histo2[:, 1:]
    elif (histo1.ndim == 1):
        hist1 = histo1
        hist2 = histo2
    else:
        raise ValueError('histograms must be passed as 1D or 2D arrays')
    # --- check if the number of binned items is identical
    binsum1 = np.sum(hist1, axis=0, dtype=np.uint64)
    binsum2 = np.sum(hist2, axis=0, dtype=np.uint64)
    assert (binsum1 == binsum2).all()
    # --- contruct an error metric
    hdiff = np.subtract(hist1, hist2)
    dnorm = np.linalg.norm(hdiff, axis=0)
    # --- finally, compare err and tolerance
    if (histo1.ndim == 2):
        for col in range(0, hist1.shape[1]):
            ks_stat, p_value = stats.ks_2samp(hist1[:, col], hist2[:, col])
            assert(ks_stat < ks_stat_max)
            assert(p_value > p_value_min)
    else:
        ks_stat, p_value = stats.ks_2samp(hist1, hist2)
        assert(ks_stat < ks_stat_max)
        assert(p_value > p_value_min)


def dump_histograms(filename, histograms, r_max, n_bins):
    """Save histograms into a NumPy text file.  Legacy routine."""
    dr = old_div(float(r_max), float(n_bins))
    histos = histograms.astype(dtype=np.float64)
    radii = [dr * (float(i) + 0.5) for i in range(n_bins)]
    histos[:, 0] = np.asarray(radii)
    np.savetxt(filename, histos)


def get_executable_name():
    """Return the name of the present executable."""
    return os.path.basename(sys.argv[0])


def generate_random_coordinate_set(n_atoms=[512, 1024, 2048],
                                   coord_min=(0., 0., 0.),
                                   coord_max=(1., 1., 1.),
                                   blowup_factor=1.0):
    """Return pseudo-random coordinate sets in a box."""
    coords = []
    coord_min = np.asanyarray(coord_min)
    coord_max = np.asanyarray(coord_max)
    coord_width = (coord_max - coord_min) * blowup_factor
    for n in n_atoms:
        c = np.random.rand(n, 3) * coord_width + coord_min
        coords.append(c)
    return coords


def generate_random_point_in_sphere(R):
    """Return a coordinate triple of a randomly located point inside a sphere of radius R."""
    costheta = 2. * (np.random.rand() - 0.5)  # random(-1, 1)
    u = np.power(np.random.rand(), old_div(1., 3.))
    # random point in spherical coordinates
    r = R * u
    t = np.arccos(costheta)
    p = 2. * np.pi * np.random.rand()
    # transform to Cartesian
    x = r * np.sin(t) * np.cos(p)
    y = r * np.sin(t) * np.sin(p)
    z = r * np.cos(t)
    return [x, y, z]


def generate_random_point_on_spherical_surface(R):
    """Return a coordinate triple of a randomly located point inside a sphere of radius R."""
    costheta = 2. * (np.random.rand() - 0.5)  # random(-1, 1)
    # random point in spherical coordinates
    r = R
    t = np.arccos(costheta)
    p = 2. * np.pi * np.random.rand()
    # transform to Cartesian
    x = r * np.sin(t) * np.cos(p)
    y = r * np.sin(t) * np.sin(p)
    z = r * np.cos(t)
    return [x, y, z]


def check_parameter_labels(input, reference):
    """Check the keys of a two-level nested dictionary structure against a reference structure.

    Raises KeyError in case of an invalid key.
    """
    for key_1 in input:
        if key_1 not in reference:
            raise KeyError("'" + key_1 + "' is not a valid section label")
        else:
            for key_2 in input[key_1]:
                if key_2 not in reference[key_1]:
                    raise KeyError("'" + key_2 + "' is not a valid label in section '" + key_1 + "'")


def make_iterable(obj):
    """Pack obj into a list if it is not iterable, yet."""
    if not isinstance(obj, (list, tuple, set)):
        obj = [obj]
    return obj


def check_parameter(parameters, label, dtype, default_value,
                    valid_values=None, min_value=None, max_value=None, file_existence=False):
    """Check a parameter for validity.

    Throws ValueError or IOError with useful end-user-friendly messages.
    """
    sec, key = label.split(':')
    assert(sec in parameters)
    if not key in parameters[sec]:
        parameters[sec][key] = default_value
    # various tests of 'value' throwing exceptions with readable error messages
    value = parameters[sec][key]
    if not isinstance(value, dtype):
        raise ValueError(label + ": invalid datatype " + str(type(value)) +
                         ", expected " + str(dtype))
    if valid_values is not None:
        valid_values = make_iterable(valid_values)
        if value not in valid_values:
            raise ValueError(label + ": invalid value '" + str(value) +
                             "', valid values are '" + str(valid_values) + "'")
    elif file_existence:
        # we assume that 'value' is a file or a list of files
        file_list = make_iterable(value)
        for file in file_list:
            if not os.path.exists(file):
                raise IOError(label + ": '" + file + "' does not exist")
    else:
        if (min_value is not None) and (value < min_value):
            raise ValueError(label + ": invalid value '" + str(value) +
                             "', minimum allowed value is '" + str(min_value) + "'")
        if (max_value is not None) and (value > max_value):
            raise ValueError(label + ": invalid value '" + str(value) +
                             "', maximum allowed value is '" + str(max_value) + "'")


def redirectOutput(filename):
    """Redirect stdout and stderr of the present process to the file specified by filename."""
    o_flags = os.O_CREAT | os.O_TRUNC | os.O_WRONLY
    os.close(1)
    os.open(filename, o_flags, 0o664)
    os.close(2)
    os.dup(1)


def timeStamp(dateAndTime=False, t0=0.0):
    """Return a string with a time stamp suitable to prefix log lines with."""
    tnow = time.time()
    if dateAndTime:
        timestr = datetime.datetime.fromtimestamp(tnow).strftime('%Y-%m-%d %H:%M:%S')
    else:
        timestr = "%19.3f" % (tnow - t0)
    return "[" + timestr + "]"


class PrintWrapper(object):
    """Wrapper to implement infrequent message printing."""

    def __init__(self):
        self.context_dict = {}

    def once(self, context, msg, time_stamp=None):
        """Print context and message exactly once."""
        if (context not in self.context_dict) or ((context in self.context_dict) and
                                                  (self.context_dict[context] is not True)):
            if time_stamp is not None:
                stamp = " " + time_stamp
            else:
                stamp = ""
            print(stamp + " " + context + ": " + msg)
            self.context_dict[context] = True

    def every(self, context, msg):
        """Print at every nth invocation."""
        # Yet to be implemented.
        pass


def quote(string):
    """Add quotes to the beginning and the end of a string if not already present."""
    string_elements = []
    if not string.startswith('\''):
        string_elements.append('\'')
    string_elements.append(string)
    if not string.endswith('\''):
        string_elements.append('\'')
    return "".join(string_elements)
