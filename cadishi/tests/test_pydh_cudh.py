# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4 fileencoding=utf-8
#
# Cadishi --- CAlculation of DIStance HIstograms
#
# Copyright (c) Klaus Reuter, Juergen Koefinger
# See the file AUTHORS.rst for the full list of contributors.
#
# Released under the MIT License, see the file LICENSE.txt.

"""A set of unit tests for the pydh CPU and cudh GPU histogram modules.
"""
from __future__ import print_function


from builtins import str
from builtins import range
import os
import sys
import numpy as np
import glob
import math
import multiprocessing
import pytest
from cadishi import util


# --- select the modules to be tested via environment variables
TEST_PYDH = bool(int(os.environ.get("TEST_PYDH", "1")))
TEST_CUDH = bool(int(os.environ.get("TEST_CUDH", "0")))
# --- toggle large test case which may take a long time
TEST_LARGE = bool(int(os.environ.get("TEST_LARGE", "0")))
# --- toggle extra-large test case which may take even longer
TEST_XLARGE = bool(int(os.environ.get("TEST_XLARGE", "0")))
# --- dump the histograms from the medium and large problem sets for manual inspection
DUMP_DATA = bool(int(os.environ.get("DUMP_DATA", "0")))


# --- global variables ---
# r_max, coordinates are in a unit box
r_max = math.sqrt(3.0)


# --- set up the number of threads to be tested, depending on the machine
n_cores = multiprocessing.cpu_count()
n_threads = [1]
while (n_threads[-1] < n_cores):
    n_threads.append(2 * n_threads[-1])
if (n_threads[-1] > n_cores):
    n_threads.pop()
n_threads.remove(1)

print("n_threads = " + str(n_threads))

# --- import the dist module which serves as the reference implementation
from cadishi.kernel import dist


# --- import the pydh module
from cadishi.kernel import pydh


# --- import the cudh module
if TEST_CUDH:
    try:
        from cadishi.kernel import cudh
    except Exception as e:
        print("Error importing >> cudh <<.  Disabling CUDA tests.")
        print("Exception message : " + e.message)
        TEST_CUDH = False

if TEST_CUDH:
    # test if we are able to run the tests at all
    if (cudh.get_num_devices() == 0):
        print("No usable CUDA device detected.  Disabling CUDA tests.")
        TEST_CUDH = False
    else:
        print("CUDA tests: " + str(cudh.get_num_devices()) + " GPUs detected.")


def get_triclinic_box():
    """Return an (arbitrarily defined) triclinic box."""
    return np.asarray([0.66, 0.75, 0.88, 33., 45., 66.])


def get_orthorhombic_triclinic_box():
    """Return an (arbitrarily defined) orthorhombic box (using a triclinic specifier)."""
    return np.asarray([0.66, 0.75, 0.88, 90., 90., 90.])


def get_orthorhombic_box():
    """Return an (arbitrarily defined) orthorhombic box."""
    box = np.zeros((3, 3))
    box[0][0] = 0.66
    box[1][1] = 0.75
    box[2][2] = 0.88
    return box


testcase_small = None


@pytest.fixture
def fixture_small():
    """Create a reference case using the very first dist implementation."""
    global testcase_small
    if testcase_small is None:
        n_atoms = [2847, 3918]
        n_bins = 1000
        coords = util.generate_random_coordinate_set(n_atoms)
        histo = dist.histograms(coords, r_max, n_bins)
        # if DUMP_DATA:
        #     file_name = sys._getframe().f_code.co_name + ".dat"
        #     util.dump_histograms(file_name, histo, r_max, n_bins)
        testcase_small = (n_atoms, n_bins, coords, histo)
    return testcase_small


testcase_small_orthorhombic = None


@pytest.fixture
def fixture_small_orthorhombic():
    """Create a orthorhombic reference case in double precision using pydh."""
    global testcase_small_orthorhombic
    if testcase_small_orthorhombic is None:
        n_atoms = [2847, 3918]
        n_bins = 1000
        coords = util.generate_random_coordinate_set(n_atoms)
        box = get_orthorhombic_box()
        histo = pydh.histograms(coords, r_max, n_bins, box=box, precision="double", n_threads=1)
        testcase_small_orthorhombic = (n_atoms, n_bins, coords, box, histo)
    return testcase_small_orthorhombic


testcase_small_triclinic = None


@pytest.fixture
def fixture_small_triclinic():
    """Create a triclinic reference case in double precision using pydh."""
    global testcase_small_triclinic
    if testcase_small_triclinic is None:
        n_atoms = [2847, 3918]
        n_bins = 1000
        coords = util.generate_random_coordinate_set(n_atoms)
        box = get_triclinic_box()
        histo = pydh.histograms(coords, r_max, n_bins, box=box, precision="double", n_threads=1)
        testcase_small_triclinic = (n_atoms, n_bins, coords, box, histo)
    return testcase_small_triclinic


if TEST_PYDH:
    def test_pydh_small_double_blocked(fixture_small):
        n_atoms, n_bins, coords, histo_ref = fixture_small
        for check_input in [True, False]:
            for nt in n_threads:
                histo_blocked = pydh.histograms(coords, r_max, n_bins, precision="double",
                                                n_threads=nt, blocksize=200, check_input=check_input)
                util.compare(histo_ref, histo_blocked)

    def test_pydh_small_double(fixture_small):
        """Test if pydh gives the same answer as dist()."""
        n_atoms, n_bins, coords, histo_ref = fixture_small
        for check_input in [True, False]:
            histo = pydh.histograms(coords, r_max, n_bins, precision="double",
                                    n_threads=1, check_input=check_input)
            util.compare(histo_ref, histo)

    def test_n_threads_small_double(fixture_small):
        """Test if pydh gives the same answer as dist()."""
        n_atoms, n_bins, coords, histo_ref = fixture_small
        for check_input in [True, False]:
            for nt in n_threads:
                histo_pydh = pydh.histograms(coords, r_max, n_bins, precision="double",
                                             n_threads=nt, check_input=check_input)
                util.compare(histo_ref, histo_pydh)

    def test_pydh_small_single(fixture_small):
        """Test if pydh gives the same answer as dist()."""
        n_atoms, n_bins, coords, histo_ref = fixture_small
        for check_input in [True, False]:
            histo_pydh = pydh.histograms(coords, r_max, n_bins, precision="single",
                                         n_threads=1, check_input=check_input)
            util.compare(histo_ref, histo_pydh)

    def test_n_threads_small_single(fixture_small):
        """Test if pydh gives the same answer as dist()."""
        n_atoms, n_bins, coords, histo_ref = fixture_small
        for check_input in [True, False]:
            for nt in n_threads:
                histo_pydh = pydh.histograms(coords, r_max, n_bins, precision="single",
                                             n_threads=nt, check_input=check_input)
                util.compare(histo_ref, histo_pydh)

    def test_pydh_small_orthorhombic_single(fixture_small_orthorhombic):
        """Test if the orthorhombic implementation gives the same answer in single precision."""
        n_atoms, n_bins, coords, box, histo_ref = fixture_small_orthorhombic
        for check_input in [True, False]:
            histo = pydh.histograms(coords, r_max, n_bins, box=box, precision="single",
                                    n_threads=1, check_input=check_input)
            util.compare(histo_ref, histo)

    def test_pydh_small_triclinic_single(fixture_small_triclinic):
        """Test if the triclinic implementation gives the same answer in single precision."""
        n_atoms, n_bins, coords, box, histo_ref = fixture_small_triclinic
        for check_input in [True, False]:
            histo = pydh.histograms(coords, r_max, n_bins, box=box, precision="single",
                                    n_threads=1, check_input=check_input)
            util.compare(histo_ref, histo)

    def test_pydh_small_orthorhombic_triclinic(fixture_small_triclinic):
        """Test if the triclinic and orthorhombic implementations give the same answer for an orthorhombic box."""
        n_atoms, n_bins, coords, box, histo_ref = fixture_small_triclinic
        box_ort = get_orthorhombic_box()
        box_tri = get_orthorhombic_triclinic_box()
        for precision in ['single', 'double']:
            for check_input in [True, False]:
                histo_ort = pydh.histograms(coords, r_max, n_bins, box=box_ort, force_triclinic=False,
                                            precision=precision, n_threads=1, check_input=check_input)
                histo_tri = pydh.histograms(coords, r_max, n_bins, box=box_tri, force_triclinic=True,
                                            precision=precision, n_threads=1, check_input=check_input)
                util.compare(histo_ort, histo_tri)

if TEST_CUDH:
    def test_cudh_small_double(fixture_small):
        n_atoms, n_bins, coords, histo_ref = fixture_small
        for check_input in [True, False]:
            for gpu_id in range(cudh.get_num_devices()):
                for algo in [1, 2, 3]:
                    histo = cudh.histograms(coords, r_max, n_bins, precision="double",
                                            gpu_id=gpu_id, check_input=check_input, algorithm=algo)
                    util.compare(histo_ref, histo)

    def test_cudh_small_single(fixture_small):
        n_atoms, n_bins, coords, histo_ref = fixture_small
        for check_input in [True, False]:
            for gpu_id in range(cudh.get_num_devices()):
                for algo in [1, 2, 3]:
                    histo = cudh.histograms(coords, r_max, n_bins, precision="single",
                                            gpu_id=gpu_id, check_input=check_input, algorithm=algo)
                    util.compare(histo_ref, histo)

    def test_cudh_small_orthorhombic(fixture_small_orthorhombic):
        """Check if pydh and cudh give the same answer with orthorhombic boxes."""
        n_atoms, n_bins, coords, box, histo_ref = fixture_small_orthorhombic
        for precision in ['single', 'double']:
            for check_input in [True, False]:
                for algo in [1, 2, 3]:
                    histo = cudh.histograms(coords, r_max, n_bins, box=box, precision=precision,
                                            check_input=check_input, algorithm=algo)
                    util.compare(histo_ref, histo)

    def test_cudh_small_triclinic(fixture_small_triclinic):
        """Check if pydh and cudh give the same answer with triclinic boxes."""
        n_atoms, n_bins, coords, box, histo_ref = fixture_small_triclinic
        for precision in ['single', 'double']:
            for check_input in [True, False]:
                for algo in [1, 2, 3]:
                    histo = cudh.histograms(coords, r_max, n_bins, box=box, precision=precision,
                                            check_input=check_input, algorithm=algo)
                    util.compare(histo_ref, histo)

    def test_cudh_small_orthorhombic_triclinic(fixture_small_triclinic):
        """Test if the triclinic and orthorhombic implementations give the same answer for an orthorhombic box."""
        n_atoms, n_bins, coords, box, histo_ref = fixture_small_triclinic
        box_ort = get_orthorhombic_box()
        box_tri = get_orthorhombic_triclinic_box()
        for precision in ['single', 'double']:
            for check_input in [True, False]:
                for algo in [1, 2, 3]:
                    histo_ort = cudh.histograms(coords, r_max, n_bins, box=box_ort, force_triclinic=False,
                                                precision=precision, check_input=check_input, algorithm=algo)
                    histo_tri = cudh.histograms(coords, r_max, n_bins, box=box_tri, force_triclinic=True,
                                                precision=precision, check_input=check_input, algorithm=algo)
                    util.compare(histo_ort, histo_tri)


testcase_small_invalid = None


@pytest.fixture
def fixture_small_invalid():
    global testcase_small_invalid
    if testcase_small_invalid is None:
        n_atoms = [2000, 1000]
        n_bins = 1000
        coords = util.generate_random_coordinate_set(n_atoms, blowup_factor=3.14)
        histo_ref = None
        testcase_small_invalid = (n_atoms, n_bins, coords, histo_ref)
    return testcase_small_invalid


if TEST_PYDH:
    def test_pydh_invalid_small_double(fixture_small_invalid):
        n_atoms, n_bins, coords, histo_ref = fixture_small_invalid
        with pytest.raises(ValueError):
            pydh.histograms(coords, r_max, n_bins, precision="double", n_threads=1, check_input=True)

    def test_pydh_invalid_threads_small_double(fixture_small_invalid):
        n_atoms, n_bins, coords, histo_ref = fixture_small_invalid
        for nt in n_threads:
            with pytest.raises(ValueError):
                pydh.histograms(coords, r_max, n_bins, precision="double", n_threads=nt, check_input=True)

    def test_pydh_invalid_small_single(fixture_small_invalid):
        n_atoms, n_bins, coords, histo_ref = fixture_small_invalid
        with pytest.raises(ValueError):
            pydh.histograms(coords, r_max, n_bins, precision="single", n_threads=1, check_input=True)

    def test_pydh_invalid_threads_small_single(fixture_small_invalid):
        n_atoms, n_bins, coords, histo_ref = fixture_small_invalid
        for nt in n_threads:
            with pytest.raises(ValueError):
                pydh.histograms(coords, r_max, n_bins, precision="single", n_threads=nt, check_input=True)

if TEST_CUDH:
    def test_cudh_invalid_small_double(fixture_small_invalid):
        n_atoms, n_bins, coords, histo_ref = fixture_small_invalid
        for gpu_id in range(cudh.get_num_devices()):
            with pytest.raises(ValueError):
                cudh.histograms(coords, r_max, n_bins, precision="double", gpu_id=gpu_id, check_input=True)

    def test_cudh_invalid_small_single(fixture_small_invalid):
        n_atoms, n_bins, coords, histo_ref = fixture_small_invalid
        for gpu_id in range(cudh.get_num_devices()):
            with pytest.raises(ValueError):
                cudh.histograms(coords, r_max, n_bins, precision="single", gpu_id=gpu_id, check_input=True)


testcase_medium = None


@pytest.fixture
def fixture_medium():
    global testcase_medium
    if testcase_medium is None:
        n_atoms = [3000, 1000, 5000, 3500]
        n_bins = 8192
        coords = util.generate_random_coordinate_set(n_atoms)
        histo_ref = dist.histograms(coords, r_max, n_bins)
        testcase_medium = (n_atoms, n_bins, coords, histo_ref)
    return testcase_medium


if TEST_PYDH:
    def test_pydh_medium_double(fixture_medium):
        n_atoms, n_bins, coords, histo_ref = fixture_medium
        for check_input in [True, False]:
            histo_pydh = pydh.histograms(coords, r_max, n_bins, precision="double",
                                         n_threads=1, check_input=check_input)
            util.compare(histo_ref, histo_pydh)

    def test_n_threads_medium_double(fixture_medium):
        n_atoms, n_bins, coords, histo_ref = fixture_medium
        for check_input in [True, False]:
            for nt in n_threads:
                histo_pydh = pydh.histograms(coords, r_max, n_bins, precision="double",
                                             n_threads=nt, check_input=check_input)
                util.compare(histo_ref, histo_pydh)

    def test_pydh_medium_single(fixture_medium):
        n_atoms, n_bins, coords, histo_ref = fixture_medium
        for check_input in [True, False]:
            histo_pydh = pydh.histograms(coords, r_max, n_bins, precision="single",
                                         n_threads=1, check_input=check_input)
            util.compare(histo_ref, histo_pydh)

    def test_n_threads_medium_single(fixture_medium):
        n_atoms, n_bins, coords, histo_ref = fixture_medium
        for check_input in [True, False]:
            for nt in n_threads:
                histo_pydh = pydh.histograms(coords, r_max, n_bins, precision="single",
                                             n_threads=nt, check_input=check_input)
                util.compare(histo_ref, histo_pydh)

    def test_pydh_medium_masked_single(fixture_medium):
        n_atoms, n_bins, coords, histo_ref = fixture_medium
        n_el = len(n_atoms)
        mask_array = np.ones(n_el * (n_el + 1) / 2)
        mask_array[::2] = 0
        for check_input in [True, False]:
            histo_pydh = pydh.histograms(coords, r_max, n_bins, precision="single",
                                         n_threads=1, mask_array=mask_array, check_input=check_input)
            col_sum = histo_pydh.sum(axis=0)
            assert(np.sum(col_sum[1::2]) == 0)

    def test_pydh_medium_scaled_single(fixture_medium):
        n_atoms, n_bins, coords, histo_ref = fixture_medium
        n_el = len(n_atoms)
        scale_factors = np.ones(n_el * (n_el + 1) / 2)
        scale_factors *= 0.5
        for check_input in [True, False]:
            histo_pydh = pydh.histograms(coords, r_max, n_bins, precision="single",
                                         n_threads=1, scale_factors=scale_factors, check_input=check_input)
            assert(histo_ref.sum() == 2.0 * histo_pydh.sum())

if TEST_CUDH:
    def test_cudh_medium_double(fixture_medium):
        n_atoms, n_bins, coords, histo_ref = fixture_medium
        for check_input in [True, False]:
            for gpu_id in range(cudh.get_num_devices()):
                for algo in [1, 2, 3]:
                    histo_cudh = cudh.histograms(coords, r_max, n_bins, precision="double",
                                                 gpu_id=gpu_id, check_input=check_input, algorithm=algo)
                    util.compare(histo_ref, histo_cudh)

    def test_cudh_medium_single(fixture_medium):
        n_atoms, n_bins, coords, histo_ref = fixture_medium
        for check_input in [True, False]:
            for gpu_id in range(cudh.get_num_devices()):
                for algo in [1, 2, 3]:
                    histo_cudh = cudh.histograms(coords, r_max, n_bins, precision="single",
                                                 gpu_id=gpu_id, check_input=check_input, algorithm=algo)
                    util.compare(histo_ref, histo_cudh)

    def test_cudh_medium_masked_single(fixture_medium):
        n_atoms, n_bins, coords, histo_ref = fixture_medium
        n_el = len(n_atoms)
        mask_array = np.ones(n_el * (n_el + 1) / 2)
        mask_array[::2] = 0
        for check_input in [True, False]:
            for gpu_id in range(cudh.get_num_devices()):
                for algo in [1, 2, 3]:
                    histo_cudh = cudh.histograms(coords, r_max, n_bins, precision="single",
                                                 gpu_id=gpu_id, mask_array=mask_array, check_input=check_input, algorithm=algo)
                    col_sum = histo_cudh.sum(axis=0)
                    assert(np.sum(col_sum[1::2]) == 0)

    def test_cudh_medium_scaled_single(fixture_medium):
        n_atoms, n_bins, coords, histo_ref = fixture_medium
        n_el = len(n_atoms)
        scale_factors = np.ones(n_el * (n_el + 1) / 2)
        scale_factors *= 0.5
        for check_input in [True, False]:
            for gpu_id in range(cudh.get_num_devices()):
                for algo in [1, 2, 3]:
                    histo_cudh = cudh.histograms(coords, r_max, n_bins, precision="single", gpu_id=gpu_id,
                                                 scale_factors=scale_factors, check_input=check_input, algorithm=algo)
                    assert(histo_ref.sum() == 2.0 * histo_cudh.sum())


# test case designed to select the simple kernels in cudh
testcase_medium_manybins = None


@pytest.fixture
def fixture_medium_manybins():
    global testcase_medium_manybins
    if testcase_medium_manybins is None:
        n_atoms = [3000, 5000, 3500]
        n_bins = 68000
        coords = util.generate_random_coordinate_set(n_atoms)
        histo_ref = dist.histograms(coords, r_max, n_bins)
        testcase_medium_manybins = (n_atoms, n_bins, coords, histo_ref)
    return testcase_medium_manybins


if TEST_PYDH:
    def test_pydh_medium_manybins_double(fixture_medium_manybins):
        n_atoms, n_bins, coords, histo_ref = fixture_medium_manybins
        for check_input in [True, False]:
            histo_pydh = pydh.histograms(coords, r_max, n_bins, precision="double",
                                         n_threads=1, check_input=check_input)
            util.compare(histo_ref, histo_pydh)

    def test_n_threads_medium_manybins_double(fixture_medium_manybins):
        n_atoms, n_bins, coords, histo_ref = fixture_medium_manybins
        for check_input in [True, False]:
            for nt in n_threads:
                histo_pydh = pydh.histograms(coords, r_max, n_bins, precision="double",
                                             n_threads=nt, check_input=check_input)
                util.compare(histo_ref, histo_pydh)

    def test_pydh_medium_manybins_single(fixture_medium_manybins):
        n_atoms, n_bins, coords, histo_ref = fixture_medium_manybins
        for check_input in [True, False]:
            histo_pydh = pydh.histograms(coords, r_max, n_bins, precision="single",
                                         n_threads=1, check_input=check_input)
            util.compare(histo_ref, histo_pydh)

    def test_n_threads_medium_manybins_single(fixture_medium_manybins):
        n_atoms, n_bins, coords, histo_ref = fixture_medium_manybins
        for check_input in [True, False]:
            for nt in n_threads:
                histo_pydh = pydh.histograms(coords, r_max, n_bins, precision="single",
                                             n_threads=nt, check_input=check_input)
                util.compare(histo_ref, histo_pydh)

if TEST_CUDH:
    def test_cudh_medium_manybins_double(fixture_medium_manybins):
        n_atoms, n_bins, coords, histo_ref = fixture_medium_manybins
        for check_input in [True, False]:
            for gpu_id in range(cudh.get_num_devices()):
                for algo in [1, 2, 3]:
                    histo_cudh = cudh.histograms(coords, r_max, n_bins, precision="double",
                                                 gpu_id=gpu_id, check_input=check_input, algorithm=algo)
                    util.compare(histo_ref, histo_cudh)

    def test_cudh_medium_manybins_single(fixture_medium_manybins):
        n_atoms, n_bins, coords, histo_ref = fixture_medium_manybins
        for check_input in [True, False]:
            for gpu_id in range(cudh.get_num_devices()):
                for algo in [1, 2, 3]:
                    histo_cudh = cudh.histograms(coords, r_max, n_bins, precision="single",
                                                 gpu_id=gpu_id, check_input=check_input, algorithm=algo)
                    util.compare(histo_ref, histo_cudh)


if TEST_LARGE:
    testcase_large = None

    @pytest.fixture
    def fixture_large():
        global testcase_large
        if testcase_large is None:
            n_atoms = [105000, 110000, 133000]
            n_bins = 16000
            coords = util.generate_random_coordinate_set(n_atoms)
            # --- note that we use pydh to generate the test dataset
            histo_ref = pydh.histograms(coords, r_max, n_bins, precision="double", n_threads=n_threads[-1])
            testcase_large = (n_atoms, n_bins, coords, histo_ref)
        return testcase_large

    if TEST_PYDH:
        def test_pydh_large_single(fixture_large):
            n_atoms, n_bins, coords, histo_ref = fixture_large
            histo_pydh = pydh.histograms(coords, r_max, n_bins, precision="single", n_threads=1)
            util.compare(histo_ref, histo_pydh)

        def test_n_threads_large_single(fixture_large):
            n_atoms, n_bins, coords, histo_ref = fixture_large
            for nt in n_threads:
                histo_pydh = pydh.histograms(coords, r_max, n_bins, precision="single", n_threads=nt)
                util.compare(histo_ref, histo_pydh)

    if TEST_CUDH:
        def test_cudh_large_double(fixture_large):
            n_atoms, n_bins, coords, histo_ref = fixture_large
            for gpu_id in range(cudh.get_num_devices()):
                for algo in [1, 2, 3]:
                    histo_cudh = cudh.histograms(coords, r_max, n_bins, precision="double",
                                                 gpu_id=gpu_id, algorithm=algo)
                    util.compare(histo_ref, histo_cudh)

        def test_cudh_large_single(fixture_large):
            n_atoms, n_bins, coords, histo_ref = fixture_large
            for gpu_id in range(cudh.get_num_devices()):
                for algo in [1, 2, 3]:
                    histo_cudh = cudh.histograms(coords, r_max, n_bins, precision="single",
                                                 gpu_id=gpu_id, algorithm=algo)
                    util.compare(histo_ref, histo_cudh)


if TEST_XLARGE:
    testcase_xlarge = None

    @pytest.fixture
    def fixture_xlarge():
        global testcase_xlarge
        if testcase_xlarge is None:
            n_atoms = [250000, 275000, 225000]
            n_bins = 18000
            coords = util.generate_random_coordinate_set(n_atoms)
            # --- note that we use pydh to generate the test dataset, max number of threads and no blocking
            histo_ref = pydh.histograms(coords, r_max, n_bins, precision="double",
                                        n_threads=n_threads[-1], blocksize=-1)
            testcase_xlarge = (n_atoms, n_bins, coords, histo_ref)
        return testcase_xlarge

    if TEST_PYDH:
        def test_n_threads_xlarge_single(fixture_xlarge):
            n_atoms, n_bins, coords, histo_ref = fixture_xlarge
            for nt in n_threads:
                histo_pydh = pydh.histograms(coords, r_max, n_bins, precision="single", n_threads=nt)
                util.compare(histo_ref, histo_pydh)

    if TEST_CUDH:
        def test_cudh_xlarge_double(fixture_xlarge):
            n_atoms, n_bins, coords, histo_ref = fixture_xlarge
            for gpu_id in range(cudh.get_num_devices()):
                for algo in [1, 2, 3]:
                    histo_cudh = cudh.histograms(coords, r_max, n_bins, precision="double",
                                                 gpu_id=gpu_id, algorithm=algo)
                    util.compare(histo_ref, histo_cudh)

        def test_cudh_xlarge_single(fixture_xlarge):
            n_atoms, n_bins, coords, histo_ref = fixture_xlarge
            for gpu_id in range(cudh.get_num_devices()):
                for algo in [1, 2, 3]:
                    histo_cudh = cudh.histograms(coords, r_max, n_bins, precision="single",
                                                 gpu_id=gpu_id, algorithm=algo)
                    util.compare(histo_ref, histo_cudh)
