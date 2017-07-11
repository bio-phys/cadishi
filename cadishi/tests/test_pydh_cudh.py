#!/usr/bin/env python2.7

"""A set of unit tests for the pydh CPU and cudh GPU histogram modules.

This file is part of the Cadishi package.  See README.rst,
LICENSE.txt, and the documentation for details.
"""


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


# --- import the dist module which serves as the reference implementation
from cadishi.kernel import dist


# --- import the pydh module
pydh_threads = [2, 3, 4]
pydh_threads.append(multiprocessing.cpu_count())
pydh_threads = sorted(list(set(pydh_threads)))

from cadishi.kernel import pydh


# --- import the cudh module
if TEST_CUDH:
    try:
        from cadishi.kernel import cudh
    except Exception as e:
        print "Error importing >> cudh <<.  Disabling CUDA tests."
        print "Exception message : " + e.message
        TEST_CUDH = False

if TEST_CUDH:
    # test if we are able to run the tests at all
    if (cudh.get_num_devices() == 0):
        print "No usable CUDA device detected.  Disabling CUDA tests."
        TEST_CUDH = False
    else:
        print "CUDA tests: " + str(cudh.get_num_devices()) + " GPUs detected."


def get_triclinic_box():
    return np.asarray([0.96, 0.98, 1.00, 60., 60., 90.])


def get_orthorhombix_box():
    box = np.zeros((3, 3))
    box[0][0] = 0.96
    box[1][1] = 0.98
    box[2][2] = 1.00
    return box




# testcase_small = None
# @pytest.fixture
# def fixture_small():
#     global testcase_small
#     if testcase_small is None:
#         n_el = 2
#         n_atoms = [2000, 1000]
#         n_bins = 2048
#         coords = util.generate_random_coordinate_set(n_atoms)
#         histo = dist.histograms(coords, r_max, n_bins)
#         # if DUMP_DATA:
#         #     file_name = sys._getframe().f_code.co_name + ".dat"
#         #     util.dump_histograms(file_name, histo, r_max, n_bins)
#         testcase_small = (n_el, n_atoms, n_bins, coords, histo)
#     return testcase_small

# testcase_small_orthorhombic = None
# @pytest.fixture
# def fixture_small_orthorhombic():
#     global testcase_small_orthorhombic
#     if testcase_small_orthorhombic is None:
#         n_el = 2
#         n_atoms = [2000, 1000]
#         n_bins = 2048
#         coords = util.generate_random_coordinate_set(n_atoms)
#         box = get_orthorhombix_box()
#         histo = pydh.histograms(coords, r_max, n_bins, box=box, precision="double", pydh_threads=1)
#         # if DUMP_DATA:
#         #     file_name = sys._getframe().f_code.co_name + ".dat"
#         #     util.dump_histograms(file_name, histo, r_max, n_bins)
#         testcase_small_orthorhombic = (n_el, n_atoms, n_bins, coords, box, histo)
#     return testcase_small_orthorhombic
#
# testcase_small_triclinic = None
# @pytest.fixture
# def fixture_small_triclinic():
#     global testcase_small_triclinic
#     if testcase_small_triclinic is None:
#         n_el = 2
#         # n_atoms = [2000, 1000]
#         # n_bins = 2048
#         n_atoms = [10000]
#         n_bins = 8192
#         coords = util.generate_random_coordinate_set(n_atoms)
#         box = get_triclinic_box()
#         histo = pydh.histograms(coords, r_max, n_bins, box=box, precision="double", pydh_threads=1)
#         if DUMP_DATA:
#             file_name = sys._getframe().f_code.co_name + ".dat"
#             util.dump_histograms(file_name, histo, r_max, n_bins)
#         testcase_small_triclinic = (n_el, n_atoms, n_bins, coords, box, histo)
#     return testcase_small_triclinic
#
# if TEST_PYDH:
#     # def test_pydh_small_double(fixture_small):
#     #     n_el, n_atoms, n_bins, coords, histo_ref = fixture_small
#     #     for check_input in [True, False]:
#     #         histo_pydh = pydh.histograms(coords, r_max, n_bins, precision="double", pydh_threads=1, check_input=check_input)
#     #         util.compare_strictly(histo_ref, histo_pydh)
#     #
#     # def test_pydh_threads_small_double(fixture_small):
#     #     n_el, n_atoms, n_bins, coords, histo_ref = fixture_small
#     #     for check_input in [True, False]:
#     #         for nt in pydh_threads:
#     #             histo_pydh = pydh.histograms(coords, r_max, n_bins, precision="double", pydh_threads=nt, check_input=check_input)
#     #             util.compare_strictly(histo_ref, histo_pydh)
#     #
#     # def test_pydh_small_single(fixture_small):
#     #     n_el, n_atoms, n_bins, coords, histo_ref = fixture_small
#     #     for check_input in [True, False]:
#     #         histo_pydh = pydh.histograms(coords, r_max, n_bins, precision="single", pydh_threads=1, check_input=check_input)
#     #         util.compare_approximately(histo_ref, histo_pydh)
#     #
#     # def test_pydh_threads_small_single(fixture_small):
#     #     n_el, n_atoms, n_bins, coords, histo_ref = fixture_small
#     #     for check_input in [True, False]:
#     #         for nt in pydh_threads:
#     #             histo_pydh = pydh.histograms(coords, r_max, n_bins, precision="single", pydh_threads=nt, check_input=check_input)
#     #             util.compare_approximately(histo_ref, histo_pydh)
#     #
#     # def test_pydh_small_orthorhombic_single(fixture_small_orthorhombic):
#     #     n_el, n_atoms, n_bins, coords, box, histo_ref = fixture_small_orthorhombic
#     #     for check_input in [True, False]:
#     #         histo = pydh.histograms(coords, r_max, n_bins, box=box, precision="single", pydh_threads=1, check_input=check_input)
#     #         util.compare_approximately(histo_ref, histo)
#     #         if DUMP_DATA:
#     #             file_name = sys._getframe().f_code.co_name + ".dat"
#     #             util.dump_histograms(file_name, histo, r_max, n_bins)
#
#     def test_pydh_small_triclinic_single(fixture_small_triclinic):
#         n_el, n_atoms, n_bins, coords, box, histo_ref = fixture_small_triclinic
#         for check_input in [True, False]:
#             histo = pydh.histograms(coords, r_max, n_bins, box=box, precision="single", pydh_threads=1, check_input=check_input)
#             util.compare_approximately(histo_ref, histo)
#             if DUMP_DATA:
#                 file_name = sys._getframe().f_code.co_name + ".dat"
#                 util.dump_histograms(file_name, histo, r_max, n_bins)
#
# #
# # if TEST_CUDH:
# #     def test_cudh_small_double(fixture_small):
# #         n_el, n_atoms, n_bins, coords, histo_ref = fixture_small
# #         for check_input in [True, False]:
# #             for gpu_id in range(cudh.get_num_devices()):
# #                 for algo in [1,2,3]:
# #                     histo = cudh.histograms(coords, r_max, n_bins, precision="double", gpu_id=gpu_id, check_input=check_input, algorithm=algo)
# #                     util.compare_strictly(histo_ref, histo)
# #                     if DUMP_DATA:
# #                         file_name = sys._getframe().f_code.co_name + "_gpu" + str(gpu_id) + ".dat"
# #                         util.dump_histograms(file_name, histo, r_max, n_bins)
# #
# #     def test_cudh_small_single(fixture_small):
# #         n_el, n_atoms, n_bins, coords, histo_ref = fixture_small
# #         for check_input in [True, False]:
# #             for gpu_id in range(cudh.get_num_devices()):
# #                 for algo in [1,2,3]:
# #                     histo = cudh.histograms(coords, r_max, n_bins, precision="single", gpu_id=gpu_id, check_input=check_input, algorithm=algo)
# #                     util.compare_approximately(histo_ref, histo)
# #                     if DUMP_DATA:
# #                         file_name = sys._getframe().f_code.co_name + "_gpu" + str(gpu_id) + ".dat"
# #                         util.dump_histograms(file_name, histo, r_max, n_bins)
# #
# #     def test_cudh_small_orthorhombic_single(fixture_small_orthorhombic):
# #         n_el, n_atoms, n_bins, coords, box, histo_ref = fixture_small_orthorhombic
# #         for check_input in [True, False]:
# #             histo = cudh.histograms(coords, r_max, n_bins, box=box, precision="single", check_input=check_input)
# #             util.compare_approximately(histo_ref, histo)
# #             if DUMP_DATA:
# #                 file_name = sys._getframe().f_code.co_name + ".dat"
# #                 util.dump_histograms(file_name, histo, r_max, n_bins)
# #
# #     def test_cudh_small_triclinic_single(fixture_small_triclinic):
# #         n_el, n_atoms, n_bins, coords, box, histo_ref = fixture_small_triclinic
# #         for check_input in [True, False]:
# #             histo = cudh.histograms(coords, r_max, n_bins, box=box, precision="single", check_input=check_input)
# #             util.compare_approximately(histo_ref, histo)
# #             if DUMP_DATA:
# #                 file_name = sys._getframe().f_code.co_name + ".dat"
# #                 util.dump_histograms(file_name, histo, r_max, n_bins)
# #
# #
# # testcase_small_invalid = None
# # @pytest.fixture
# # def fixture_small_invalid():
# #     global testcase_small_invalid
# #     if testcase_small_invalid is None:
# #         n_el = 2
# #         n_atoms = [2000, 1000]
# #         n_bins = 2048
# #         coords = util.generate_random_coordinate_set(n_atoms, create_invalid=True)
# #         histo_ref = None
# #         testcase_small_invalid = (n_el, n_atoms, n_bins, coords, histo_ref)
# #     return testcase_small_invalid
# #
# # if TEST_PYDH:
# #     def test_pydh_invalid_small_double(fixture_small_invalid):
# #         n_el, n_atoms, n_bins, coords, histo_ref = fixture_small_invalid
# #         with pytest.raises(ValueError):
# #             pydh.histograms(coords, r_max, n_bins, precision="double", pydh_threads=1, check_input=True)
# #
# #     def test_pydh_invalid_threads_small_double(fixture_small_invalid):
# #         n_el, n_atoms, n_bins, coords, histo_ref = fixture_small_invalid
# #         for nt in pydh_threads:
# #             with pytest.raises(ValueError):
# #                 pydh.histograms(coords, r_max, n_bins, precision="double", pydh_threads=nt, check_input=True)
# #
# #     def test_pydh_invalid_small_single(fixture_small_invalid):
# #         n_el, n_atoms, n_bins, coords, histo_ref = fixture_small_invalid
# #         with pytest.raises(ValueError):
# #             pydh.histograms(coords, r_max, n_bins, precision="single", pydh_threads=1, check_input=True)
# #
# #     def test_pydh_invalid_threads_small_single(fixture_small_invalid):
# #         n_el, n_atoms, n_bins, coords, histo_ref = fixture_small_invalid
# #         for nt in pydh_threads:
# #             with pytest.raises(ValueError):
# #                 pydh.histograms(coords, r_max, n_bins, precision="single", pydh_threads=nt, check_input=True)
# #
# #
# # if TEST_CUDH:
# #     def test_cudh_invalid_small_double(fixture_small_invalid):
# #         n_el, n_atoms, n_bins, coords, histo_ref = fixture_small_invalid
# #         for gpu_id in range(cudh.get_num_devices()):
# #             with pytest.raises(ValueError):
# #                 cudh.histograms(coords, r_max, n_bins, precision="double", gpu_id=gpu_id, check_input=True)
# #
# #     def test_cudh_invalid_small_single(fixture_small_invalid):
# #         n_el, n_atoms, n_bins, coords, histo_ref = fixture_small_invalid
# #         for gpu_id in range(cudh.get_num_devices()):
# #             with pytest.raises(ValueError):
# #                 cudh.histograms(coords, r_max, n_bins, precision="single", gpu_id=gpu_id, check_input=True)
# #
# #
# # # test case designed to select the tiled kernels in cudh
# # testcase_medium = None
# # @pytest.fixture
# # def fixture_medium():
# #     global testcase_medium
# #     if testcase_medium is None:
# #         n_el = 5
# #         n_atoms = [3000, 1000, 5000, 3500, 6000]
# #         n_bins = 8192
# #         coords = util.generate_random_coordinate_set(n_atoms)
# #         histo_ref = dist.histograms(coords, r_max, n_bins)
# #         if DUMP_DATA:
# #             util.dump_histograms("histo_ref_medium.dat", histo_ref, r_max, n_bins)
# #         testcase_medium = (n_el, n_atoms, n_bins, coords, histo_ref)
# #     return testcase_medium
# #
# # if TEST_PYDH:
# #     def test_pydh_medium_double(fixture_medium):
# #         n_el, n_atoms, n_bins, coords, histo_ref = fixture_medium
# #         for check_input in [True, False]:
# #             histo_pydh = pydh.histograms(coords, r_max, n_bins, precision="double", pydh_threads=1, check_input=check_input)
# #             util.compare_strictly(histo_ref, histo_pydh)
# #
# #     def test_pydh_threads_medium_double(fixture_medium):
# #         n_el, n_atoms, n_bins, coords, histo_ref = fixture_medium
# #         for check_input in [True, False]:
# #             for nt in pydh_threads:
# #                 histo_pydh = pydh.histograms(coords, r_max, n_bins, precision="double", pydh_threads=nt, check_input=check_input)
# #                 util.compare_strictly(histo_ref, histo_pydh)
# #
# #     def test_pydh_medium_single(fixture_medium):
# #         n_el, n_atoms, n_bins, coords, histo_ref = fixture_medium
# #         for check_input in [True, False]:
# #             histo_pydh = pydh.histograms(coords, r_max, n_bins, precision="single", pydh_threads=1, check_input=check_input)
# #             util.compare_approximately(histo_ref, histo_pydh)
# #
# #     def test_pydh_threads_medium_single(fixture_medium):
# #         n_el, n_atoms, n_bins, coords, histo_ref = fixture_medium
# #         for check_input in [True, False]:
# #             for nt in pydh_threads:
# #                 histo_pydh = pydh.histograms(coords, r_max, n_bins, precision="single", pydh_threads=nt, check_input=check_input)
# #                 util.compare_approximately(histo_ref, histo_pydh)
# #
# #     def test_pydh_medium_masked_single(fixture_medium):
# #         n_el, n_atoms, n_bins, coords, histo_ref = fixture_medium
# #         mask_array = np.ones(n_el * (n_el + 1) / 2)
# #         mask_array[::2] = 0
# #         for check_input in [True, False]:
# #             histo_pydh = pydh.histograms(coords, r_max, n_bins, precision="single", pydh_threads=1, mask_array=mask_array, check_input=check_input)
# #             col_sum = histo_pydh.sum(axis=0)
# #             assert(np.sum(col_sum[1::2]) == 0)
# #
# #     def test_pydh_medium_scaled_single(fixture_medium):
# #         n_el, n_atoms, n_bins, coords, histo_ref = fixture_medium
# #         scale_factors = np.ones(n_el * (n_el + 1) / 2)
# #         scale_factors *= 0.5
# #         for check_input in [True, False]:
# #             histo_pydh = pydh.histograms(coords, r_max, n_bins, precision="single", pydh_threads=1, scale_factors=scale_factors, check_input=check_input)
# #             assert(histo_ref.sum() == 2.0 * histo_pydh.sum())
# #
# # if TEST_CUDH:
# #     def test_cudh_medium_double(fixture_medium):
# #         n_el, n_atoms, n_bins, coords, histo_ref = fixture_medium
# #         for check_input in [True, False]:
# #             for gpu_id in range(cudh.get_num_devices()):
# #                 for algo in [1,2,3]:
# #                     histo_cudh = cudh.histograms(coords, r_max, n_bins, precision="double", gpu_id=gpu_id, check_input=check_input, algorithm=algo)
# #                     util.compare_strictly(histo_ref, histo_cudh)
# #
# #     def test_cudh_medium_single(fixture_medium):
# #         n_el, n_atoms, n_bins, coords, histo_ref = fixture_medium
# #         for check_input in [True, False]:
# #             for gpu_id in range(cudh.get_num_devices()):
# #                 for algo in [1,2,3]:
# #                     histo_cudh = cudh.histograms(coords, r_max, n_bins, precision="single", gpu_id=gpu_id, check_input=check_input, algorithm=algo)
# #                     util.compare_approximately(histo_ref, histo_cudh)
# #
# #     def test_cudh_medium_masked_single(fixture_medium):
# #         n_el, n_atoms, n_bins, coords, histo_ref = fixture_medium
# #         mask_array = np.ones(n_el * (n_el + 1) / 2)
# #         mask_array[::2] = 0
# #         for check_input in [True, False]:
# #             for gpu_id in range(cudh.get_num_devices()):
# #                 for algo in [1,2,3]:
# #                     histo_cudh = cudh.histograms(coords, r_max, n_bins, precision="single", gpu_id=gpu_id, mask_array=mask_array, check_input=check_input, algorithm=algo)
# #                     col_sum = histo_cudh.sum(axis=0)
# #                     assert(np.sum(col_sum[1::2]) == 0)
# #
# #     def test_cudh_medium_scaled_single(fixture_medium):
# #         n_el, n_atoms, n_bins, coords, histo_ref = fixture_medium
# #         scale_factors = np.ones(n_el * (n_el + 1) / 2)
# #         scale_factors *= 0.5
# #         for check_input in [True, False]:
# #             for gpu_id in range(cudh.get_num_devices()):
# #                 for algo in [1,2,3]:
# #                     histo_cudh = cudh.histograms(coords, r_max, n_bins, precision="single", gpu_id=gpu_id, scale_factors=scale_factors, check_input=check_input, algorithm=algo)
# #                     assert(histo_ref.sum() == 2.0 * histo_cudh.sum())
# #
# #
# # # test case designed to select the simple kernels in cudh
# # testcase_medium_manybins = None
# # @pytest.fixture
# # def fixture_medium_manybins():
# #     global testcase_medium_manybins
# #     if testcase_medium_manybins is None:
# #         n_el = 5
# #         n_atoms = [3000, 1000, 5000, 3500, 6000]
# #         n_bins = 68000
# #         coords = util.generate_random_coordinate_set(n_atoms)
# #         histo_ref = dist.histograms(coords, r_max, n_bins)
# #         if DUMP_DATA:
# #             util.dump_histograms("histo_ref_medium.dat", histo_ref, r_max, n_bins)
# #         testcase_medium_manybins = (n_el, n_atoms, n_bins, coords, histo_ref)
# #     return testcase_medium_manybins
# #
# # if TEST_PYDH:
# #     def test_pydh_medium_manybins_double(fixture_medium_manybins):
# #         n_el, n_atoms, n_bins, coords, histo_ref = fixture_medium_manybins
# #         for check_input in [True, False]:
# #             histo_pydh = pydh.histograms(coords, r_max, n_bins, precision="double", pydh_threads=1, check_input=check_input)
# #             util.compare_strictly(histo_ref, histo_pydh)
# #
# #     def test_pydh_threads_medium_manybins_double(fixture_medium_manybins):
# #         n_el, n_atoms, n_bins, coords, histo_ref = fixture_medium_manybins
# #         for check_input in [True, False]:
# #             for nt in pydh_threads:
# #                 histo_pydh = pydh.histograms(coords, r_max, n_bins, precision="double", pydh_threads=nt, check_input=check_input)
# #                 util.compare_strictly(histo_ref, histo_pydh)
# #
# #     def test_pydh_medium_manybins_single(fixture_medium_manybins):
# #         n_el, n_atoms, n_bins, coords, histo_ref = fixture_medium_manybins
# #         for check_input in [True, False]:
# #             histo_pydh = pydh.histograms(coords, r_max, n_bins, precision="single", pydh_threads=1, check_input=check_input)
# #             util.compare_approximately(histo_ref, histo_pydh)
# #
# #     def test_pydh_threads_medium_manybins_single(fixture_medium_manybins):
# #         n_el, n_atoms, n_bins, coords, histo_ref = fixture_medium_manybins
# #         for check_input in [True, False]:
# #             for nt in pydh_threads:
# #                 histo_pydh = pydh.histograms(coords, r_max, n_bins, precision="single", pydh_threads=nt, check_input=check_input)
# #                 util.compare_approximately(histo_ref, histo_pydh)
# #
# # if TEST_CUDH:
# #     def test_cudh_medium_manybins_double(fixture_medium_manybins):
# #         n_el, n_atoms, n_bins, coords, histo_ref = fixture_medium_manybins
# #         for check_input in [True, False]:
# #             for gpu_id in range(cudh.get_num_devices()):
# #                 for algo in [1,2,3]:
# #                     histo_cudh = cudh.histograms(coords, r_max, n_bins, precision="double", gpu_id=gpu_id, check_input=check_input, algorithm=algo)
# #                     util.compare_strictly(histo_ref, histo_cudh)
# #
# #     def test_cudh_medium_manybins_single(fixture_medium_manybins):
# #         n_el, n_atoms, n_bins, coords, histo_ref = fixture_medium_manybins
# #         for check_input in [True, False]:
# #             for gpu_id in range(cudh.get_num_devices()):
# #                 for algo in [1,2,3]:
# #                     histo_cudh = cudh.histograms(coords, r_max, n_bins, precision="single", gpu_id=gpu_id, check_input=check_input, algorithm=algo)
# #                     util.compare_approximately(histo_ref, histo_cudh)
# #
# #
# #
# # if TEST_LARGE:
# #     testcase_large = None
# #     @pytest.fixture
# #     def fixture_large():
# #         global testcase_large
# #         if testcase_large is None:
# #             n_el = 8
# #             n_atoms = [3000, 15000, 35500, 6000, 15000, 500, 7500, 100]
# #             n_bins = 12000
# #             coords = util.generate_random_coordinate_set(n_atoms)
# #             # --- note that we use pydh to generate the test dataset
# #             histo_ref = pydh.histograms(coords, r_max, n_bins, precision="double", pydh_threads=pydh_threads[-1])
# #             if DUMP_DATA:
# #                 file_name = sys._getframe().f_code.co_name + ".dat"
# #                 util.dump_histograms(file_name, histo_ref, r_max, n_bins)
# #             testcase_large = (n_el, n_atoms, n_bins, coords, histo_ref)
# #         return testcase_large
# #
# #     if TEST_PYDH:
# #         def test_pydh_large_single(fixture_large):
# #             n_el, n_atoms, n_bins, coords, histo_ref = fixture_large
# #             histo_pydh = pydh.histograms(coords, r_max, n_bins, precision="single", pydh_threads=1)
# #             util.compare_approximately(histo_ref, histo_pydh)
# #
# #         def test_pydh_threads_large_single(fixture_large):
# #             n_el, n_atoms, n_bins, coords, histo_ref = fixture_large
# #             for nt in pydh_threads:
# #                 histo_pydh = pydh.histograms(coords, r_max, n_bins, precision="single", pydh_threads=nt)
# #                 util.compare_approximately(histo_ref, histo_pydh)
# #
# #     if TEST_CUDH:
# #         def test_cudh_large_double(fixture_large):
# #             n_el, n_atoms, n_bins, coords, histo_ref = fixture_large
# #             for gpu_id in range(cudh.get_num_devices()):
# #                 for algo in [1,2,3]:
# #                     histo_cudh = cudh.histograms(coords, r_max, n_bins, precision="double", gpu_id=gpu_id, algorithm=algo)
# #                     util.compare_strictly(histo_ref, histo_cudh)
# #
# #         def test_cudh_large_single(fixture_large):
# #             n_el, n_atoms, n_bins, coords, histo_ref = fixture_large
# #             for gpu_id in range(cudh.get_num_devices()):
# #                 for algo in [1,2,3]:
# #                     histo_cudh = cudh.histograms(coords, r_max, n_bins, precision="single", gpu_id=gpu_id, algorithm=algo)
# #                     util.compare_approximately(histo_ref, histo_cudh)
# #
# #
# # if TEST_XLARGE:
# #     testcase_xlarge = None
# #     @pytest.fixture
# #     def fixture_xlarge():
# #         global testcase_xlarge
# #         if testcase_xlarge is None:
# #             n_el = 10
# #             n_atoms = [10000, 15000, 35500, 50000,
# #                        500, 7500, 100, 75000, 150000, 225000]
# #             n_bins = 18000
# #             coords = util.generate_random_coordinate_set(n_atoms)
# #             # --- note that we use pydh to generate the test dataset
# #             histo_ref = pydh.histograms(coords, r_max, n_bins, precision="double", pydh_threads=pydh_threads[-1])
# #             testcase_xlarge = (n_el, n_atoms, n_bins, coords, histo_ref)
# #         return testcase_xlarge
# #
# #     if TEST_PYDH:
# #         def test_pydh_threads_xlarge_single(fixture_xlarge):
# #             n_el, n_atoms, n_bins, coords, histo_ref = fixture_xlarge
# #             for nt in pydh_threads:
# #                 histo_pydh = pydh.histograms(coords, r_max, n_bins, precision="single", pydh_threads=nt)
# #                 util.compare_approximately(histo_ref, histo_pydh)
# #
# #     if TEST_CUDH:
# #         def test_cudh_xlarge_double(fixture_xlarge):
# #             n_el, n_atoms, n_bins, coords, histo_ref = fixture_xlarge
# #             for gpu_id in range(cudh.get_num_devices()):
# #                 for algo in [1,2,3]:
# #                     histo_cudh = cudh.histograms(coords, r_max, n_bins, precision="double", gpu_id=gpu_id, algorithm=algo)
# #                     util.compare_strictly(histo_ref, histo_cudh)
# #
# #         def test_cudh_xlarge_single(fixture_xlarge):
# #             n_el, n_atoms, n_bins, coords, histo_ref = fixture_xlarge
# #             for gpu_id in range(cudh.get_num_devices()):
# #                 for algo in [1,2,3]:
# #                     histo_cudh = cudh.histograms(coords, r_max, n_bins, precision="single", gpu_id=gpu_id, algorithm=algo)
# #                     util.compare_approximately(histo_ref, histo_cudh)
