#!/usr/bin/env python2.7

import os
import sys
import numpy as np
import math
import pytest
from .. import util
from ..kernel import pydh


# number of points
n = 1000
# cutoff radius for the spherical point set
R = 1.0
# orthorhombic box specifications
mini_cube = [0.5, 0.5, 0.5, 90., 90., 90.]
unit_cube = [1.0, 1.0, 1.0, 90., 90., 90.]
huge_cube = [4.0, 4.0, 4.0, 90., 90., 90.]


# triclinic test boxes
a = 0.41
b = 0.39
c = 0.42
# alpha = 87.5
# beta = 89.
# gamma = 88.
alpha = 60.
beta = 60.
gamma = 60.
mini_triclinic = [a, b, c, alpha, beta, gamma]
unit_triclinic = [1.0, 1.0, 1.0, alpha, beta, gamma]
huge_triclinic = [10.*a, 10.*b, 10.*c, alpha, beta, gamma]

# precisions to be tested below
precision = ["single", "double"]  # ["double"]  #
tolerance = {"single" : 1.e-7,
             "double" : 1.e-15}

setup = None
@pytest.fixture
def random_coordinates():
    global setup
    if setup is None:
        setup = np.random.rand(n, 3)
    return setup


setup_sphere = None
@pytest.fixture
def random_spherical_coordinates():
    global setup_sphere
    if setup_sphere is None:
        coords = []
        for i in xrange(n):
            coords.append(util.generate_random_point_in_sphere(R))
        setup_sphere = np.asarray(coords)
    return setup_sphere


setup_spherical_surface = None
@pytest.fixture
def random_spherical_surface():
    global setup_spherical_surface
    if setup_spherical_surface is None:
        coords = []
        for i in xrange(n):
            coords.append(util.generate_random_point_on_spherical_surface(R))
        setup_spherical_surface = np.asarray(coords)
    return setup_spherical_surface


def test_dist_simple(random_coordinates):
    """run the distance calculation on a data set with coordinates between 0 and 1"""
    for prec in precision:
        distances = pydh.dist_driver(random_coordinates, precision=prec)
    #     print distances


def test_dist_simple_sphere(random_spherical_coordinates):
    """run the distance calculation on a simple spherically bounded data set"""
    for prec in precision:
        distances = pydh.dist_driver(random_spherical_coordinates, precision=prec)
    #     print distances


# def test_dist_simple_triclinic(random_coordinates):
#     """run the distance calculation on a data set with coordinates between 0 and 1"""
#     for prec in precision:
#         # distances = pydh.dist_driver(random_coordinates, precision=prec, box=mini_cube, force_triclinic=True)
#         distances = pydh.dist_driver(random_coordinates, precision=prec, box=mini_triclinic)
#         # print distances


# def test_dist_unit_box(random_spherical_coordinates):
#     """check if the orthorhombic and triclinic box implementations give the same answer for a simple unit box"""
#     for prec in precision:
#         distances_ortho = pydh.dist_driver(random_spherical_coordinates, precision=prec, box=unit_cube)
#         distances_tricl = pydh.dist_driver(random_spherical_coordinates, precision=prec, box=unit_cube, force_triclinic=True)
# #         print max(distances_ortho - distances_tricl)
#         assert(np.allclose(distances_ortho, distances_tricl, atol=tolerance[prec]))
#
#
# def test_dist_mini_box(random_spherical_coordinates):
#     """check if the orthorhombic and triclinic box implementations give the same answer for a simple small box"""
#     for prec in precision:
#         distances_ortho = pydh.dist_driver(random_spherical_coordinates, precision=prec, box=mini_cube)
#         distances_tricl = pydh.dist_driver(random_spherical_coordinates, precision=prec, box=mini_cube, force_triclinic=True)
# #         print max(distances_ortho), max(distances_tricl)
#         assert(np.allclose(distances_ortho, distances_tricl, atol=tolerance[prec]))
#
#
# def test_dist_huge_box(random_spherical_coordinates):
#     """check if the orthorhombic and triclinic box implementations give the same answer for a simple small box"""
#     for prec in precision:
#         distances_nobox = pydh.dist_driver(random_spherical_coordinates, precision=prec)
#         distances_ortho = pydh.dist_driver(random_spherical_coordinates, precision=prec, box=huge_cube)
#         distances_tricl = pydh.dist_driver(random_spherical_coordinates, precision=prec, box=huge_cube, force_triclinic=True)
# #         print max(distances_ortho), max(distances_tricl)
#         assert(np.allclose(distances_ortho, distances_nobox, atol=tolerance[prec]))
#         assert(np.allclose(distances_ortho, distances_tricl, atol=tolerance[prec]))


# --- the following tests are (partly) wrong ---


# def test_dist_huge_triclinic(random_spherical_coordinates):
#     """check if the orthorhombic and triclinic box implementations give the same answer for a simple huge box"""
#     global setup_sphere
# #     print setup_sphere[0]
#     setup_sphere += np.asarray([10., 10., 10.])
# #     print setup_sphere[0]
#     for prec in ['single']:  # precision:
#         distances_nobox = pydh.dist_driver(random_spherical_coordinates, precision=prec, box=[])
#         distances_tricl = pydh.dist_driver(random_spherical_coordinates, precision=prec, box=huge_triclinic)
# #         print max(distances_nobox - distances_tricl)
#         assert(np.allclose(distances_nobox, distances_tricl, atol=tolerance[prec]))


# def test_dist_unit_box_spherical_surface(random_spherical_surface):
#     """check if the orthorhombic and triclinic box implementations give the same answer for a simple huge box"""
#     global setup_spherical_surface
# #     print setup_sphere[0]
#     # shrink the sphere to make it fit into half of the the box
#     setup_spherical_surface *= 0.25
#     for prec in precision:
#         distances_nobox = pydh.dist_driver(random_spherical_surface, precision=prec, box=[])
#         distances_ortho = pydh.dist_driver(random_spherical_surface, precision=prec, box=unit_cube)
#         distances_tricl = pydh.dist_driver(random_spherical_surface, precision=prec, box=unit_cube, force_triclinic=True)
# #         print max(distances_nobox - distances_tricl)
#         assert(np.allclose(distances_nobox, distances_ortho, atol=tolerance[prec]))
#         assert(np.allclose(distances_nobox, distances_tricl, atol=tolerance[prec]))
#     setup_spherical_surface = None
#
#
# def test_dist_huge_box_spherical_surface(random_spherical_surface):
#     """check if the orthorhombic and triclinic box implementations give the same answer for a simple huge box"""
#     global setup_spherical_surface
# #     print setup_sphere[0]
#     # shrink the sphere to make it fit into half of the the box
#     setup_spherical_surface *= 2.01
#     for prec in ['double']:  # precision:
#         distances_nobox = pydh.dist_driver(random_spherical_surface, precision=prec, box=[])
#         distances_ortho = pydh.dist_driver(random_spherical_surface, precision=prec, box=huge_cube, force_triclinic=True)
#         print max(distances_nobox - distances_ortho)
# #         assert(np.allclose(distances_nobox, distances_ortho, atol=tolerance[prec]))
#         max_nobox = np.max(distances_nobox)
#         max_ortho = np.max(distances_ortho)
#         print("max(nobox) = {}, max(ortho) = {}".format(max_nobox, max_ortho))
#     setup_spherical_surface = None
#
#
# def test_dist_huge_triclinic_spherical_surface(random_spherical_surface):
#     """check if the orthorhombic and triclinic box implementations give the same answer for a simple huge box"""
#     global setup_spherical_surface
# #     print setup_sphere[0]
#     # shrink the sphere to make it fit into half of the the box
#     setup_spherical_surface *= 0.95
#     for prec in ['double']:  # precision:
#         distances_nobox = pydh.dist_driver(random_spherical_surface, precision=prec, box=[])
#         distances_tricl = pydh.dist_driver(random_spherical_surface, precision=prec, box=huge_triclinic)
# #         print max(distances_nobox - distances_tricl)
# #         assert(np.allclose(distances_nobox, distances_tricl, atol=tolerance[prec]))
#         max_nobox = np.max(distances_nobox)
#         max_tricl = np.max(distances_tricl)
#         print("max(nobox) = {}, max(triclinic) = {}".format(max_nobox, max_tricl))
#     setup_spherical_surface = None
