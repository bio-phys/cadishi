# vim:fileencoding=utf-8
"""Code related to periodic boxes.
"""
# This file is part of the Cadishi package.  See README.rst,
# LICENSE.txt, and the documentation for details.


import numpy as np


# --- translate string into C enum for interfacing, see <cpp_common.hpp>
_enum_box_type = {}
_enum_box_type['none'] = 0
_enum_box_type['orthorhombic'] = 1
_enum_box_type['triclinic'] = 2
# for compatibility with MDAnalysis:
_enum_box_type[None] = 0
_enum_box_type['ortho'] = 1
_enum_box_type['tri_box'] = 2


def _box_type_to_enum(str_id):
    """Helper function to translate a string box identification to the integer
    identification used inside the kernels PYDH and CUDH."""
    return _enum_box_type[str_id]


def _triclinic_angles_to_matrix(box_tup):
    """Convert a length-angle box representation to a matrix box representation.
    Credit: This implementation was inspired by a mailing list post at
    <https://www.mail-archive.com/gmx-users@gromacs.org/msg28032.html>,
    similar to the MDAnalysis triclinic_vectors() implementation.
    Further reference: https://en.wikipedia.org/wiki/Fractional_coordinates
    """
    box_mat = np.zeros((3, 3))
    if np.all(box_tup[:3] == 0.):
        pass
    else:
        x, y, z, a_deg, b_deg, c_deg = box_tup[:6]
        box_mat[0][0] = x
        if (a_deg == 90.) and (b_deg == 90.) and (c_deg == 90.):
            box_mat[1][1] = y
            box_mat[2][2] = z
        else:
            a = np.deg2rad(a_deg)
            b = np.deg2rad(b_deg)
            c = np.deg2rad(c_deg)
            box_mat[1][0] = y * np.cos(c)
            box_mat[1][1] = y * np.sin(c)
            box_mat[2][0] = z * np.cos(b)
            box_mat[2][1] = z * (np.cos(a) - np.cos(b) * np.cos(c)) / np.sin(c)
            box_mat[2][2] = np.sqrt(z * z - box_mat[2][0] * box_mat[2][0] - box_mat[2][1] * box_mat[2][1])
    return box_mat


def get_standard_box(box_in, force_triclinic=False, verbose=False):
    """Convert any valid and non-valid box specification into a 3x3 matrix
    that can be understood by the CUDH and PYDH kernels."""
    box = np.asarray(box_in)
    if (box.shape == (3,)) and (box[0] * box[1] * box[2] != 0.0):
        box_type = "orthorhombic"
        box_matrix = np.zeros((3, 3))
        for i in xrange(3):
            box_matrix[i, i] = box[i]
    # length-and-angles representation
    elif (box.shape == (6,)):
        if (np.count_nonzero(box) < 6):
            box_type = None
        else:
            if (not force_triclinic) and np.all(box[3:] == 90.):
                box_type = "orthorhombic"
                box_matrix = np.zeros((3, 3))
                for i in xrange(3):
                    box_matrix[i, i] = box[i]
            else:
                box_type = "triclinic"
                box_matrix = _triclinic_angles_to_matrix(box)
    # vector representation
    elif (box.shape == (3, 3)):
        if (np.count_nonzero(box) == 3) and (box[0, 0] * box[1, 1] * box[2, 2] != 0.0):
            box_type = "orthorhombic"
            box_matrix = np.asarray(box)
        elif (np.count_nonzero(box) == 6) and (box[0, 1] + box[0, 2] + box[1, 2] == 0.0):
            box_type = "triclinic"
            box_matrix = np.asarray(box)
        else:
            box_type = None
    else:
        box_type = None
    # ---
    if box_type is None:
        box_matrix = np.zeros((3, 3))
    box_id = _box_type_to_enum(box_type)
    if (verbose):
        print("box_type=" + str(box_type) + ", box_matrix=" + str(box_matrix))
    return (box_matrix, box_id, box_type)


def get_box_volume(box_in):
    box = np.asanyarray(box_in)
    # assert(box.shape == (6,))  # works with 6-tuple box representation, coming e.g. from the MDAnalysis reader
    _box_matrix, _box_id, box_type = get_standard_box(box)
    if (box_type == "orthorhombic"):
        a = box[0]
        b = box[1]
        c = box[2]
        volume = a * b * c
    elif (box_type == "triclinic"):
        a = box[0]
        b = box[1]
        c = box[2]
        alpha = np.deg2rad(box[3])
        beta = np.deg2rad(box[4])
        gamma = np.deg2rad(box[5])
        volume = a * b * c * np.sqrt(1.0 - np.cos(alpha) ** 2 - np.cos(beta) ** 2 - np.cos(gamma) ** 2 \
                                     + 2.0 * np.cos(alpha) * np.cos(beta) * np.cos(gamma))
        # reference e.g. http://www.fxsolver.com/browse/formulas/Triclinic+crystal+system+(Unit+cell's+volume)
    else:
        volume = None
    return volume
