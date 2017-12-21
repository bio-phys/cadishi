# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4 fileencoding=utf-8
#
# Cadishi --- CAlculation of DIStance HIstograms
#
# Copyright (c) Klaus Reuter, Juergen Koefinger
# See the file AUTHORS.rst for the full list of contributors.
#
# Released under the MIT License, see the file LICENSE.txt.

"""Cadishi HDF5 IO library.

HDF5 data reader/writer for base.Container instances.  Heavily used by Cadishi
and Capriqorn.
"""


import re
import json
import h5py
import random
from six.moves import range

from .. import base
from .. import util
from .. import h5pickle

shuffle_reproducible_seed = 42


class H5Reader(base.Reader):
    """HDF5 reader returning base.Container instances."""
    _depends = []
    _conflicts = []

    def close_h5fp(self):
        if (self.file_idx_open >= 0):
            self.file_pointer.close()
#             print "H5Reader: closed " + self.file_names[self.file_idx_open]
            self.file_idx_open = -1

    def get_h5fp(self, file_idx):
        if (file_idx != self.file_idx_open):
            self.close_h5fp()
            file_name = self.file_names[file_idx]
            self.file_pointer = h5py.File(file_name, "r")
#             print "H5Reader: opened " + file_name
            self.file_idx_open = file_idx
        return self.file_pointer

    def __init__(self, file=["default.h5"], first=1, last=None, step=1,
                 shuffle=False, shuffle_reproducible=False, verbose=False):
        # update: file supports lists or tuples of multiple file names
        if isinstance(file, basestring):
            self.file_names = [file]
        else:
            self.file_names = list(file)
        #
        self.file_idx_open = -1
        self.file_pointer = None
        self.first = first
        self.last = last
        self.step = step
        self.shuffle = shuffle
        self.shuffle_reproducible = shuffle_reproducible
        self.verb = verbose
        # ---
        self._depends.extend(super(base.Reader, self)._depends)
        self._conflicts.extend(super(base.Reader, self)._conflicts)
        # --- build a list of all available frames explicitly
        self.frame_pool = []
        for file_idx in range(len(self.file_names)):
            for frame_idx in sorted(int(keys) for keys in self.get_h5fp(file_idx).keys()):
                # check if frame_idx is actually a frame number
                frame_idx = str(int(frame_idx))
                if (re.match("[0-9]+", frame_idx) == None):
                    continue
                else:
                    self.frame_pool.append((file_idx, frame_idx))
        if (self.last is not None):
            self.frame_pool = self.frame_pool[:self.last]
        if (self.first is not None):
            self.frame_pool = self.frame_pool[self.first - 1:]
        if (self.step is not None):
            self.frame_pool = self.frame_pool[::self.step]
        if self.shuffle:
            if self.shuffle_reproducible:
                random.Random(shuffle_reproducible_seed).shuffle(self.frame_pool)
            else:
                random.shuffle(self.frame_pool)
        if self.verb:
            print "H5Reader.next() : frame_pool ", self.frame_pool

    def __del__(self):
        self.close_h5fp()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.__del__

    def get_meta(self):
        """Return information on the HDF5 reader,
        ready to be added to a frame object's list of
        pipeline meta information.
        """
        meta = {}
        label = 'H5Reader'
        param = {'file': self.file_names,
                 'first': self.first, 'last': self.last, 'step': self.step,
                 'shuffle': False, 'shuffle_reproducible': False}
        meta[label] = param
        return meta

    def get_frame(self, idx_tuple):
        """Read a frame identified by its (file_idx, frame_idx)
        and return a Container object."""
        if isinstance(idx_tuple, int):
            # outside we start counting at 1, however list indexing starts at zero
            idx = idx_tuple - 1
            (file_idx, frame_idx) = self.frame_pool[idx]
        else:
            (file_idx, frame_idx) = idx_tuple
        group_name = str(int(frame_idx))
        group = self.get_h5fp(file_idx)[group_name]
        frm = base.Container()
        frm.i = int(frame_idx)  # may be changed outside with absolute numbering
        # ---
        frm.data = h5pickle.load(group)
        if 'log' in frm.data.keys():
            json_str = frm.data['log']
            del frm.data['log']
            if len(json_str) > 0:
                frm.data['log'] = json.loads(json_str)
            else:
                frm.data['log'] = []
        # ---
        return frm

    def next(self):
        """Generator yielding frame by frame (re-numbering frames from one)."""
        c = 1
        for idx_tuple in self.frame_pool:
            (file_idx, frame_idx) = idx_tuple
            frm = self.get_frame(idx_tuple)
            frm.i = c  # re-introduce numbering
            frm.put_meta(self.get_meta())
            if self.verb:
                print "H5Reader.next() : ", frm.i
            yield frm
            c += 1

    def get_trajectory_information(self):
        """Collect information from the first frame, assume it to be
        representative for all the frames, and return it via a trajectory
        information object."""
        ti = base.TrajectoryInformation()
        idx_tuple = self.frame_pool[0]
        frm = self.get_frame(idx_tuple)
        if frm.contains_key(base.loc_coordinates):
            ti.species = sorted(frm.get_keys(base.loc_coordinates))
        else:
            ti.species = []
        ti.pipeline_log = frm.get_meta()
        ti.frame_numbers = list(range(1, len(self.frame_pool) + 1))
        return ti


class H5Writer(base.Writer):
    """HDF5 writer for base.Container instances."""
    _depends = []
    _conflicts = []

    valid_compression = [None, "gzip", "lzf"]
    default_compression = "lzf"

    def __init__(self, file="default.hdf5", source=-1,
                 compression=None, mode="w", verbose=False):
        self.file = file
        util.md(file)
        try:
            self.h5fp = h5py.File(file, mode)
        except:
            self.file_is_open = False
            raise
        else:
            self.file_is_open = True
        self.src = source
        self.comp = compression
        self.verb = verbose
        self.info = ''
        # ---
        self._depends.extend(super(base.Writer, self)._depends)
        self._conflicts.extend(super(base.Writer, self)._conflicts)

    def __enter__(self):
        return self

    def __del__(self):
        self.close_file_safely()

    def __exit__(self, type, value, traceback):
        self.close_file_safely()

    def close_file_safely(self):
        if self.file_is_open:
            self.h5fp.flush()
            self.h5fp.close()
            self.file_is_open = False

    def get_meta(self):
        """Return information on the HDF5 writer,
        ready to be added to a frame object's list of
        pipeline meta information.
        """
        meta = {}
        label = 'H5Writer'
        param = {'file': self.file,
                 'compression': self.comp}
        meta[label] = param
        return meta

    def put_frame(self, frm):
        """Save a single frame into a HDF5 group
        labeled with the frame number.
        """
        frm.put_meta(self.get_meta())
        if 'log' in frm.data.keys():
            json_str = json.dumps(frm.data['log'])
            del frm.data['log']
            if len(json_str) > 0:
                frm.data['log'] = json_str
        # ---
        h5pickle.save(self.h5fp, str(frm.i), frm.data, compression=self.comp)

    def dump(self):
        """Save a series of frames (== trajectory).
        The dump() method saves all the frames pending
        from the Writer's data source (self.src).
        If the user code sets up a data processing pipeline,
        the dump() routine drives it by providing the final sink.
        """
        for frm in self.src.next():
            if isinstance(frm, base.Container):
                if self.verb:
                    print "H5Writer.dump() : ", frm.i
                self.put_frame(frm)
            else:
                # None objects
                pass
