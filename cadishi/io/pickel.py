# vim:fileencoding=utf-8
"""Cadishi IO library using pickle. The name was chosen
deliberately 'pickel' to avoid name conflicts.

Pickle data reader/writer for base.Container instances.
"""
# This file is part of the Cadishi package.  See README.rst,
# LICENSE.txt, and the documentation for details.


import pickle
from six.moves import range

from .. import base
from .. import util


class PickleReader(base.Reader):
    """Pickle reader for base.Container instances."""
    _depends = []
    _conflicts = []

    def __init__(self, file="default_", first=None, last=None, step=1,
                 verbose=False):
        self.file = file
        self.first = first
        self.last = last
        self.step = step
        self.verb = verbose
        # ---
        self._depends.extend(super(base.Reader, self)._depends)
        self._conflicts.extend(super(base.Reader, self)._conflicts)

    def get_meta(self):
        """Return information on the pickle reader,
        ready to be added to a frame object's list of
        pipeline meta information.
        """
        meta = {}
        label = 'PickleReader'
        param = {'file': self.file, 'first': self.first,
                 'last': self.last, 'step': self.step}
        meta[label] = param
        return meta

    def get_frame(self, number):
        """Read a frame identified by its number and
        return a container object."""
        frm = base.Container()
        frm.i = int(number)
        # ---
        file = self.file + str(frm.i) + '.p'
        with open(file, 'rb') as fp:
            frm.data = pickle.load(fp)
        # ---
        return frm

    def next(self):
        """Iterate through all the frames and yield frame by frame."""
        for idx in range(self.first, self.last, self.step):
            frm = self.get_frame(idx)
            frm.put_meta(self.get_meta())
            if self.verb:
                print "PickleReader.next() : ", frm.i
            yield frm


class PickleWriter(base.Writer):
    """Pickle writer for base.Container instances."""
    _depends = []
    _conflicts = []

    def __init__(self, file="default_", source=-1, verbose=False):
        self.file = file
        util.md(file)
        self.src = source
        self.verb = verbose
        self.info = ''
        # ---
        self._depends.extend(super(base.Writer, self)._depends)
        self._conflicts.extend(super(base.Writer, self)._conflicts)

    def get_meta(self):
        """Return information on the pickle writer,
        ready to be added to a frame object's list of
        pipeline meta information.
        """
        meta = {}
        label = 'PickleWriter'
        param = {'file': self.file}
        meta[label] = param
        return meta

    def put_frame(self, frm):
        """Save a single frame into a pickle file
        labeled with the frame number."""
        file = self.file + str(frm.i) + '.p'
        pickle.dump(frm.data, open(file, 'wb'))

    def dump(self):
        """Save a series of base.Container instances pending
        from the writer's data source to individual pickle
        files.
        """
        for frm in self.src.next():
            if self.verb:
                print "PickleWriter.dump() : ", frm.i
            frm.put_meta(self.get_meta())
            self.put_frame(frm)
