# vim:fileencoding=utf-8
"""Cadishi base library.

Provides the basic data container and some more base classes
that are used throughout the code.
"""
# This file is part of the Cadishi package.  See README.rst,
# LICENSE.txt, and the documentation for details.


import copy

from . import util
from . import dictfs
from . import dict_util


# --- some string constants ---
# Locations (loc_*) used to store data either in memory using container/dictfs
# as well as on disk using HDF5 files.  (Comparable to paths of a file system.)
loc_parallel = 'parallel.tmp'
loc_coordinates = 'coordinates'
loc_dimensions = 'dimensions'  # 6-tuple periodic-box specifier
loc_volumes = 'volumes'
loc_len_histograms = 'len_histograms'
loc_nr_particles = 'particle_numbers'
loc_histograms = 'histograms'
loc_histogram_scale_factors = loc_histograms + '/' + 'scale_factors'
loc_histogram_mask = loc_histograms + '/' + 'mask'
loc_intensity = 'intensity'
loc_solv_match = 'solvent_matching'
loc_delta_h = 'delta_h'
loc_pddf = 'pddf'
loc_rdf = 'rdf'
# adding shell_Hxx for MultiReferenceStructure to hold average of properly scaled (volume-weighted)
# histograms
loc_shell_Hxx = 'shell_Hxx'
# Miscellaneous string constants.
id_radii = 'radii'


class Container:
    """Central container to hold/accumulate data while it is proparaged through
    the pipeline.  Heavily uses dictfs internally.
    """

    def __init__(self, number=-1, mkdir=[]):
        self.i = number  # frame number
        # any other data is to be stored in the data dictionary
        self.data = {}
        self.data['log'] = []  # pipeline log uses a list to preserve the order
        for path in mkdir:
            dictfs.save(self.data, path, {})

    def put_meta(self, meta):
        """Append pipeline log information to the instance's log list."""
        self.data['log'].append(copy.deepcopy(meta))

    def get_meta(self):
        """Return the instance's pipeline log list."""
        return self.data['log']

    def query_meta(self, path):
        """Obtain a value from the pipeline log list by using
        a Unix-path-like string identifier."""
        log = self.data['log']
        path = util.tokenize(path)
        assert (len(path) > 0)
        # inversely search the pipeline log list
        entry = util.search_pipeline(path[0], log)
        if (entry is not None):
            path.pop(0)
            try:
                if (len(path) > 0):
                    return dictfs.load(entry, path)
                else:
                    return entry
            except KeyError:
                return None
        else:
            return None

    def get_geometry(self, valid_geom=['Sphere', 'Cuboid', 'Ellipsoid',
                                       'ReferenceStructure', 'MultiReferenceStructure', 'Voxels']):
        """Search the pipeline log backwards for the geometry filter that was
        used, and return the result as a string."""
        # ---
        for geom in valid_geom:
            entry = util.search_pipeline(geom, self.get_meta())
            if (entry != None and len(entry) > 0):
                return geom
        return None

    def put_data(self, location, data):
        dictfs.save(self.data, location, data)

    def get_data(self, location):
        return dictfs.load(self.data, location)

    def del_data(self, location):
        dictfs.delete(self.data, location)

    def sum_data(self, other, location, skip_keys=['radii', 'frame']):
        """
        Add (+) data at location from other to self. If location does not exist
        in the current instance, it is created.
        """
        assert isinstance(other, Container)
        if not dictfs.exists(self.data, location):
            dictfs.save(self.data, location, {})
        X = dictfs.load(self.data, location)
        Y = dictfs.load(other.data, location)
        dict_util.sum_values(X, Y, skip_keys)

    def scale_data(self, C, location, skip_keys=['radii', 'frame']):
        """Scale (ie multiply) data at location by the factor C."""
        X = dictfs.load(self.data, location)
        dict_util.scale_values(X, C, skip_keys)

    def append_data(self, other, location, skip_keys=['radii']):
        """Append data at location from other to self. If location does not exist
        in the current instance, it is created.
        """
        assert isinstance(other, Container)
        if not dictfs.exists(self.data, location):
            dictfs.save(self.data, location, {})
        X = dictfs.load(self.data, location)
        Y = dictfs.load(other.data, location)
        dict_util.append_values(X, Y, skip_keys)

    def get_keys(self, location, skip_keys=None):
        """Get a list of the keys of the data stored at location.
        """
        keys = sorted((dictfs.load(self.data, location)).keys())
        if skip_keys is not None:
            if isinstance(skip_keys, list):
                _skip = skip_keys
            else:
                _skip = [skip_keys]
            for _key in _skip:
                if _key in keys:
                    keys.remove(_key)
        return keys

    def has_key(self, location):
        """Check if the current object instance has data stored at location.
        """
        return dictfs.exists(self.data, location)


class TrajectoryInformation:
    """Handle trajectory meta data."""

    def __init__(self):
        self.species = []
        self.frame_numbers = []
        self.pipeline_log = []

    def get_pipeline_parameter(self, _id):
        """Return the value of the _last_ occurrence of "id" in the pipeline, ie.
        the pipeline is searched in reversed order.
        """
        value = None
        for entry in reversed(self.pipeline_log):
            _label = ""
            parameters = {}
            for (_label, parameters) in entry.iteritems():
                break
            if _id in parameters:
                value = parameters[_id]
        return value


class PipelineElement(object):
    """Base class common to Filter, Reader, and Writer.  Provides methods needed
    to implement dependency checking between pipeline elements.
    Note: The "object" parameter makes it a new style class which is necessary to make
    the "super()" mechanism work to implement inheritance of the _depends and _conflicts
    lists.
    """
    _depends = []
    _conflicts = []

    def depends(self):
        return self._depends

    def conflicts(self):
        return self._conflicts


class Filter(PipelineElement):
    """Filter base class, to be overloaded by an actual implementation."""
    _depends = ["Reader"]

    def __init__(self, source=-1, verbose=False):
        self._depends.extend(super(Filter, self)._depends)
        self.src = source
        self.verb = verbose

    def set_input(self, source):
        self.src = source

    def get_meta(self):
        """Return information on the present filter, ready to be added to a frame
        object's list of pipeline meta information.
        """
        meta = {}
        label = 'Filter base class (this message should never appear)'
        param = {}
        meta[label] = param
        return meta


class Reader(PipelineElement):
    _conflicts = ['Reader']
    """Reader base class, to be overloaded by an actual implementation."""

    def get_meta(self):
        """Return information on the present filter, ready to be added to a frame
        object's list of pipeline meta information.
        """
        meta = {}
        label = 'Reader base class (this message should never appear)'
        param = {}
        meta[label] = param
        return meta


class Writer(PipelineElement):
    _depends = ['Reader']
    _conflicts = ['Writer']
    """Writer base class, to be overloaded by an actual implementation."""

    def set_input(self, source):
        self.src = source
