# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4 fileencoding=utf-8
#
# Cadishi --- CAlculation of DIStance HIstograms
#
# Copyright (c) Klaus Reuter, Juergen Koefinger
# See the file AUTHORS.rst for the full list of contributors.
#
# Released under the MIT License, see the file LICENSE.txt.

# We handle version information in this single file as is commonly done, see e.g.
# https://packaging.python.org/single_source_version/#single-sourcing-the-version

# The 'ver' tuple is read by the setup.py, conf.py of the Sphinx documentation
# system and __init__.py of the package.

ver = (1, 1, 2)


def get_version_string():
    """Return the full version number."""
    return '.'.join(map(str, ver))


def get_short_version_string():
    """Return the version number without the patchlevel."""
    return '.'.join(map(str, ver[:-1]))


def get_printable_version_string():
    version_string = " Cadishi " + get_version_string()
    try:
        from . import githash
    except:
        pass
    else:
        version_string += " (git: " + githash.human_readable + ")"
    return version_string
