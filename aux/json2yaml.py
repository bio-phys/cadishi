#!/usr/bin/env python2.7
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4 fileencoding=utf-8
#
# Cadishi --- CAlculation of DIStance HIstograms
#
# Copyright (c) Klaus Reuter, Juergen Koefinger
# See the file AUTHORS.rst for the full list of contributors.
#
# Released under the MIT License, see the file LICENSE.txt.

"""A simple JSON to YAML file converter.

Usage: json2yaml.py input_file [output_file]
"""
from __future__ import print_function


def main():
    import os
    import sys
    from cadishi import util

    if (len(sys.argv) == 2):
        if (sys.argv[1].endswith('json')):
            output_file = sys.argv[1].rstrip('json') + 'yaml'
        else:
            output_file = sys.argv[1] + '.yaml'
    elif (len(sys.argv) == 3):
        output_file = sys.argv[2]
    else:
        print("Usage: %s json_file [yaml_file]" % util.get_executable_name())
        sys.exit(1)

    data = util.load_json(sys.argv[1])
    util.save_yaml(data, output_file)


if __name__ == "__main__":
    main()
