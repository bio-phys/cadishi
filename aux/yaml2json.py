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

"""A simple YAML to JSON file converter.

Usage: yaml2json.py input_file [output_file]
"""
from __future__ import print_function


def main():
    import os
    import sys
    from cadishi import util

    if (len(sys.argv) == 2):
        if (sys.argv[1].endswith('yaml')):
            output_file = sys.argv[1].rstrip('yaml') + 'json'
        else:
            output_file = sys.argv[1] + '.json'
    elif (len(sys.argv) == 3):
        output_file = sys.argv[2]
    else:
        print("Usage: %s yaml_file [json_file]" % util.get_executable_name())
        sys.exit(1)

    data = util.load_yaml(sys.argv[1])
    util.save_json(data, output_file)


if __name__ == "__main__":
    main()
