#!/usr/bin/env python2.7
# vim:fileencoding=utf-8
"""A simple JSON to YAML file converter.

Usage: json2yaml.py input_file [output_file]
"""
# This file is part of the Cadishi package.  See README.rst,
# LICENSE.txt, and the documentation for details.


def main():
    import os
    import sys
    from .. import util

    if (len(sys.argv) == 2):
        if (sys.argv[1].endswith('json')):
            output_file = sys.argv[1].rstrip('json') + 'yaml'
        else:
            output_file = sys.argv[1] + '.yaml'
    elif (len(sys.argv) == 3):
        output_file = sys.argv[2]
    else:
        print "Usage: %s json_file [yaml_file]" % util.get_executable_name()
        sys.exit(1)

    data = util.load_json(sys.argv[1])
    util.save_yaml(data, output_file)


if __name__ == "__main__":
    main()
