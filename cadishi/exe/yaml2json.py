#!/usr/bin/env python2.7
# vim:fileencoding=utf-8
"""A simple YAML to JSON file converter.

Usage: yaml2json.py input_file [output_file]
"""
# This file is part of the Cadishi package.  See README.rst,
# LICENSE.txt, and the documentation for details.


def main():
    import os
    import sys
    from .. import util

    if (len(sys.argv) == 2):
        if (sys.argv[1].endswith('yaml')):
            output_file = sys.argv[1].rstrip('yaml') + 'json'
        else:
            output_file = sys.argv[1] + '.json'
    elif (len(sys.argv) == 3):
        output_file = sys.argv[2]
    else:
        print "Usage: %s yaml_file [json_file]" % util.get_executable_name()
        sys.exit(1)

    data = util.load_yaml(sys.argv[1])
    util.save_json(data, output_file)


if __name__ == "__main__":
    main()
