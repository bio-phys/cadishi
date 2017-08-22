#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4 fileencoding=utf-8
#
# Cadishi --- CAlculation of DIStance HIstograms
#
# Copyright (c) Klaus Reuter, Juergen Koefinger
# See the file AUTHORS.rst for the full list of contributors.
#
# Released under the MIT License, see the file LICENSE.txt.

import os
import subprocess as sub

# check if this script is executed from the packages root directory
assert(os.path.isfile("./scripts/update_git_hash.py"))

package_name = "cadishi"

try:
    cmd = "git describe --all --long --dirty --tags".split()
    raw = sub.check_output(cmd).rstrip().split("/")[1]
except:
    raw = "not available"
with open("./" + package_name + "/githash.py", "w") as fp:
    fp.write("human_readable = \"" + raw + "\"\n")
