#!/usr/bin/env python

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
