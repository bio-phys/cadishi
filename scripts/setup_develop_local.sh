#!/bin/bash
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
./scripts/update_git_hash.py
# link all packages from ~/.local to the source location, for development
python setup.py clean config build develop --user
