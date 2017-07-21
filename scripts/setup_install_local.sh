#!/bin/bash
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
./scripts/update_git_hash.py
python setup.py clean config build install --user
