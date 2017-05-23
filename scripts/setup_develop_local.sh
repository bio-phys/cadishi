#!/bin/bash


./scripts/update_git_hash.py

#export CAD_CUDA=0

# link all packages from ~/.local to the source location, for development
python setup.py clean config build develop --user

