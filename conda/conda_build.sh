#!/bin/bash

set -e

# trick to determine the absolute directory of the script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# where to put the output (the packaged file)
CONDA_BLD_OUTDIR=$DIR/output/
mkdir -p $CONDA_BLD_OUTDIR

# for some reason there must be no overlap between the package source path
# and the build path, so we put it into /tmp.
# Note that the directory is removed below using "rm -rf"!
export CONDA_BLD_PATH=/tmp/$USER/conda-bld
mkdir -p $CONDA_BLD_PATH

conda build --no-anaconda-upload --output-folder $CONDA_BLD_OUTDIR ./recipe/

rm -rf "$CONDA_BLD_PATH"

