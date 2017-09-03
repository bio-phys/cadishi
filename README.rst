=======
Cadishi
=======


Introduction
------------

Cadishi -- CAlculation of DIStance HIstograms -- is a software package that
enables scientists to compute (Euclidean) distance histograms efficiently. Any
sets of objects that have 3D Cartesian coordinates may be used as input, for
example, atoms in molecular dynamics datasets or galaxies in astrophysical
contexts. Cadishi drives the high-performance kernels pydh (CPU) and cudh (GPU,
optional) to do the actual histogram computation. The kernels pydh and cudh are
part of Cadishi and are written in C++ and CUDA.


Installation
------------

The Cadishi package is installed in the canonical way e.g. as follows::

   python setup.py install --user

The `setup_install_local.sh` script may be used to perform the local
installation.  Make sure to add `$HOME/.local/bin` to your PATH environment
variable.

Cadishi was developed, built, and tested on SUSE Linux Enterprise Server 11 and
12, Ubuntu Linux 14.04 LTS and 16.04 LTS, and Scientific Linux 7 using the
Anaconda Python distribution version 2, release 4.0.0, and newer. Cadishi
requires gcc, and optionally nvcc, to compile C++ code during the installation.


Quick start guide
-----------------

Cadishi provides a single executable `cadishi` that gives access to the distance
histogram calculations.  Run `cadishi --help` to get an overview on the
available commands and options.

To run an example calculation based on the data set included in Cadishi proceed
as follows::

1. Run `cadishi example` to generate an example input file `histograms.yaml`.
2. Optional: Adapt the file `histograms.yaml`.
3. Run `cadishi histo` to run the distance histogram calculation.

Note that the input data needs to be prepared in HDF5 format for performance
reasons. See the included example dataset for details. The histograms are written
to an HDF5 file as well.  Cadishi uses multiple processes to be able to utilize
all the compute resources (CPU cores, GPUs) available on a node simultaneously.


Documentation
-------------

Please visit `doc/html/index.html`.


License and Citation
--------------------

The Cadishi package is released under the permissive MIT license.  See the file
`LICENSE.txt` for details.

Copyright 2015-2017  Klaus Reuter (MPCDF), Juergen Koefinger (MPIBP)

In case you're using Cadishi for own academic or non-academic research, we
kindly request that you cite Cadishi in your publications and presentations. We
suggest the following citations as appropriate:

TODO: Add proper paper reference once it is publicly available. Please use the
following reference meanwhile:
"K. Reuter, J. Koefinger: Cadishi, https://github.com/bio-phys/cadishi, 2017."
