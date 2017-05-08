========
Cadishi
========


Introduction
------------

Cadishi -- CAlculation of DIStance HIstograms -- is a software package written
largely in Python that enables scientists to compute (Euclidean) distance
histograms efficiently. Any sets of objects that have 3D Cartesian coordinates
may be used as input, for example atoms in molecular dynamics datasets or
galaxies in astrophysical contexts. Cadishi drives the kernels pydh (CPU) and
cudh (GPU) to do the actual histogram computation. These high-performance
kernels are written in C++ and CUDA.


Installation
------------

The package is installed in the Pythonic way e.g. as follows::

   python setup.py install --user

The `setup_install_local.sh` script in the repository root may be used to
perform the local installation.  Make sure to add `$HOME/.local/bin` to your
PATH environment variable.

Cadishi was developed, built, and tested on SUSE Linux Enterprise Server 11 SP
4, Ubuntu Linux 14.04 LTS, and Scientific Linux 7 using cPython 2.7.11 from the
Anaconda Python distribution.


Quick start guide
-----------------

Cadishi provides a single executable `cadishi` that gives access to the distance
histogram calculations.  Run `cadishi --help` to get an overview on the
available commands and options.

To run an example calculation based on the data set included in Cadishi proceed
as follows::

1. Run `cadishi histo-example` to generate an example input file `histograms.yaml`.
2. Optional: Adapt the file `histograms.yaml`.
3. Run `cadishi histo` to run the distance histogram calculation.


License and Citation
--------------------

Cadishi is released under the permissive MIT license.  See the file
`LICENSE.txt` for details.

Copyright 2015-2017  Klaus Reuter (MPCDF), Juergen Koefinger (MPIBP)

In case you're using Cadishi for your own academic or non-academic research, we
kindly request that you cite Cadishi in your publications and presentations. We
suggest the following citations as appropriate:

TODO: Add paper reference once it is on the arxiv and/or published. 
