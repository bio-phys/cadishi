=======
CADISHI
=======


Introduction
------------

CADISHI -- CAlculation of DIStance HIstograms -- is a software package that
enables scientists to compute (Euclidean) distance histograms efficiently. Any
sets of objects that have 3D Cartesian coordinates may be used as input, for
example, atoms in molecular dynamics datasets or galaxies in astrophysical
contexts. CADISHI drives the high-performance kernels pydh (CPU) and cudh (GPU,
optional) to do the actual histogram computation. The kernels pydh and cudh are
part of CADISHI and are written in C++ and CUDA.

For more information, we refer to our publication:

K. Reuter, J. Koefinger; CADISHI: Fast parallel calculation of particle-pair
distance histograms on CPUs and GPUs; Computer Physics Communications (2018);
<https://doi.org/10.1016/j.cpc.2018.10.018>.

A preprint of the paper is available at <https://arxiv.org/abs/1808.01478>.



Installation
------------

The CADISHI package is installed in the canonical way e.g. as follows::

   python setup.py install --user

The `setup_install_local.sh` script may be used to perform the local
installation.  Make sure to add `$HOME/.local/bin` to your PATH environment
variable.

CADISHI was developed, built, and tested on SUSE Linux Enterprise Server 11 and
12, Ubuntu Linux 14.04 LTS and 16.04 LTS, and Scientific Linux 7 using the
Anaconda Python distribution version 2, release 4.0.0, and newer. CADISHI
requires gcc, and optionally nvcc, to compile C++ code during the installation.
CADISHI was successfully tested Mac computers as well.


Quick start guide
-----------------

CADISHI provides a single executable `cadishi` that gives access to the distance
histogram calculations.  Run `cadishi --help` to get an overview on the
available commands and options.

To run an example calculation based on the data set included in CADISHI proceed
as follows::

1. Run `cadishi example` to generate an example input file `histograms.yaml`.
2. Optional: Adapt the file `histograms.yaml`.
3. Run `cadishi histo` to run the distance histogram calculation.

Note that the input data needs to be prepared in HDF5 format for performance
reasons. See the included example dataset for details. The histograms are written
to an HDF5 file as well.  CADISHI uses multiple processes to be able to utilize
all the compute resources (CPU cores, GPUs) available on a node simultaneously.


Documentation
-------------

Documentation is available at `http://cadishi.readthedocs.io/en/latest/
<http://cadishi.readthedocs.io/en/latest/>`_.
Alternatively, you may access the local copy at `doc/html/index.html` after having
cloned the repository.


Directories and Files
---------------------

The CADISHI software is designed and packaged as a Python package.  Each Python
source file contains docstrings, explaining each module and function.
Nevertheless, an explanation of the directory structure and of important files
therein is given in the following.

`cadishi` directory, and subdirectories
  CADISHI Python module, implementing various functions necessary to run the
  CADISHI engine.

`cadishi/exe` directory
  CADISHI executables, to be invoked via the `cadishi` command.  Most
  importantly, `histograms.py` is the main program of the CADISHI engine.

`cadishi/kernel` directory, and subdirectories
  C++/CUDA high-performance implementations of the distance histogram
  computation, and interfaces to Python.  Noteworthy files are:

`cadishi/kernel/c_pydh.pyx`
  Cython interface to the CPU distance histogram kernel (pydh).

`cadishi/kernel/c_pydh_functions.cc`
  C++ CPU distance histogram kernel implementation.

`cadishi/kernel/c_cudh.pyx`
  Cython interface to the GPU distance histogram kernel (cudh).

`cadishi/kernel/c_cudh_functions.cu`
  CUDA GPU distance histogram kernel.

`doc` directory
  Documentation in rst format, to be processed using the Sphinx documentation
  system.

`aux` directory
  Auxiliary scripts for CADISHI, e.g. bash completion and data import.

`scripts` directory
  Example installation scripts for CADISHI.


License and Citation
--------------------

The CADISHI package is released under the permissive MIT license.  See the file
`LICENSE.txt` for details.

Copyright 2015-2018  Klaus Reuter (MPCDF), Juergen Koefinger (MPIBP)

In case you're using CADISHI for own academic or non-academic research, we
kindly request that you cite CADISHI in your publications and presentations. We
suggest the following citation as appropriate:

K. Reuter, J. Koefinger; CADISHI: Fast parallel calculation of particle-pair
distance histograms on CPUs and GPUs; Computer Physics Communications (2018);
<https://doi.org/10.1016/j.cpc.2018.10.018>.

