Cadishi documentation
=====================

Introduction
------------

The Cadishi package is designed to perform fast computations of histograms of
the (Euclidean) distances between objects. For example, these objects may be
atoms from molecular dynamics (MD) simulation datasets, or stars or galaxies
from astrophysical datasets. While Cadishi is written mostly in Python, it is
built upon high-performance kernels written in C++ and CUDA to exploit the CPU
and GPU resources of a compute node as best as possible.

In the field of MD simulations, Cadishi is most useful in conjunction with the
Capriqorn package. The name Cadishi is simply an acronym derived from (or, more
precisely, the syllabic abbreviation of) *calculation of distance histograms*.


Requirements
------------

Cadishi requires a Python 2.7 installation including the NumPy, SciPy, Cython,
h5py, and PyYAML modules. We recommend to use the Anaconda Python Distribution
which provides all the necessary modules out of the box. Cadishi was mostly
developed using Anaconda Python 2, versions 4.0.0 and newer. Moreover, to
compile the high-performance kernels, recent GCC and CUDA (optional)
installations are required. GCC 4.9 and GCC 5.4, and CUDA 7.5 and 8.0 were used
successfully.


Features
--------

Cadishi features a two-level parallelization. On the top level, Cadishi
parallelizes over frames (i.e. snapshots of the ensemble of objects at certain
points in time) using the Python multiprocessing module. A single frame is
processed by a single worker process running a CPU or GPU kernel. Multiple
worker processes may run simultaneously on a shared-memory multi-core machine.
On the frame level, OpenMP threads are used by the CPU kernel, and CUDA threads
are used by the GPU kernel. Hence, it is possible to fully exploit the
resources of a shared-memory machine. E.g., on a dual-socket server with two
GPUs, Cadishi would use two CPU worker processes (one per multi-core chip) and
two GPU worker processes (one per GPU card).

Optionally, Cadishi supports orthorhombic and triclinic periodic boxes and
internally applies the minimum image convention to the distances. Computations
can be performed in single (default) or double precision. Optionally, the
distances can be checked if they fit into the desired histogram width. Given
the combinatorial space resulting from these possibilities, template C++ code
is used to generate machine code with a minimum amount of branches at runtime.
Recent compilers are known to generate well-vectorized machine code from the
distance calculation for the CPU. The GPU kernel benefits strongly from the
fast shared-memory atomic operations introduced with the MAXWELL generation of
NVIDIA GPUs.



Installation
------------

The package comes with a standard Python setup.py file.  It is installed e.g. as
follows into the user's homedirectory::

   python setup.py install --user

In this case, setup.py copies the Cadishi files into the directory ``~/.local``.
Make sure that your ``PATH`` environment variable includes the directory
``~/.local/bin``.  To enable a CUDA build the ``nvcc`` compiler wrapper must be
found via the ``PATH`` environment variable.  Influential boolean environment variables
are ``CAD_DEBUG``, ``CAD_OPENMP``, ``CAD_CUDA``, and ``CAD_SAFE_CUDA_FLAGS``; if
set they override the keys listed in the options section of the file ``setup.cfg``.


Usage
-----

The Cadishi package provides a single main executable ``cadishi`` which supports
commands and options (similar to the well-known concept used by ``git``).

To get a quick overview on the options of Cadishi issue the following command::

   cadishi --help

To get started a parameter file controlling the distance histogram calculation
is required. A basic example is included with the package and can be created as
follows::

   cadishi example [--expert]

By default, the parameter file's name is ``histograms.yaml``. Edit the parameter
file to your needs.  In particular, the compute kernels *pydh* (CPU) and *cudh*
(GPU) can be configured.  In the example parameter file, the input configuration
points to the default test case included with the Cadishi package. Adapt the
input configuration to your actual data. As a second step the distance histogram
calculation is run as follows::

   cadishi histo


Source documentation
--------------------

.. toctree::
   :maxdepth: 2

   modules.rst

   kernel.rst

   executables.rst


License and Citation
--------------------

Cadishi is released under the permissive MIT license.  See the file
`LICENSE.txt` for details.

Copyright 2015-2017  Klaus Reuter (MPCDF), Juergen Koefinger (MPIBP)

In case you're using Cadishi for your own academic or non-academic research, we
kindly request that you cite Cadishi in your publications and presentations. We
suggest the following citations as appropriate:

TODO: Add paper reference once it is on the arxiv and/or published.


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
