Changelog
=========

Unreleased
----------

[1.1.0] - 2018-02-05
--------------------
Added
^^^^^
- Automated detection of the L2 cache size for the CPU kernels.
- Support for 96 kB of shared memory on Volta GPUs.
- Python 3 compatibility.
- MD reader from Capriqorn to make data import from MD codes easier.

Changed
^^^^^^^

[1.1.0b] - 2018-02-05
---------------------
Added
^^^^^
- Support for orthorhombic and triclinic periodic boxes.

Changed
^^^^^^^
- Greatly improved the performance of the CPU kernel (pydh) by
  introducing a cache blocking scheme adapted to the L2 cache
  for problems larger than 100k^2.
- Rewrote the interfaces of the CPU (pydh) and the GPU (cudh)
  kernels using Cython to enable Python 3 compatibility.

[1.0.0] - 2017-09-15
--------------------
Initial public release.
