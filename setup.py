# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4 fileencoding=utf-8
#
# Cadishi --- CAlculation of DIStance HIstograms
#
# Copyright (c) Klaus Reuter, Juergen Koefinger
# See the file AUTHORS.rst for the full list of contributors.
#
# Released under the MIT License, see the file LICENSE.txt.

"""Cadishi setup.py builder and installer.
"""

from __future__ import print_function
import os
import sys
import glob
import platform

try:
    import ConfigParser as configparser
except:
    import configparser

import subprocess as sub
from setuptools import setup, Command, Extension

try:
    import numpy
except ImportError:
    print("Need numpy for installation")
    sys.exit(1)

try:
    from Cython.Distutils import build_ext
except ImportError:
    print("Need cython for installation")
    sys.exit(1)

# Obtain the numpy include directory.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()


class Config(object):
    """Config wrapper class to get build options

    This class looks for options in the environment variables and the
    'setup.cfg' file. The order how we look for an option is.

    1. Environment Variable
    2. set in 'setup.cfg'
    3. given default

    Environment variables should start with 'CAD_' and be all uppercase.
    Values passed to environment variables are checked (case-insensitively)
    for specific strings with boolean meaning: 'True' or '1' will cause `True`
    to be returned. '0' or 'False' cause `False` to be returned.

    """

    def __init__(self, fname='setup.cfg'):
        if os.path.exists(fname):
            self.config = configparser.SafeConfigParser()
            self.config.read(fname)

    def get(self, option_name, default=None):
        environ_name = 'CAD_' + option_name.upper()
        if environ_name in os.environ:
            val = os.environ[environ_name]
            if val.upper() in ('1', 'TRUE'):
                return True
            elif val.upper() in ('0', 'FALSE'):
                return False
            else:
                raise RuntimeError("Unkown option type for environment var: {}={}".format(environ_name, val))
        try:
            option = self.config.getboolean('options', option_name)
            return option
        except configparser.NoOptionError:
            return default


def get_version_string():
    ver = {}
    with open("./cadishi/version.py") as fp:
        exec(fp.read(), ver)
    return ver['get_version_string']()


#########################
# Handle build options  #
#########################
config = Config()
CAD_DEBUG = config.get('debug', default=False)
CAD_OPENMP = config.get('openmp', default=True)
CAD_GCC_NATIVE = config.get('gcc_native', default=False)
CAD_CUDA = config.get('cuda', default=True)
CAD_SAFE_CUDA_FLAGS = config.get("safe_cuda_flags", default=False)

print("### Cadishi " + get_version_string() + " setup configuration")
print(" debug           : " + str(CAD_DEBUG))
print(" openmp          : " + str(CAD_OPENMP))
print(" gcc_native      : " + str(CAD_GCC_NATIVE))
print(" cuda            : " + str(CAD_CUDA))
print(" safe_cuda_flags : " + str(CAD_SAFE_CUDA_FLAGS))
print("###")


####################
# Common functions #
####################
def get_gcc_flags():
    # set up compiler flags for the C extensions
    cc_flags = ['-g', '-D_GLIBCXX_USE_CXX11_ABI=0']
    if CAD_DEBUG:
        cc_flags += ['-O0']
    else:
        cc_flags += ['-O3']
        if (find_in_path(['g++']) is not None):
            cc_flags += ['-ffast-math']  # essential to get vectorization and performance
            cc_flags += ['-funroll-loops']
            cc_flags += ['-mtune=native']  # optimize for the current CPU but preserve portability
            if platform.processor() == 'x86_64':
                if CAD_GCC_NATIVE:
                    # flag does not work e.g. on IBM Minsky systems
                    cc_flags += ['-march=native']
                else:
                    cc_flags += ['-msse4.2']  # required for fast round() instruction
            if not on_mac():
                if CAD_OPENMP:
                    cc_flags += ['-fopenmp']
                # cc_flags += ['-fopt-info']
    return cc_flags


def on_mac():
    """Check if we're running on a Mac."""
    if "Darwin" in platform.system():
        return True
    else:
        return False


def find_in_path(filenames):
    """Find file on system path."""
    # http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/52224
    from os.path import exists, join, abspath
    from os import pathsep, environ
    search_path = environ["PATH"]
    paths = search_path.split(pathsep)
    for path in paths:
        for filename in filenames:
            if exists(join(path, filename)):
                return abspath(join(path, filename))


#######################
# CUDA configuration  #
#######################
def get_cuda_ver(nvcc="nvcc"):
    cmd = [nvcc, '--version']
    major = -1
    minor = -1
    patch = -1
    raw = sub.check_output(cmd).split('\n')
    for line in raw:
        if line.startswith('Cuda'):
            tokens = line.split(',')
            # we obtain a version string such as "7.5.17"
            verstr = tokens[2].strip().strip('V')
            vertup = verstr.split('.')
            major = int(vertup[0])
            minor = int(vertup[1])
            patch = int(vertup[2])
    ver = major, minor, patch
    return ver


def locate_cuda():
    """Locate the CUDA environment on the system. Returns a dict with keys 'home',
    'nvcc', 'include', and 'lib' and values giving the absolute path to each
    directory. Starts by looking for the CUDAHOME env variable. If not found,
    everything is based on finding 'nvcc' in the PATH.
    """
    # adapted from
    # https://stackoverflow.com/questions/10034325/can-python-distutils-compile-cuda-code
    nvcc = None
    envs = ['CUDA_HOME', 'CUDA_ROOT', 'CUDAHOME', 'CUDAROOT']
    for env in envs:
        if env in os.environ:
            nvcc = os.path.join(os.environ[env], 'bin', 'nvcc')
            break
    else:
        # otherwise, search PATH for NVCC
        nvcc = find_in_path(['nvcc'])
    if nvcc is None:
        raise EnvironmentError(
            'The nvcc executable could not be found.  ' +
            'Add it to $PATH or set one of the environment variables ' +
            ', '.join(envs))
    home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {}
    cudaconfig['home'] = home
    cudaconfig['nvcc'] = nvcc
    cudaconfig['include'] = os.path.join(home, 'include')
    # on Linux, CUDA has the libraries in lib64
    lib_dir = os.path.join(home, 'lib64')
    if not os.path.isdir(lib_dir):
        # on the MAC they are in lib
        lib_dir = os.path.join(home, 'lib')
    cudaconfig['lib'] = lib_dir

    for k, v in cudaconfig.iteritems():
        if not os.path.exists(v):
            raise EnvironmentError(
                'The CUDA %s path could not be located in %s' % (k, v))
    # print "CUDA installation detected: " + home
    return cudaconfig


def cuda_compiler_flags():
    gcc_flags = get_gcc_flags()
    gcc_flags += ['-DCUDA_DEBUG']
    gcc_flags_string = " ".join(gcc_flags)
    nvcc_flags = ['-DCUDA_DEBUG']  # hardly adds overhead, recommended
    if CAD_DEBUG:
        nvcc_flags += ['-O0', '-g', '-G']
    else:
        if CAD_SAFE_CUDA_FLAGS:
            nvcc_flags += ['-O2']
            nvcc_flags += ['-use_fast_math']
            nvcc_flags += ['--generate-code', 'arch=compute_35,code=compute_35']
        else:
            nvcc_flags += ['-O3']
            nvcc_flags += ['-use_fast_math']
            nvcc_flags += ['--generate-code', 'arch=compute_35,code=sm_35']
            nvcc_flags += ['--generate-code', 'arch=compute_37,code=sm_37']
            if (CUDAVER[0] >= 6):
                nvcc_flags += ['--generate-code', 'arch=compute_50,code=sm_50']
            if (CUDAVER[0] >= 7):
                nvcc_flags += ['--generate-code', 'arch=compute_52,code=sm_52']
                nvcc_flags += ['--generate-code', 'arch=compute_53,code=sm_53']
            if (CUDAVER[0] >= 8):
                nvcc_flags += ['--generate-code', 'arch=compute_60,code=sm_60']
                nvcc_flags += ['--generate-code', 'arch=compute_61,code=sm_61']
    nvcc_flags += ['--compiler-options=' + gcc_flags_string + ' -fPIC']
    return {'gcc': gcc_flags, 'nvcc': nvcc_flags}


if CAD_CUDA:
    try:
        CUDA = locate_cuda()
        CUDAVER = get_cuda_ver(CUDA['nvcc'])
    except:
        CUDA = None
        print("CUDA was _not_ detected")
else:
    CUDA = None

############################
# Setuptools modifications #
############################


class cuda_build_ext(build_ext):
    @staticmethod
    def customize_compiler_for_nvcc(compiler):
        """Inject deeply into distutils to customize how the dispatch to gcc/nvcc
        works.

        """
        # adapted from
        # https://stackoverflow.com/questions/10034325/can-python-distutils-compile-cuda-code
        # --- tell the compiler it can processes .cu
        compiler.src_extensions.append('.cu')
        # --- save references to the default compiler_so and _comple methods
        default_compiler_so = compiler.compiler_so
        super = compiler._compile

        # --- now redefine the _compile method. This gets executed for each
        # object but distutils doesn't have the ability to change compilers
        # based on source extension: we add it.
        def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
            if os.path.splitext(src)[1] == '.cu':
                # use cuda for .cu files
                compiler.set_executable('compiler_so', CUDA['nvcc'])
                # use only a subset of the extra_postargs, which are 1-1
                # translated from the extra_compile_args in the Extension class
                postargs = extra_postargs['nvcc']
            else:
                if isinstance(extra_postargs, dict):
                    postargs = extra_postargs['gcc']
                else:
                    postargs = extra_postargs
            super(obj, src, ext, cc_args, postargs, pp_opts)
            # reset the default compiler_so, which we might have changed for
            # cuda
            compiler.compiler_so = default_compiler_so

        # inject our redefined _compile method into the class
        compiler._compile = _compile
        return compiler

    def build_extensions(self):
        self.compiler = cuda_build_ext.customize_compiler_for_nvcc(
            self.compiler)
        build_ext.build_extensions(self)


class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    # https://stackoverflow.com/questions/3779915/why-does-python-setup-py-sdist-create-unwanted-project-egg-info-in-project-r
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        os.system('rm -vrf ./*.so')
        os.system('rm -vrf build')
        os.system('rm -vrf ./doc/_build')
        os.system('rm -vrf dist')
        os.system('rm -vrf cadishi.egg-info')
        os.system("find cadishi -name '*.pyc' -delete -print")
        os.system("find cadishi -name '*.so' -delete -print")


#########################
# Cadishi Configuration #
#########################
def extensions():
    cc_flags = get_gcc_flags()

    exts = []
    exts.append(
        Extension(
            'cadishi.kernel.c_dist',
            sources=['cadishi/kernel/c_dist.pyx'],
            include_dirs=[numpy_include],
            extra_compile_args=cc_flags,
            extra_link_args=cc_flags))

    exts.append(
        Extension(
            'cadishi.kernel.c_pydh',
            sources=['cadishi/kernel/c_pydh.pyx',
                     'cadishi/kernel/c_pydh_functions.cc'],
            language="c++",
            include_dirs=[numpy_include, 'cadishi/kernel/include'],
            extra_compile_args=cc_flags,
            extra_link_args=cc_flags))

    if CUDA is None:
        print("Skipping cudh build")
    else:
        link_libraries=['cudart', 'stdc++']
        if CAD_OPENMP:
            link_libraries.append('gomp')
        exts.append(
            Extension(
                'cadishi.kernel.c_cudh',
                sources=['cadishi/kernel/c_cudh.pyx',
                         'cadishi/kernel/c_cudh_functions.cu'],
                language="c++",
                include_dirs=[numpy_include, 'cadishi/kernel/include'],
                libraries=link_libraries,
                library_dirs=[CUDA['lib']],
                runtime_library_dirs=[CUDA['lib']],
                extra_compile_args=cuda_compiler_flags()))

    return exts


data_files = [('cadishi/tests/data', glob.glob('cadishi/tests/data/*')),
              ('cadishi/data', glob.glob('cadishi/data/*'))]

entry_points = {
    'console_scripts': [
        'cadishi=cadishi.exe.cli:main',
        'json2yaml=cadishi.exe.json2yaml:main',
        'yaml2json=cadishi.exe.yaml2json:main'
    ]
}

# update_git_hash("cadishi")

setup(
    name="cadishi",
    version=get_version_string(),
    description='distance histogram calculation framework',
    author='Juergen Koefinger, Max Linke, Klaus Reuter',
    author_email='khr@mpcdf.mpg.de',
    packages=['cadishi',
              'cadishi.io',
              'cadishi.kernel',
              'cadishi.tests',
              'cadishi.exe'],
    install_requires=[
        'six',
        'future',
        'numpy',
        'scipy',
        'h5py',
        'pyyaml'
    ],
    cmdclass={'clean': CleanCommand,
              'build_ext': cuda_build_ext},
    entry_points=entry_points,
    data_files=data_files,
    ext_modules=extensions(),
    scripts=['cadishi.bash'],
    zip_safe=False)
