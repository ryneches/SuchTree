#!/usr/bin/env python
from __future__ import print_function

from setuptools import setup
from setuptools.command.build_ext import build_ext as _build_ext
import sys
from sys import version_info

import numpy

try:
    import Cython
    from Cython.Distutils import Extension
    HAS_CYTHON = True
    cy_ext = 'pyx'

except ImportError:
    from setuptools import Extension
    HAS_CYTHON = False
    cy_ext = 'c'


CY_OPTS = {
    'embedsignature': True,
    'language_level': version_info.major,
    'c_string_type': 'unicode',
    'c_string_encoding': 'utf8'
}


EXT_DICT = {
    'sources': ['SuchTree/SuchTree.{0}'.format(cy_ext)],
    'include_dirs': [numpy.get_include()]
}
if HAS_CYTHON:
    EXT_DICT['cython_directives'] = CY_OPTS

MODULE = Extension('SuchTree.SuchTree',
                   **EXT_DICT)


class VerboseBuildExt(_build_ext):

    def run(self):
        if HAS_CYTHON:
            print('*** NOTE: Found Cython, extension files will be '
                  'transpiled if this is an install invocation.',
                  file=sys.stderr)
        else:
            print('*** WARNING: Cython not found, assuming cythonized '
                  'C files available for compilation.',
                  file=sys.stderr)
        
        _build_ext.run(self)


setup(
    name='SuchTree',
    version='0.5',
    description='A Python library for doing fast, thread-safe computations on phylogenetic trees.',
    url='http://github.com/ryneches/SuchTree',
    author='Russell Neches',
    author_email='ryneches@ucdavis.edu',
    license='BSD',
    packages=['SuchTree'],
    download_url='https://github.com/ryneches/SuchTree/archive/0.4.tar.gz',
    install_requires=[
        'scipy>=0.18',
        'numpy',
        'dendropy',
        'cython',
        'pandas'
    ],
    zip_safe=False,
    ext_modules = [MODULE],
    setup_requires = [ 
        'pytest-runner',
        'Cython',
        'numpy'
    ],
    tests_require = [ 'pytest', ],
    cmdclass = {
        'build_ext': VerboseBuildExt
    }
)
