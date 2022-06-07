#!/usr/bin/env python

from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

sourcefiles = [ 'SuchTree/SuchTree.pyx' ]

extensions = [ Extension( '_SuchTree', sourcefiles, include_dirs=[numpy.get_include()]) ]

extensions = cythonize( extensions, language_level = "3" )

setup(
    ext_modules = extensions
)
