#!/usr/bin/env python

from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

sourcefiles = [ 'SuchTree/SuchTree.pyx' ]

extensions = [ Extension( 'SuchTree', sourcefiles ) ]

extensions = cythonize( extensions, language_level = "3",
                        include_path = [ numpy.get_include() ] )

setup(
    ext_modules = cythonize( 'SuchTree/SuchTree.pyx' )
)
