#!/usr/bin/env python

from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy
from pathlib import Path

this_directory = Path(__file__).parent
long_description = ( this_directory / 'README.md' ).read_text()

sourcefiles = [ 'SuchTree/*.pyx' ]
extensions = [ Extension( 'MuchTree', sourcefiles, include_dirs=[numpy.get_include()]) ]
extensions = cythonize( extensions, compiler_directives={ 'language_level' : '3' } )

setup(
    ext_modules = extensions,
    py_modules = [ 'SuchTree' ],
    long_description=long_description,
    long_description_content_type='text/markdown'
)
