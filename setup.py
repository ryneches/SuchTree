#!/usr/bin/env python

from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy
from pathlib import Path

this_directory = Path(__file__).parent
long_description = ( this_directory / 'README.md' ).read_text()

sourcefiles = [ 'SuchTree/*.pyx' ]
extensions = [ Extension( 'SuchTree.MuchTree', sourcefiles, include_dirs=[numpy.get_include()]) ]
extensions = cythonize( extensions, compiler_directives={ 'language_level' : '3' } )

setup(
    ext_modules = extensions,
    py_modules = [ 'SuchTree' ],
    long_description=long_description,
    long_description_content_type='text/markdown',
    use_scm_version = { 'write_to' : 'SuchTree/__version__.py',
                        'local_scheme' : 'no-local-version' },
    setup_requires = [ 'setuptools_scm' ]
)
