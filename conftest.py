import pytest
import sys
from setuptools import setup
from setuptools.command.build_ext import build_ext
from setuptools.dist import Distribution

def build_extensions_inplace():
    '''
    Programmatically run `build_ext --inplace`
    before pytest collects tests.
    '''
    dist = Distribution()
    dist.parse_config_files()
    cmd = build_ext(dist)
    cmd.inplace = 1
    cmd.ensure_finalized()
    cmd.run()

@pytest.hookimpl( tryfirst=True )
def pytest_sessionstart(session):
    print( '[pytest] building Cython extensions in place...' )
    build_extensions_inplace()
