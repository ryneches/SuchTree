#!/usr/bin/env python

import setuptools_scm

version = setuptools_scm.get_version( local_scheme='no-local-version' )
with open( 'SuchTree/__version__.py', 'w' ) as f :
    f.write( f'__version__ = "{version}"\n' )

print( f"Wrote __version__.py : __version__ = {version}" )
