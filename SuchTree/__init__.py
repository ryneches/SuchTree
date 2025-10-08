#!/usr/bin/env python
from __future__ import absolute_import
from SuchTree.MuchTree import SuchTree, SuchLinkedTrees
#from .__version__ import __version__

# Import version - will be available from __version__.py after build
try:
    from SuchTree.__version__ import __version__
except ImportError:
    # Fallback for development when __version__.py doesn't exist yet
    try:
        from importlib.metadata import version, PackageNotFoundError
        try:
            __version__ = version("SuchTree")
        except PackageNotFoundError:
            __version__ = "0.0.0+dev"
    except ImportError:
        __version__ = "0.0.0+dev"

# Import exceptions last to avoid circular dependencies
from .exceptions import (
    SuchTreeError,
    NodeNotFoundError,
    InvalidNodeError,
    TreeStructureError
)

__all__ = [
    'SuchTree',
    'SuchLinkedTrees',
    'SuchTreeError',
    'NodeNotFoundError',
    'InvalidNodeError',
    'TreeStructureError',
    '__version__'
]

