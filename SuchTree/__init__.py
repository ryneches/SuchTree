#!/usr/bin/env python
from __future__ import absolute_import
from SuchTree.MuchTree import SuchTree, SuchLinkedTrees
from .__version__ import __version__

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

