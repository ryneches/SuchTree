#!/bin/env python

class SuchTreeError( Exception ) :
    '''Base exception class errors.'''
    pass

class NodeNotFoundError( SuchTreeError ) :
    '''Raised when a node ID or leaf name is not found in the tree.'''
    
    def __init__( self, node, message=None ):
        if message is None :
            if isinstance( node, str ) :
                message = 'Leaf name not found: {node}.'.format( node=str(node) )
            else :
                message = 'Node not found: {node}'.format( node=str(node) )
        super().__init__( message )
        self.node = node

class InvalidNodeError( SuchTreeError ) :
    '''Raised when a node ID is out of bounds or invalid.'''

    def __init__( self,
                  node_id,
                  tree_size=None,
                  message=None ) :
        if message is None :
            if tree_size is not None :
                message = 'Node ID {node_id} out of bounds (tree size: {tree_size})'.format( node_id=str(node_id),
                                                                                             tree_size=str(tree_size) )
            else :
                message = 'Invalid node ID: {node_id}'.format( node_id=str(node_id) )
        super().__init__( message )
        self.node_id = node_id
        self.tree_size = tree_size


class TreeStructureError( SuchTreeError ) :
    '''Raised when tree structure is invalid or inconsistent.'''
    pass

