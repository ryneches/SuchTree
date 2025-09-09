import cython
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from dendropy import Tree
from random import sample
from itertools import combinations
from collections import deque
from pathlib import Path
import numpy as np
cimport numpy as np
import pandas as pd
from scipy.linalg.cython_lapack cimport dsyev
from numbers import Integral, Real
from typing import Any, Union, Dict, Tuple, Generator, Optional

from warnings import warn
from exceptions import SuchTreeError, NodeNotFoundError, InvalidNodeError, TreeStructureError

# if igraph is available, enable
# SuchLinkedTrees.to_igraph()
try :
    from igraph import Graph, ADJ_UNDIRECTED
    with_igraph = True
except ImportError :
    with_igraph = False

cdef extern from 'stdint.h' :
    ctypedef unsigned long int uint64_t
    uint64_t UINT64_MAX

# Trees are built from arrays of Node structs. 'parent', 'left_child'
# and 'right_child' attributes represent integer offsets within the
# array that specify other Node structs.
#
# WARNING : When part of a SuchLinkedTree, the right_child is used to
# store the column ID for leaf nodes. Check *ONLY* left_child == -1 to
# to see if a Node is a leaf!
cdef struct Node :
    int parent
    int left_child
    int right_child
    float support
    float distance

@cython.boundscheck(False)
cdef double _pearson( double[:] x, double[:] y, unsigned int n ) nogil :
    #cdef unsigned int n = len(x)
    cdef unsigned long j
    cdef float yt, xt, t, df
    cdef float syy=0.0, sxy=0.0, sxx=0.0, ay=0.0, ax=0.0
    for j in xrange( n ) :
        ax += x[j]
        ay += y[j]
    ax /= n
    ay /= n
    for j in xrange( n ) :
        xt  =  x[j] - ax
        yt  =  y[j] - ay
        sxx += xt * xt
        syy += yt * yt
        sxy += xt * yt
    return sxy / ( ( sxx * syy ) + 1.0e-20 )**(0.5)

def pearson( double[:] x, double[:] y ) :
    if not len(x) == len(y) :
        raise Exception( 'vectors must be the same length.', (len(x),len(y)) )
    try :
        return _pearson( x, y, len(x) )
    except ZeroDivisionError :
        return 0.0

@cython.no_gc_clear
cdef class SuchTree :
    '''
    SuchTree extention type. The constructor accepts a filesystem
    path or URL to a file that describes the tree in NEWICK format.
    For now, SuchTree uses dendropy to parse the NEWICK file.

    An array of type Node is allocated, and freed when
    SuchTree.__dealloc__ is invoked.
    
    Node.parent, Node.left_child and Node.right_child are integer
    offsets within this array, describing the tree structure.
    Nodes where left_child and right_child are -1 are leaf nodes,
    Nodes where the parent attribute is -1 are the root nodes
    (there should be only one of these in any given tree).
    
    SuchTree expects trees to be strictly bifrucating. There
    should not be any nodes that have only one child.
    
    SuchTrees are immutable; they cannot be modified once
    initialized. If you need to manipulate your tree before
    performing computations, you will need to use a different tool
    to perform those manipulations first.
    '''
    
    cdef Node* data
    cdef unsigned int length
    cdef unsigned int depth
    cdef unsigned int n_leafs
    cdef unsigned int root
    cdef np.float64_t epsilon
    cdef object leafs
    cdef object internal_nodes
    cdef object leafnodes
    cdef object np_buffer
    cdef object RED
    
    def __init__( self, tree_input ) :
        '''
        SuchTree constructor.
        '''
        cdef unsigned int n
        cdef int node_id
        self.np_buffer = None
        self.n_leafs = 0
        
        # tiny nonzero distance for representing polytomies
        self.epsilon = np.finfo( np.float64 ).eps
        
        url_strings = [ 'http://', 'https://', 'ftp://' ]
        
        if any( [ tree_input.startswith(x) for x in url_strings ] ) :
            t = Tree.get( url=tree_input,
                          schema='newick',
                          preserve_underscores=True,
                          suppress_internal_node_taxa=True )
        if all( [ '(' in tree_input,
                  ')' in tree_input,
                  tree_input.count( '(' ) == tree_input.count( ')' ) ] ) :
            t = Tree.get( data=tree_input,
                          schema='newick',
                          preserve_underscores=True,
                          suppress_internal_node_taxa=True )
        else :
            t = Tree.get( file=open(tree_input),
                          schema='newick',
                          preserve_underscores=True,
                          suppress_internal_node_taxa=True )
        
        t.resolve_polytomies()
        size = len( t.nodes() )
        # allocate some memory
        self.data    = <Node*> PyMem_Malloc( size * sizeof(Node) )
        if self.data == NULL :
            raise Exception( 'SuchTree could not allocate memory' )
        
        self.length = size
        if not self.data :
            raise MemoryError()
        
        self.leafs = {}
        self.leafnodes = {}
        self.internal_nodes = []
        for node_id,node in enumerate( t.inorder_node_iter() ) :
            node.node_id = node_id
            if node_id >= size :
                raise Exception( 'node label out of bounds : ' + str(node_id) )
            if node.taxon :
                self.leafs[ node.taxon.label ] = node_id
                self.leafnodes[ node_id ] = node.taxon.label
            else :
                self.internal_nodes.append( node_id )
        
        for node_id,node in enumerate( t.inorder_node_iter() ) :
            if not node.parent_node :
                distance = -1.0
                parent   = -1
                self.root = node_id
            else :
                if not node.edge_length :
                    distance = self.epsilon
                else :
                    if node.edge_length == 0 :
                        distance = self.epsilon
                    else :
                        distance = node.edge_length
                parent   = node.parent_node.node_id
            if node.taxon :
                left_child, right_child = -1, -1
                self.n_leafs += 1
            else :
                l_child, r_child = node.child_nodes()
                left_child  = l_child.node_id
                right_child = r_child.node_id
            
            if node_id >= size :
                raise Exception( 'node label out of bounds : ' + str(node_id) )
            
            try :
                support = float( node.label )
            except ( TypeError, ValueError ) :
                support = -1
            
            self.data[node_id].parent      = parent
            self.data[node_id].left_child  = left_child
            self.data[node_id].right_child = right_child
            self.data[node_id].distance    = distance
            self.data[node_id].support     = support
        
        for node_id in self.leafs.values() :
            n = 1
            while True :
                if self.data[node_id].parent == -1 : break
                node_id = self.data[node_id].parent
                n += 1
            if n > self.depth :
                self.depth = n
        
        # RED dictionary stub
        self.RED = {}
    
    def __dealloc__( self ) :
        '''SuchTree destructor.'''
        PyMem_Free( self.data )     # no-op if self.data is NULL

    # ====== SuchTree properties ======
    
    @property
    def size( self ) -> int :
        '''The number of nodes in the tree.'''
        # renamed from 'length'
        return self.length
    
    @property
    def depth( self ) -> int :
        '''The maximum depth of the tree.'''
        return self.depth
    
    @property
    def num_leaves( self ) -> int :
        '''The number of leaf nodes in the tree.'''
        # renamed from 'n_leafs'
        return self.n_leafs
    
    @property
    def leaves( self ) -> Dict[ str, int ] :
        '''Dictionary mapping leaf names to node IDs.'''
        # renamed from 'leafs'
        return self.leafs
    
    @property
    def leaf_nodes( self ) -> Dict[ int, str ] :
        '''Dictionary mapping leaf node IDs to names.'''
        # renamed from leafnodes
        return self.leafnodes
    
    @property
    def root_node( self ) -> int :
        '''The ID of the root node.'''
        # renamed from root
        return self.root
    
    @property
    def internal_nodes( self ) -> np.ndarray :
        '''Array of internal node IDs.'''
        return self.internal_nodes
    
    @property
    def all_nodes( self ) -> np.ndarray :
        '''Array of all node IDs in the tree.'''
        return np.concatiante( np.array( self.leaves.values() ),
                               np.array( self.internal_nodes ) )
    
    @property
    def leaf_node_ids( self ) -> np.ndarray :
        '''Array of leaf node IDs.'''
        return np.array( list( self.leaves.values() ) )
    
    @property
    def leaf_names( self ) -> list :
        '''List of all leaf names.'''
        return list( self.leaves.keys() )
    
    @property
    def polytomy_epsilon(self) -> float :
        '''Tiny, arbitrary, nonzero distance for polytomies.'''
        # renamed from polytomy_distance
        return self.epsilon
    
    @polytomy_epsilon.setter
    def polytomy_epsilon( self, new_epsilon: float ) -> None :
        '''Set the polytomy epsilon value.'''
        self.epsilon = new_epsilon
    
    @property
    def relative_evolutionary_divergence( self ) -> Dict[ int, float ] :
        ''' 
        The relative evolutionary divergence (RED) of the nodes in the tree.
        The RED of a node is the relative placement between the root and its
        descending tips (Parks et al. 2018). RED is defined to range from
        0 at the root node to 1 at each leaf. Traversing the tree in pre-order,
        RED is P+(a/(a+b))*(1-P), where P is the RED of the node's parent,
        a is the distance to its parent, and b is the average distance from
        the node to its leaf descendants.
        
        RED is calculated for every node in the tree and returned as
        a dictionary. Once computed, the RED dictionary will be cached and made
        available as the SuchTree.RED attribute.
        '''
        if not self.RED :
        
            self.RED = { self.root : 0 }
        
            for node in list( self.pre_order() )[1:] :
                P = self.RED[ self.get_parent(node) ]
                a = self.distance( node, self.get_parent(node) )
                b = np.mean( [ self.distance( node, leaf ) for leaf in self.get_leafs(node) ] )
                if a+b == 0 :
                    raise Exception( 'node {n} : a={a}, b={b}'.format( n=node, a=a, b=b ) )
                self.RED[ node ] = P+(a/(a+b))*(1-P)
    
        return self.RED
    
    # ====== Depricated properties ======
    
    property length :
        'The number of nodes in the tree.'
        def __get__( self ) :
            warn( 'SuchTree.length is depiracted in favor of SuchTree.size',
                  category=DeprecationWarning, stacklevel=2 )
            return self.length
            
    property n_leafs :
        'The number of leafs in the tree.'
        def __get__( self ) :
            warn( 'SuchTree.n_leafs is depricated in favor of SuchTree.num_leafs',
                  category=DeprecationWarning, stacklevel=2 )
            return self.n_leafs
            
    property leafs :
        'A dictionary mapping leaf names to leaf node ids.'
        def __get__( self ) :
            warn( 'SuchTree.leafs is depricated in favor of SuchTree.leaves',
                  category=DeprecationWarning, stacklevel=2 )
            return self.leafs
    
    property leafnodes :
        'A dictionary mapping leaf node ids to leaf names.'
        def __get__( self ) :
            warn( 'SuchTree.leafnodes is depricated in favor of SuchTree.leaf_nodes',
                  category=DeprecationWarning, stacklevel=2 )
            return self.leafnodes

    property root :
        'The id of the root node.'
        def __get__( self ) :
            warn( 'SuchTree.root is depriacted in favor of SuchTree.root_node',
                  category=DeprecationWarning, stacklevel=2 )
            return self.root
            
    property polytomy_distance :
        'Tiny, nonzero distance for polytomies in the adjacency matrix.'
        def __get__( self ) :
            warn( 'SuchTree.polytomy_distance is depricated in favor of SuchTree.polytomy.epsilon',
                  category=DeprecationWarning, stacklevel=2 )
            return self.epsilon
        def __set__( self, np.float64_t new_epsilon ) :
            warn( 'SuchTree.polytomy_distance is depricated in favor of SuchTree.polytomy.epsilon',
                  category=DeprecationWarning, stacklevel=2 )
            self.epsilon = new_epsilon
    
    @property
    def RED( self ) -> Dict[ int, float ] :
        'Alias for relative_evolutionary_divergence (deprecated).'''
        warn( 'SuchTree.RED is deprecated in favor of SuchTree.relative_evolutionary_divergence', 
              category=DeprecationWarning, stacklevel=2 )
        return self.relative_evolutionary_divergence
    
    # ====== Node query methods ======
    
    def get_parent( self,
                    node : Union[ int, str ] ) -> int :
        '''
        Return the parent node ID for a given node.
        
        Args
            node : Node ID or leaf name
            
        Returns
            int : Parent node ID (parent of root is -1)
            
        Raises
            NodeNotFoundError : If leaf name is not found
            InvalidNodeError  : If node ID is out of bounds
        '''
        node_id = self._validate_node( node )
        return self.data[ node_id ].parent
    
    def get_children(self, node: Union[int, str]) -> Tuple[int, int]:
        '''
        Return the child node IDs for a given node.
        
        Args
            node : Node ID or leaf name
            
        Returns
            Tuple[ int, int ] : Left and right child node IDs (child of leaf is -1)
            
        Raises
            NodeNotFoundError : If leaf name is not found
            InvalidNodeError  : If node ID is out of bounds
        '''
        node_id = self._validate_node( node )
        return ( self.data[ node_id ].left_child,
                 self.data[ node_id ].right_child )
    
    def get_ancestors( self, 
                       node : Union[ int, str ] ) -> Generator[ int, None, None ] :
        '''
        Generator yielding ancestor node IDs from node to root.
        
        Renamed from get_lineage() for clarity.
        
        Args
            node : Node ID or leaf name
            
        Yields
            int : Ancestor node IDs in order from parent to root
            
        Raises
            NodeNotFoundError: If leaf name is not found
            InvalidNodeError: If node ID is out of bounds
        '''
        node_id = self._validate_node( node )
        
        while True :
            parent_id = self.data[ node_id ].parent
            if parent_id == -1 :
                break
            yield parent_id
            node_id = parent_id
    
    def get_lineage( self, query ) :
        '''
        Generator of parent nodes up to the root node. Will accept
        node id of leaf name.
        '''
        # FIXME : Not sure if this will work as a pass-through for a generator
        warn( 'SuchTree.get_lineage is depricated in favor of SuchTree.get_ancestors',
              category=DeprecationWarning, stacklevel=2 )
        return self.get_ancestors( query )
    
    def get_descendants( self,
                         node_id : int ) -> Generator[ int, None, None ] :
        '''
        Generator yielding all descendant node IDs from a given node.
        
        Renamed from get_descendant_nodes() for consistency.
        
        Args
            node_id : Node ID
            
        Yields
            int : Descendant node IDs including the starting node
            
        Raises
            NodeNotFoundError : If leaf name is not found
            InvalidNodeError  : If node ID is out of bounds
        '''
        node_id = self._validate_node( node_id )
        
        to_visit = [ node_id ]
        for current_id in to_visit :
            left_child, right_child = self.get_children( current_id )
            if left_child == -1 :
                # Leaf node
                yield current_id
            else :
                # Internal node
                to_visit.append( left_child  )
                to_visit.append( right_child )
                yield current_id
    
    def get_descendant_nodes( self, node_id ) :
        '''
        Generator for ids of all nodes descendent from a given node,
        starting with the given node. Can only accept a node_ids, 
        returns node_ids for internal and leaf nodes.
        '''
        # FIXME : Not sure if this will work as a pass-through for a generator
        warn( 'SuchTree.get_lineage is depricated in favor of SuchTree.get_ancestors',
              category=DeprecationWarning, stacklevel=2 )
        return self.get_descendants( node_id )
    
    def get_leaves( self,
                    node : Union[ int, str ] ) -> np.ndarray :
        '''
        Return array of leaf node IDs descended from a given node.
        
        Renamed from get_leafs() with corrected pluralization.
        
        Args
            node : Node ID or leaf name
            
        Returns
            np.ndarray : Array of leaf node IDs
            
        Raises
            NodeNotFoundError : If leaf name is not found
            InvalidNodeError  : If node ID is out of bounds
        '''
        node_id = self._validate_node( node )
        
        if self.np_buffer is None:
            self.np_buffer = np.ndarray( self.num_leaves, dtype=int )
        
        to_visit = [ node_id ]
        leaf_count = 0
        
        for current_id in to_visit :
            left_child, right_child = self.get_children( current_id )
            if left_child == -1 :
                # This is a leaf node
                self.np_buffer[ leaf_count ] = current_id
                leaf_count += 1
            else :
                # This is an internal node, add children to visit
                to_visit.append( left_child  )
                to_visit.append( right_child )
        
        return np.array( self.np_buffer[ : leaf_count ] )
    
    def get_leafs( self, node_id ) :
        '''
        Return an array of ids of all leaf nodes descendent from a given node.
        '''
        # FIXME : Not sure if this will work as a pass-through for a generator
        warn( 'SuchTree.get_leafs is depricated in favor of SuchTree.get_leaves',
              category=DeprecationWarning, stacklevel=2 )
        return self.get_leaves( node_id )
    
    def get_support( self,
                     node : Union[ int, str ] ) -> float :
        '''
        Return the support value for a given node.
        
        Args
            node : Node ID or leaf name
            
        Returns
            float : Support value (-1 if no support available)
            
        Raises
            NodeNotFoundError : If leaf name is not found
            InvalidNodeError  : If node ID is out of bounds
        '''
        node_id = self._validate_node(node)
        return self.data[node_id].support
    
    def get_internal_nodes( self,
                            from_node : Union[ int, str ] = None ) -> np.ndarray :
        '''
        Return array of internal node IDs.
        
        Args
            from_node : Starting node (default: root)
            
        Returns
            np.ndarray : Array of internal node IDs
            
        Raises
            NodeNotFoundError : If from_node leaf name is not found
            InvalidNodeError  : If from_node ID is out of bounds
        '''
        if from_node is None :
            from_node = self.root_node
        else:
            from_node = self._validate_node( from_node )
        
        if self.np_buffer is None :
            self.np_buffer = np.ndarray( self.num_leaves, dtype=int )
        
        to_visit = [ from_node ]
        internal_count = 0
        
        for current_id in to_visit :
            left_child, right_child = self.get_children( current_id )
            if left_child == -1 :
                # Leaf node, skip
                continue
            else :
                # Internal node
                to_visit.append( left_child  )
                to_visit.append( right_child )
                self.np_buffer[ internal_count ] = current_id
                internal_count += 1
        
        return np.array( self.np_buffer[ : internal_count ] )
    
    def get_nodes( self,
                   from_node : Union[ int, str ] = None ) -> np.ndarray :
        '''
        Return array of all node IDs.
        
        Args
            from_node : Starting node (default: root)
            
        Returns
            np.ndarray : Array of all node IDs
            
        Raises:
            NodeNotFoundError : If from_node leaf name is not found
            InvalidNodeError  : If from_node ID is out of bounds
        '''
        if from_node is None :
            from_node = self.root_node
        else :
            from_node = self._validate_node( from_node )
        
        if self.np_buffer is None :
            self.np_buffer = np.ndarray( self.size, dtype=int )
        
        to_visit = [ from_node ]
        node_count = 0
        
        for current_id in to_visit :
            self.np_buffer[ node_count ] = current_id
            node_count += 1
            
            left_child, right_child = self.get_children( current_id )
            if left_child != -1 :
                to_visit.append( left_child  )
                to_visit.append( right_child )
        
        return np.array( self.np_buffer[ : node_count ] )

    def get_internal_nodes( self, from_node=-1 ) :
        '''
        Return an array of the ids of all internal nodes.
        '''
        cdef unsigned int i
        cdef int l
        cdef int r
        cdef unsigned int n = 0
        
        if from_node == -1 : from_node = self.root
        
        self.np_buffer = np.ndarray( self.n_leafs, dtype=int )
        
        # this doesn't look like it should work, but strictly
        # bifrucating trees always have one fewer internal nodes
        # than leaf nodes
        
        to_visit = [from_node]
        for i in to_visit :
            l,r = self.get_children( i )
            if l == -1 :
                continue
            else :
                to_visit.append( l )
                to_visit.append( r )
                self.np_buffer[n] = i
                n += 1
        return np.array(self.np_buffer[:n])
        
    def get_nodes( self, from_node=-1 ) :
        '''
        Return an array of the ids of all nodes.
        '''
        cdef unsigned int i
        cdef int l
        cdef int r
        cdef unsigned int n = 0
        
        if from_node == -1 : from_node = self.root
        
        self.np_buffer = np.ndarray( self.length, dtype=int )
        
        to_visit = [from_node]
        for i in to_visit :
            l,r = self.get_children( i )
            self.np_buffer[n] = i
            n += 1
            if l != -1 :
                to_visit.append( l )
                to_visit.append( r )
        return np.array(self.np_buffer[:n])
    
    # ====== Node test methods ======    
    
    def is_leaf( self,
                 node : Union[ int, str ] ) -> bool :
        '''
        Test if a node is a leaf node.
        
        Args
            node : Node ID or leaf name
            
        Returns
            bool : True if node is a leaf, False otherwise
            
        Raises
            NodeNotFoundError : If leaf name is not found
            InvalidNodeError  : If node ID is out of bounds
        '''
        node_id = self._validate_node( node )
        return self._is_leaf( node_id )
    
    def is_internal( self,
                     node : Union[ int, str ] ) -> bool :
        '''
        Test if a node is an internal node.
        
        Renamed from is_internal_node() for consistency.
        
        Args
            node : Node ID or leaf name
            
        Returns
            bool : True if node is internal, False otherwise
            
        Raises
            NodeNotFoundError : If leaf name is not found
            InvalidNodeError  : If node ID is out of bounds
        '''
        node_id = self._validate_node( node )
        return not self._is_leaf( node_id )
    
    def is_internal_node( self, node_id ) :
        '''
        Returns True if node_id is an internal node, False otherwise.
        '''
        # FIXME : Not sure if this will work as a pass-through for a generator
        warn( 'SuchTree.is_internal_node is depricated in favor of SuchTree.is_internal',
              category=DeprecationWarning, stacklevel=2 )
        return not self._is_leaf( node_id )
    
    @cython.boundscheck(False)
    cdef bint _is_leaf( self, int node_id ) nogil :
        if self.data[node_id].left_child == -1 :
            return True
        else :
            return False
    
    def is_ancestor( self,
                     ancestor   : Union[ int, str ],
                     descendant : Union[ int, str ] ) -> int :
        '''
        Test ancestral relationship between two nodes.
        
        Args
            ancestor   : Potential ancestor node (ID or name)
            descendant : Potential descendant node (ID or name)
            
        Returns
            int :  1 if ancestor is ancestor of descendant,
                  -1 if descendant is ancestor of ancestor,
                   0 if neither is ancestor of the other
                
        Raises
            NodeNotFoundError : If any leaf name is not found
            InvalidNodeError  : If any node ID is out of bounds
        '''
        ancestor_id, descendant_id = self._validate_node_pair( ancestor, descendant )
        return self._is_ancestor( ancestor_id, descendant_id )
    
    def is_descendant( self,
                       descendant : Union[ int, str ],
                       ancestor   : Union[ int, str ] ) -> bool:
        '''
        Test if descendant is a descendant of ancestor.
        
        New method for clarity - complements is_ancestor().
        
        Args
            descendant : Potential descendant node (ID or name)
            ancestor   : Potential ancestor node (ID or name)
            
        Returns
            bool : True if descendant is a descendant of ancestor
            
        Raises
            NodeNotFoundError : If any leaf name is not found
            InvalidNodeError  : If any node ID is out of bounds
        '''
        ancestor_id, descendant_id = self._validate_node_pair( ancestor, descendant )
        return self._is_ancestor( ancestor_id, descendant_id ) == 1
    
    @cython.boundscheck(False)
    cdef int _is_ancestor( self, int a, int b ) nogil :
        cdef int i
        cdef int n
        
        # is a an ancestor of b?
        i = b
        while True :
            n = self.data[i].parent
            if n == -1 :
                break
            if n == a :
                return 1
            i = n
        
        # is b an ancestor of a?
        i = a
        while True :
            n = self.data[i].parent
            if n == -1 :
                break
            if n == b :
                return -1
            i = n
        
        # or neither?
        return 0
    
    def is_root( self,
                 node : Union[ int, str ] ) -> bool :
        '''
        Test if a node is the root node.
        
        Args
            node : Node ID or leaf name
            
        Returns
            bool : True if node is the root
            
        Raises
            NodeNotFoundError : If leaf name is not found
            InvalidNodeError  : If node ID is out of bounds
        '''
        node_id = self._validate_node( node )
        return node_id == self.root_node
    
    def is_sibling( self,
                    node1 : Union[ int, str ],
                    node2 : Union[ int, str ] ) -> bool :
        '''
        Test if two nodes are siblings (share the same parent).
        
        Args
            node1 : First node (ID or name)
            node2 : Second node (ID or name)
            
        Returns
            bool : True if nodes are siblings
            
        Raises
            NodeNotFoundError : If any leaf name is not found
            InvalidNodeError  : If any node ID is out of bounds
        '''
        node1_id, node2_id = self._validate_node_pair( node1, node2 )
        
        # Root node has no siblings
        if node1_id == self.root_node or node2_id == self.root_node :
            return False
        
        parent1 = self.data[node1_id].parent
        parent2 = self.data[node2_id].parent
        
        return parent1 == parent2 and parent1 != -1
    
    def has_children( self,
                      node : Union[ int, str ] ) -> bool :
        '''
        Test if a node has children (i.e., is not a leaf).
        
        Args
            node : Node ID or leaf name
            
        Returns
            bool : True if node has children
            
        Raises
            NodeNotFoundError : If leaf name is not found
            InvalidNodeError  : If node ID is out of bounds
        '''
        return self.is_internal(node)
    
    def has_parent( self,
                    node : Union[ int, str ] ) -> bool :
        '''
        Test if a node has a parent (i.e., is not the root).
        
        Args
            node : Node ID or leaf name
            
        Returns
            bool : True if node has a parent
            
        Raises
            NodeNotFoundError : If leaf name is not found
            InvalidNodeError  : If node ID is out of bounds
        '''
        return not self.is_root(node)
    
    # ====== Distance methods ======
    
    def distance_to_root( self,
                          node : Union[ int, str ] ) -> float :
        '''
        Return distance from a node to the root.
        
        Renamed from get_distance_to_root() for consistency.
        
        Args
            node : Node ID or leaf name
            
        Returns
            float : Distance to root node
            
        Raises
            NodeNotFoundError : If leaf name is not found
            InvalidNodeError  : If node ID is out of bounds
        '''
        node_id = self._validate_node(node)
        return self._get_distance_to_root(node_id)
    
    def get_distance_to_root( self, a ) :
        '''
        Return distance to root for a given node. Will accept node id
        or a leaf name.
        '''
        warn( 'SuchTree.get_distance_to_root is depricated in favor of SuchTree.distance_to_root',
              category=DeprecationWarning, stacklevel=2 )
        return self.distance_to_root( a )
    
    @cython.boundscheck(False)
    cdef float _get_distance_to_root( self, node_id ) :
        '''
        Calculate the distance from a node of a given id to the root node.
        Will work for both leaf and internal nodes. Private cdef method.
        '''
        cdef float d = 0.0
        cdef float d_i = 0.0
        cdef int i = node_id
        cdef int a_depth = 0
        cdef int mrca = -1
        
        while True :
            d_i = self.data[i].distance
            if d_i == -1 : break
            d = d + d_i
            i = self.data[i].parent
        return d
 
    def distance( self, 
                  a : Union[ int, str ],
                  b : Union[ int, str ] ) -> float :
        '''
        Calculate patristic distance between two nodes.
        
        Args
            a : First node (ID or name)
            b : Second node (ID or name)
            
        Returns
            float : Patristic distance between the nodes
            
        Raises
            NodeNotFoundError : If any leaf name is not found
            InvalidNodeError  : If any node ID is out of bounds
        '''
        node_a, node_b = self._validate_node_pair( a, b )
        return self._distance( node_a, node_b )
    
    def distances_bulk( self,
                        pairs : np.ndarray ) -> np.ndarray :
        '''
        Calculate distances for multiple node pairs efficiently.
        
        Renamed from distances() for clarity about bulk operation.
        
        Args
            pairs : (n, 2) array of node ID pairs
            
        Returns
            np.ndarray : Array of n distances
            
        Raises
            ValueError       : If pairs array shape is incorrect
            InvalidNodeError : If any node ID is out of bounds
        '''
        if not isinstance( pairs, np.ndarray ) :
            pairs = np.array( pairs, dtype=np.int64 )
        
        if pairs.ndim != 2 or pairs.shape[1] != 2 :
            shape = str( ( pairs.shape[0], pairs.shape[1] ) )
            raise ValueError( 'Expected (n, 2) array, got shape {shape}'.format( shape=shape ) )
        
        # Validate all node IDs in the array
        max_id = pairs.max()
        min_id = pairs.min()
        if min_id < 0 or max_id >= self.size :
            raise InvalidNodeError(
                max_id if max_id >= self.size else min_id, 
                self.size
            )
        
        # Use optimized Cython method
        visited = np.zeros( self.depth, dtype=int )
        result = np.zeros( pairs.shape[0], dtype=float )
        self._distances( pairs.shape[0], visited, pairs, result )
        return result
    
    def distances( self, long[:,:] ids ) :
        '''
        Returns an array of distances between pairs of node ids,
        which are expected as an (n,2) array of type int.
        '''
        warn( 'SuchTree.distances is depricated in favor of SuchTree.distances_bulk',
              category=DeprecationWarning, stacklevel=2 )
        return self.distances_bulk( ids )
        
    @cython.boundscheck(False)
    cdef void _distances( self, unsigned int length,
                                long[:] visited,
                                long[:,:] ids,
                                double[:] result ) nogil :
        '''
        For each pair of node ids in the given (n,2) array, calculate the
        distance to the root node for each pair and store their differece
        in the given (1,n) result array. Calculations are performed within
        a 'nogil' context, allowing the interpreter to perform other tasks
        concurrently if desired. Private cdef method.
        '''
        cdef unsigned int mrca
        cdef float d
        cdef unsigned int n
        cdef unsigned int a
        cdef unsigned int b
        cdef unsigned int i
        
        for i in xrange( ids.shape[0] ) :
            a = ids[i,0]
            b = ids[i,1]
            mrca = self._mrca( visited, a, b )
            n = a
            d = 0
            while n != mrca :
                d += self.data[n].distance
                n =  self.data[n].parent
            n = b
            while n != mrca :
                d += self.data[n].distance
                n =  self.data[n].parent
            result[i] = d
    
    def distances_by_name( self,
                           pairs : List[ Tuple[ str, str ] ] ) -> List[ float ] :
        '''
        Calculate distances for pairs of leaf names.
        
        Args
            pairs : List of (leaf_name1, leaf_name2) tuples
            
        Returns
            List[ float ] : List of patristic distances
            
        Raises
            NodeNotFoundError : If any leaf name is not found
            TypeError         : If pairs is not a list of tuples
        '''
        if not isinstance( pairs, list ) :
            raise TypeError( 'pairs must be a list of tuples' )
        
        # Convert names to IDs
        node_pairs = []
        for i, ( name_a, name_b ) in enumerate( pairs ) :
            if not isinstance( name_a, str ) or not isinstance( name_b, str ) :
                raise TypeError( 'Pair {i}: both elements must be strings'.format( i=str(i) ) )
            
            if not name_a in self.leaves :
                raise NodeNotFoundError( name_a )
            if not name_b in self.leaves :
                raise NodeNotFoundError( name_b )
            
            node_pairs.append( ( self.leaves[name_a],
                                 self.leaves[name_b] ) )
        
        # Use bulk calculation
        pairs_array = np.array( node_pairs, dtype=np.int64 )
        return self.distances_bulk( pairs_array ).tolist()
    
    @cython.boundscheck(False)
    cdef float _distance( self, int a, int b ) :
        cdef int mrca
        cdef float d = 0
        cdef int n
        
        mrca = self.mrca( a, b )
        
        n = a
        while n != mrca :
            d += self.data[n].distance
            n =  self.data[n].parent
        n = b
        while n != mrca :
            d += self.data[n].distance
            n =  self.data[n].parent
        return d
    
    @cython.boundscheck(False)
    cdef int _mrca( self, long[:] visited, int a, int b ) noexcept nogil :
        cdef int n
        cdef int i
        cdef int mrca = -1
        cdef int a_depth
        
        n = a
        i = 0
        while True :
            visited[i] = n
            n = self.data[n].parent
            i += 1
            if n == -1 : break
        a_depth = i
        
        n = b
        while True :
            i = 0
            while True :
                if i >= a_depth : break
                if visited[i] == n :
                    mrca = visited[i]
                    break
                i += 1
            if mrca != -1 : break
            n = self.data[n].parent
            if n == -1 :
                mrca = n
                break
            
        return mrca
    
    def nearest_neighbors( self,
                           node : Union[ int, str ],
                           k : int = 1,
                           from_nodes : List[ Union[ int, str ] ] = None ) -> List[ Tuple[ Union[ int, str ],
                                                                                           float ] ] :
        '''
        Find the k nearest neighbors to a given node.
        
        Args
            node       : Query node (ID or name)
            k          : Number of nearest neighbors to return
            from_nodes : Nodes to search among (default: all leaves except query)
            
        Returns
            List[ Tuple[ Union[ int, str ], float ] ] : List of (node, distance) pairs
            
        Raises
            NodeNotFoundError : If any leaf name is not found
            InvalidNodeError  : If any node ID is out of bounds
            ValueError        : If k is not positive
        '''
        if k <= 0:
            raise ValueError( 'k must be positive' )
        
        query_node_id = self._validate_node(node)
        
        if from_nodes is None :
            # Use all leaves except the query node (if it's a leaf)
            if self.is_leaf(query_node_id) :
                from_node_ids   = [ nid for nid in self.leaf_node_ids if nid != query_node_id ]
                from_nodes_orig = [ self.leaf_nodes[nid] for nid in from_node_ids ]
            else :
                from_node_ids   = self.leaf_node_ids
                from_nodes_orig = [ self.leaf_nodes[nid] for nid in from_node_ids ]
        else :
            from_node_ids   = [ self._validate_node(n) for n in from_nodes ]
            from_nodes_orig = from_nodes.copy()
        
        # Calculate distances to all candidate nodes
        pairs = [ (query_node_id, nid) for nid in from_node_ids ]
        distances = self.distances_bulk( np.array( pairs, dtype=np.int64 ) )
        
        # Sort by distance and return top k
        sorted_indices = np.argsort( distances )
        result = []
        for i in sorted_indices[:k] :
            result.append( ( from_nodes_orig[i], distances[i] ) )
        
        return result
    
    def pairwise_distances( self,
                            nodes : List[ Union[ int, str ] ] = None ) -> np.ndarray :
        '''
        Calculate all pairwise distances between given nodes.
        
        Args
            nodes : List of nodes (IDs or names). If None, uses all leaves.
            
        Returns
            np.ndarray : Symmetric distance matrix
            
        Raises
            NodeNotFoundError : If any leaf name is not found
            InvalidNodeError  : If any node ID is out of bounds
        '''
        if nodes is None :
            # Use all leaf nodes by default
            node_ids = self.leaf_node_ids
        else :
            # Validate and convert to node IDs
            node_ids = np.array( [ self._validate_node(node) for node in nodes ] )
        
        n = len(node_ids)
        distance_matrix = np.zeros( (n, n), dtype=float )
        
        # Generate all pairs (upper triangle)
        # FIXME : use combinations()
        pairs = []
        pair_indices = []
        for i in range(n) :
            for j in range(i + 1, n) :
                pairs.append( ( node_ids[i], node_ids[j] ) )
                pair_indices.append( (i, j) )
        
        if pairs :  # Only calculate if there are pairs
            distances = self.distances_bulk( np.array( pairs, dtype=np.int64 ) )
            
            # Fill in the distance matrix
            for dist, (i, j) in zip( distances, pair_indices ) :
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist  # Symmetric
        
        return distance_matrix
    
    # ====== Topology Methods ======
    
    def common_ancestor( self,
                         a : Union[ int, str ],
                         b : Union[ int, str ] ) -> int :
        '''
        Find the most recent common ancestor of two nodes.
        
        Renamed from mrca().
        
        Args
            a : First node (ID or name)
            b : Second node (ID or name)
            
        Returns
            int : Node ID of most recent common ancestor
            
        Raises
            NodeNotFoundError : If any leaf name is not found
            InvalidNodeError  : If any node ID is out of bounds
        '''
        node_a, node_b = self._validate_node_pair( a, b )
        visited = np.zeros( self.depth, dtype=int )
        return self._mrca( visited, node_a, node_b )

    def mrca( self, a, b ) :
        '''
        Return the id of the most recent common ancestor of two nodes
        if given ids. Leaf names or node_ids can be used for leafs,
        but node_ids must be used for internal nodes.
        '''
        warn( 'SuchTree.mrca is depricated in favor of SuchTree.common_ancestor',
              category=DeprecationWarning, stacklevel=2 )
        visited = np.zeros( self.depth, dtype=int )
        return self._mrca( visited, a, b )
    
    def bipartition( self,
                     node  : Union[ int, str ],
                     by_id : bool = False ) -> frozenset :
        '''
        Get the bipartition created by an internal node.
        
        Renamed from get_bipartition() for consistency.
        
        Args
            node  : Internal node (ID or name)
            by_id : If True, return node IDs; if False, return leaf names
            
        Returns
            frozenset : Frozenset of two frozensets representing the bipartition
            
        Raises
            NodeNotFoundError : If leaf name is not found
            InvalidNodeError  : If node ID is out of bounds or is a leaf node
        '''
        node_id = self._validate_internal_node( node )
        left_child, right_child = self.get_children( node_id )
        
        if by_id :
            return frozenset((
                frozenset( self.get_leaves( left_child  )),
                frozenset( self.get_leaves( right_child ))
            ))
        else :
            left_leaves  = self._convert_to_leaf_names( self.get_leaves( left_child  ))
            right_leaves = self._convert_to_leaf_names( self.get_leaves( right_child ))
            return frozenset((
                frozenset( left_leaves  ),
                frozenset( right_leaves )
            ))

    def get_bipartition( self, node_id, by_id=False ) :
        '''
        Find the two sets of leaf nodes partitioned at an internal
        node in the tree.
        '''
        warn( 'SuchTree.get_bipartition is depricated in favor of SuchTree.bipartition',
              category=DeprecationWarning, stacklevel=2 )
        return self.bipartition( node_id, by_id=by_id )
    
    def bipartitions( self,
                      by_id : bool = False ) -> Generator[ frozenset, None, None ] :
        '''
        Generate all bipartitions in the tree. Each bipartition is
        the pair of sets of leaf nodes partitioned by an internal
        node in the tree. 
        
        Args
            by_id : If True, yield node IDs; if False, yield leaf names
            
        Yields
            frozenset : Bipartition as frozenset of two frozensets
        '''
        for node_id in self.get_internal_nodes():
            yield self.bipartition(node_id, by_id=by_id)
   
    def quartet_topology( self,
                          a : Union[ int, str ],
                          b : Union[ int, str ], 
                          c : Union[ int, str ],
                          d : Union[ int, str ] ) -> frozenset :
        '''
        Determine the topology of a quartet of taxa.
        
        Renamed from get_quartet_topology() for consistency.
        
        Args
            ( a, b ), ( c, d ) : Four nodes (IDs or names) forming the quartet
            
        Returns
            frozenset : Topology as frozenset of two frozensets representing sister pairs
            
        Raises
            NodeNotFoundError : If any leaf name is not found
            InvalidNodeError  : If any node ID is out of bounds
        '''
        # Validate inputs
        nodes = [ a, b, c, d ]
        node_ids = [ self._validate_node(node) for node in nodes ]
        
        # Keep track of original input types for return value
        has_strings = any( isinstance( node, str ) for node in nodes )
        
        visited = np.zeros( self.depth, dtype=int )
        
        # Calculate all pairwise MRCAs
        pairs = list( combinations( node_ids, 2 ) )
        mrcas = [ self._mrca( visited, x, y ) for x, y in pairs ]
        
        # Find unique MRCA (appears only once)
        unique_mrcas = [ mrca for mrca in mrcas if mrcas.count(mrca) == 1 ]
        
        if len( unique_mrcas ) == 1 :
            unique_mrca     = unique_mrcas[0]
            unique_pair_idx = mrcas.index( unique_mrca )
            sisters_ids     = pairs[ unique_pair_idx ]
            
            # The other two nodes form the second sister pair
            remaining_ids = frozenset( node_ids ) - frozenset( sisters_ids )
            
            if has_strings :
                # Convert back to original names if input had strings
                def id_to_original( node_id ) :
                    for orig, nid in zip( nodes, node_ids ) :
                        if nid == node_id :
                            return orig if isinstance( orig, str ) else self.leaf_nodes[ node_id ]
                    return self.leaf_nodes[node_id]
                
                sister_pair1 = frozenset( id_to_original( nid ) for nid in sisters_ids   )
                sister_pair2 = frozenset( id_to_original( nid ) for nid in remaining_ids )
            else:
                sister_pair1 = frozenset( sisters_ids   )
                sister_pair2 = frozenset( remaining_ids )
                
            return frozenset( ( sister_pair1, sister_pair2 ) )
        
        # Should not happen with valid quartet
        raise TreeStructureError( 'Could not determine unique topology for quartet {nodes}'.format( nodes=nodes ) )
    
    def get_quartet_topology( self, a, b, c, d ) :
        '''
        For a given quartet of taxa, return the topology of the quartet
        as a pair of tuples.
        '''
        warn( 'SuchTree.get_quartet_topology is depricated in favor of SuchTree.quartet_topology',
              category=DeprecationWarning, stacklevel=2 )
        return self.quartet_topology( a, b, c, d )
    
    def quartet_topologies_by_name( self, quartets ) :
        '''
        Wrapper method for quartet_topologies_bulk that accepts leaf names in
        the form

            [ [ a, b, c, d ], [ e, f, g, h ], ... ]
        '''
        # FIXME : This function should be merged with quartet_topologies_bulk
        #         so that it works the same way as quartet_topology
        Q = np.array( [ [ self.leaves[a],
                          self.leaves[b],
                          self.leaves[c],
                          self.leaves[d] ]
                        for a,b,c,d in quartets ] )
        
        return [ frozenset( ( frozenset( ( self.leaf_nodes[a],
                                           self.leaf_nodes[b] ) ),
                              frozenset( ( self.leaf_nodes[c],
                                           self.leaf_nodes[d] ) ) ) )
                 for a,b,c,d in self.quartet_topologies_bulk( Q ) ]
     
    def quartet_topologies_bulk( self,
                                 quartets : np.ndarray ) -> np.ndarray :
        '''
        Bulk processing function for computing quartet topologies.
        Takes an [N,4] matrix of taxon IDs, where the IDs are in
        arbitrary order
        
            [ [ a, b, c, d ], [ e, f, g, h ], ... ]
        
        and returns an [N,4] matrix of taxon IDs ordered such that
            
            [ { { a, b }, { c, d } }, { { e, f }, { g, h } }, ... ]
        
        Ordered taxa can be represented as a topology like so :
        
            topology = frozenset( ( frozenset( ( T[i,0], T[i,1] ),
                                    frozenset( ( T[i,2], T[i,3] ) ) ) ) )
        
        Renamed from quartet_topologies() for clarity.
        
        Args
            quartets : (n, 4) array of node IDs
            
        Returns
            np.ndarray : (n, 4) array where each row contains ordered node IDs 
                    representing the quartet topology
            
        Raises:
            ValueError       : If quartets array shape is incorrect
            InvalidNodeError : If any node ID is out of bounds
        '''
        if not isinstance(quartets, np.ndarray) :
            quartets = np.array( quartets, dtype=np.int64 )
        
        if quartets.ndim != 2 or quartets.shape[1] != 4 :
            shape = (<object>quartets).shape
            raise ValueError( f'Expected (n, 4) array, got shape {shape}' )
        
        # Validate all node IDs
        max_id = quartets.max()
        min_id = quartets.min()
        if min_id < 0 or max_id >= self.size :
            raise InvalidNodeError(
                max_id if max_id >= self.size else min_id,
                self.size
            )
        
        topologies = np.zeros( ( len(quartets), 4 ), dtype=int )
        visited = np.zeros( self.depth, dtype=int )
        M = np.zeros( 6, dtype=int )
        C = np.zeros( 6, dtype=int )
        
        # Possible topologies matrix
        I = np.array( [ [ 0, 1, 2, 3 ], [ 0, 2, 1, 3 ], [ 0, 3, 1, 2 ],
                        [ 1, 2, 0, 3 ], [ 1, 3, 0, 2 ], [ 2, 3, 0, 1 ] ] )
        
        self._quartet_topologies( quartets, topologies, visited, M, C, I )
        
        return topologies
    
    @cython.boundscheck(False)
    cdef void _quartet_topologies( self, long[:,:] quartets,
                                         long[:,:] topologies,
                                         long[:]   visited,
                                         long[:]   M,
                                         long[:]   C,
                                         long[:,:] I ) noexcept nogil : 
        cdef int i
        cdef int j
        cdef int k
        
        cdef int a
        cdef int b
        cdef int c
        cdef int d
        
        cdef int x
        cdef int y

        for i in range( len( quartets ) ) :
            a = quartets[i,0]
            b = quartets[i,1]
            c = quartets[i,2]
            d = quartets[i,3]
            
            # find MRCAs
            M[0] = self._mrca( visited, a, b )
            M[1] = self._mrca( visited, a, c )
            M[2] = self._mrca( visited, a, d )
            M[3] = self._mrca( visited, b, c )
            M[4] = self._mrca( visited, b, d )
            M[5] = self._mrca( visited, c, d )
            
            # find unique MCRA(s)
            for j in range( 6 ) :
                C[j] = 0
            for j in range( 6 ) :
                for k in range( 6 ) :
                    if M[j] == M[k] :
                        C[j] = C[j] + 1
            for j in range( 6 ) :
                if C[j] == 1 : break
            
            # reorder quartet ids by topology
            for k in range( 4 ) :
                topologies[i,k] = quartets[ i, I[j,k] ]
    
    def quartet_topologies_by_name( self,
                                    quartets : List[ Tuple[ str, 
                                                            str,
                                                            str,
                                                            str ] ] ) -> List[frozenset] :
        '''
        Compute quartet topologies for quartets specified by leaf names.
        
        Args
            quartets : List of tuples containing four leaf names each
            
        Returns
            List[frozenset] : List of quartet topologies as frozensets
            
        Raises:
            NodeNotFoundError : If any leaf name is not found
            TypeError         : If input format is incorrect
        '''
        # Convert names to IDs
        quartet_ids = []
        for i, (a, b, c, d) in enumerate(quartets) :
            if not all( isinstance( name, str ) for name in ( a, b, c, d ) ) :
                raise TypeError( f'Quartet {i}: all elements must be strings' )
            
            try :
                quartet_ids.append([
                    self.leaves[a], self.leaves[b], 
                    self.leaves[c], self.leaves[d]
                ])
            except KeyError as e :
                raise NodeNotFoundError( str(e).strip("'") )
        
        # Use bulk calculation
        quartet_array = np.array( quartet_ids, dtype=np.int64 )
        topologies = self.quartet_topologies_bulk( quartet_array )
        
        # Convert back to names and format as frozensets
        result = []
        for topology in topologies :
            a, b, c, d = topology
            sister1 = frozenset( ( self.leaf_nodes[a], self.leaf_nodes[b] ) )
            sister2 = frozenset( ( self.leaf_nodes[c], self.leaf_nodes[d] ) )
            result.append( frozenset( ( sister1, sister2 ) ) )
        
        return result
    
   
    def path_between_nodes(self, a: Union[int, str], b: Union[int, str]) -> List[int]:
        """Find the path between two nodes through their common ancestor.
        
        New convenience method.
        
        Args:
            a: First node (ID or name)
            b: Second node (ID or name)
            
        Returns:
            List[int]: List of node IDs forming the path from a to b
            
        Raises:
            NodeNotFoundError: If any leaf name is not found
            InvalidNodeError: If any node ID is out of bounds
        """
        node_a, node_b = self._validate_node_pair(a, b)
        
        if node_a == node_b:
            return [node_a]
        
        # Find common ancestor
        mrca = self.common_ancestor(node_a, node_b)
        
        # Path from a to MRCA (excluding MRCA)
        path_a = []
        current = node_a
        while current != mrca:
            path_a.append(current)
            current = self.data[current].parent
        
        # Path from b to MRCA (excluding MRCA)  
        path_b = []
        current = node_b
        while current != mrca:
            path_b.append(current)
            current = self.data[current].parent
        
        # Combine: a->mrca + mrca + mrca->b (reversed)
        return path_a + [mrca] + list(reversed(path_b))  
    
    # ====== Traversal methods ======

    def traverse_inorder( self,
                          include_distances : bool = True) -> Generator[ Union[ int,
                                                                                Tuple[ int,
                                                                                       float] ],
                                                                                None,
                                                                                None ] :
        '''
        Traverse the tree in inorder (left, root, right).
        
        Renamed from in_order().
        
        Args
            include_distances : If True, yield (node_id, distance_to_parent) tuples
                                If False, yield only node_ids
            
        Yields
            Union[ int, Tuple[ int, float ] ] : Node ID or (node_id, distance) tuple
        '''
        current = self.root_node
        stack = []
        
        while True :
            if current != -1 :
                stack.append(current)
                current = self.data[current].left_child
            elif stack :
                current = stack.pop()
                if include_distances :
                    yield ( current, self.data[current].distance )
                else :
                    yield current
                current = self.data[current].right_child
            else :
                break

    def in_order( self, distances=True ) :
        '''
        Generator for traversing the tree in order, yilding tuples
        of node_ids with distances to parent nodes.
        '''
        warn( 'SuchTree.in_order is depricated in favor of SuchTree.traverse_inorder',
              category=DeprecationWarning, stacklevel=2 )
        return self.traverse_inorder 

    def traverse_preorder( self,
                           from_node : Union[ int, str ] = None ) -> Generator[ int, None, None ] :
        '''
        Traverse the tree in preorder (root, left, right).
        
        Renamed from pre_order(), plus new from_node parameter.
        
        Args
            from_node : Starting node (default: root)
            
        Yields
            int : Node IDs in preorder traversal
            
        Raises
            NodeNotFoundError : If from_node leaf name is not found
            InvalidNodeError  : If from_node ID is out of bounds
        '''
        if from_node is None :
            start_node = self.root_node
        else :
            start_node = self._validate_node(from_node)
        
        stack = [start_node]
        
        while stack :
            current = stack.pop()
            right_child = self.data[current].right_child
            left_child  = self.data[current].left_child
            
            if right_child != -1 :
                stack.append(right_child)
            if left_child != -1 :
                stack.append(left_child)
            
            yield current

    def pre_order( self ) :
        '''
        Generator for traversing the tree in pre-order.
        '''
        warn( 'SuchTree.pre_order is depricated in favor of SuchTree.traverse_preorder',
              category=DeprecationWarning, stacklevel=2 )
 
        return self.traverse_preorder

    def traverse_postorder( self,
                            from_node : Union[ int, str ] = None ) -> Generator[ int,
                                                                                 None,
                                                                                 None ] :
        '''
        Traverse the tree in postorder (left, right, root).
        
        Args
            from_node : Starting node (default: root)
            
        Yields
            int : Node IDs in postorder traversal
            
        Raises
            NodeNotFoundError : If from_node leaf name is not found
            InvalidNodeError  : If from_node ID is out of bounds
        '''
        if from_node is None :
            start_node = self.root_node
        else :
            start_node = self._validate_node(from_node)
        
        stack = []
        last_visited = None
        current = start_node

        while stack or current != -1 :
            if current != -1 :
                stack.append(current)
                current = self.data[current].left_child
            else :
                peek_node = stack[-1]
                right_child = self.data[peek_node].right_child
                
                if right_child != -1 and last_visited != right_child :
                    current = right_child
                else :
                    yield peek_node
                    last_visited = stack.pop()
    
    def traverse_levelorder( self,
                             from_node: Union[ int, str ] = None ) -> Generator[ int,
                                                                                 None,
                                                                                 None ] :
        '''
        Traverse the tree level by level (breadth-first).
        
        Args
            from_node : Starting node (default: root)
            
        Yields
            int : Node IDs in level order traversal
            
        Raises
            NodeNotFoundError : If from_node leaf name is not found
            InvalidNodeError  : If from_node ID is out of bounds
        '''
        if from_node is None :
            start_node = self.root_node
        else :
            start_node = self._validate_node(from_node)
        
        queue = deque([start_node])
        
        while queue :
            current = queue.popleft()
            yield current
            
            left_child  = self.data[current].left_child
            right_child = self.data[current].right_child
            
            if left_child != -1 :
                queue.append(left_child)
            if right_child != -1 :
                queue.append(right_child)

    def traverse_leaves_only( self,
                              from_node : Union[ int, str ] = None ) -> Generator[ int,
                                                                                   None,
                                                                                   None ] :
        '''
        Traverse only the leaf nodes.
        
        Args
            from_node : Starting node (default: root)
            
        Yields
            int : Leaf node IDs
            
        Raises
            NodeNotFoundError : If from_node leaf name is not found  
            InvalidNodeError  : If from_node ID is out of bounds
        '''
        if from_node is None :
            start_node = self.root_node
        else :
            start_node = self._validate_node(from_node)
        
        for node_id in self.traverse_preorder(start_node) :
            if self._is_leaf(node_id):
                yield node_id
    
    def traverse_internal_only( self,
                                from_node : Union[ int, str ] = None ) -> Generator[ int,
                                                                                     None,
                                                                                     None ] :
        '''
        Traverse only the internal nodes.
        
        Args
            from_node : Starting node (default: root)
            
        Yields
            int : Internal node IDs
            
        Raises
            NodeNotFoundError : If from_node leaf name is not found
            InvalidNodeError  : If from_node ID is out of bounds
        '''
        if from_node is None :
            start_node = self.root_node
        else :
            start_node = self._validate_node(from_node)
        
        for node_id in self.traverse_preorder(start_node) :
            if not self._is_leaf(node_id) :
                yield node_id

    def traverse_with_depth( self,
                             from_node : Union[ int, str ] = None ) -> Generator[ Tuple[ int,
                                                                                         int ],
                                                                                  None,
                                                                                  None ] :
        '''
        Traverse the tree with depth information.
        
        Args
            from_node : Starting node (default: root)
            
        Yields
            Tuple[int, int] : (node_id, depth) pairs
            
        Raises
            NodeNotFoundError : If from_node leaf name is not found
            InvalidNodeError  : If from_node ID is out of bounds
        '''
        if from_node is None :
            start_node = self.root_node
        else :
            start_node = self._validate_node(from_node)
        
        stack = [(start_node, 0)]  # (node_id, depth)
        
        while stack :
            current, depth = stack.pop()
            yield (current, depth)
            
            right_child = self.data[current].right_child
            left_child = self.data[current].left_child
            
            if right_child != -1 :
                stack.append( ( right_child, depth + 1 ) )
            if left_child != -1 :
                stack.append( ( left_child,  depth + 1 ) )

    def traverse_with_distances( self,
                                 from_node : Union[ int, str ] = None ) -> Generator[ Tuple[ int,
                                                                                             float,
                                                                                             float],
                                                                                      None,
                                                                                      None ] :
        '''
        Traverse the tree with distance information.

        Args
            from_node : Starting node (default: root)
            
        Yields
            Tuple[ int, float, float ] : tuples ( node_id, 
                                                  distance_to_parent, 
                                                  distance_to_root)
            
        Raises
            NodeNotFoundError : If from_node leaf name is not found
            InvalidNodeError  : If from_node ID is out of bounds
        '''
        if from_node is None :
            start_node = self.root_node
        else :
            start_node = self._validate_node(from_node)
        
        stack = [(start_node, 0.0)]  # (node_id, cumulative_distance_to_root)
        
        while stack :
            current, dist_to_root = stack.pop()
            dist_to_parent = self.data[current].distance
            
            yield ( current, dist_to_parent, dist_to_root )
            
            right_child = self.data[current].right_child
            left_child  = self.data[current].left_child
            
            # Add distance to parent unless it's the root (-1 means root)
            next_dist = dist_to_root + (dist_to_parent if dist_to_parent != -1 else 0)
            
            if right_child != -1 :
                stack.append( ( right_child, next_dist ) )
            if left_child != -1 :
                stack.append( ( left_child,  next_dist ) )
    
    # ====== Graph and matrix methods ======
    
    def adjacency_matrix( self,
                          from_node : Union[ int, str ] = None ) -> Dict[ str, Any ] :
        '''
        Build the graph adjacency matrix of the tree or subtree.
        
        Renamed from adjacency().
        
        Args
            from_node : Root node for subtree (default: tree root)
            
        Returns
            Dict[ str, Any ] : Dictionary with keys:
                - 'adjacency_matrix' : np.ndarray of edge weights
                - 'node_ids'         : np.ndarray of corresponding node IDs
                
        Raises
            NodeNotFoundError : If from_node leaf name is not found
            InvalidNodeError  : If from_node ID is out of bounds
        '''
        if from_node is None :
            start_node = self.root_node
        else :
            start_node = self._validate_node(from_node)
        
        # Get all nodes in the subtree
        if self.np_buffer is None :
            self.np_buffer = np.ndarray( self.size, dtype=int )
        
        to_visit = [start_node]
        node_count = 0
        
        for current_id in to_visit :
            self.np_buffer[node_count] = current_id
            node_count += 1
            
            left_child, right_child = self.get_children(current_id)
            if left_child != -1 :
                to_visit.append(left_child)
                to_visit.append(right_child)
        
        node_ids   = np.array( self.np_buffer[ : node_count ] )
        adj_matrix = np.zeros( ( node_count, node_count ), dtype=float )
        
        # Fill adjacency matrix
        for i in range( node_count ) :
            node_id   = node_ids[i]
            parent_id = self.data[node_id].parent
            
            if parent_id == -1 :  # Root node
                continue
                
            distance = self.data[node_id].distance
            if distance == 0 :
                distance += self.polytomy_epsilon
            
            # Find parent index in the node list
            parent_idx = np.where(node_ids == parent_id)[0]
            if len( parent_idx ) > 0 :
                parent_idx = parent_idx[0]
                adj_matrix[i, parent_idx] = distance
                adj_matrix[parent_idx, i] = distance  # Symmetric
        
        return {
            'adjacency_matrix' : adj_matrix,
            'node_ids'         : node_ids
        }

    def adjacency( self, int node=-1 ) :
        '''
        The graph adjacency matrix of the tree. If parameter 
        node is given, return graph adjacency matrix of the
        subtree descendent from node_id.
        '''
        warn( 'SuchTree.adjacency is depricated in favor of SuchTree.adjacency_matrix',
              category=DeprecationWarning, stacklevel=2 )
        return self.adjacency_matrix( from_node=node )

    def laplacian_matrix( self,
                          from_node : Union[ int, str ] = None ) -> Dict[ str, Any ] :
        '''
        Build the graph Laplacian matrix of the tree or subtree.
        
        Renamed from laplacian().
        
        Args
            from_node : Root node for subtree (default: tree root)
            
        Returns
            Dict[ str, Any ] : Dictionary with keys:
                - 'laplacian' : np.ndarray of Laplacian matrix
                - 'node_ids'  : np.ndarray of corresponding node IDs
                
        Raises
            NodeNotFoundError : If from_node leaf name is not found
            InvalidNodeError  : If from_node ID is out of bounds
        '''
        if from_node is None :
            start_node = self.root_node
        else :
            start_node = self._validate_node(from_node)
        
        # Get adjacency matrix first
        adj_result = self.adjacency_matrix(start_node)
        adj_matrix = adj_result['adjacency_matrix']
        node_ids   = adj_result['node_ids']
        
        # Compute Laplacian: L = D - A, where D is degree matrix
        laplacian = np.zeros( adj_matrix.shape, dtype=float )
        np.fill_diagonal( laplacian, adj_matrix.sum( axis=0 ) )
        laplacian = laplacian - adj_matrix
        
        return {
            'laplacian' : laplacian,
            'node_ids'  : node_ids
        }

    def laplacian( self, int node=-1 ) :
        '''
        The graph Laplacian matrix of the tree, or if the parameter
        node is given, return the graph Laplacian matrix of the 
        subtree decendent from node.
        '''
        warn( 'SuchTree.laplacian is depricated in favor of SuchTree.laplacian_matrix',
              category=DeprecationWarning, stacklevel=2 )
        return self.laplacian_matrix( from_node=node )

    def incidence_matrix( self,
                          from_node : Union[ int, str ] = None ) -> Dict[ str, Any ] :
        '''
        Build the incidence matrix of the tree or subtree.
        
        Args
            from_node : Root node for subtree (default: tree root)
            
        Returns
            Dict[ str, Any ] : Dictionary with keys :
                - 'incidence_matrix' : np.ndarray where rows=nodes, cols=edges
                - 'node_ids'         : np.ndarray of node IDs
                - 'edge_list'        : List of (parent, child) tuples
                
        Raises
            NodeNotFoundError : If from_node leaf name is not found
            InvalidNodeError  : If from_node ID is out of bounds
        '''
        if from_node is None :
            start_node = self.root_node
        else :
            start_node = self._validate_node(from_node)
        
        # Get all nodes and edges in subtree
        if self.np_buffer is None:
            self.np_buffer = np.ndarray( self.size, dtype=int )
        
        to_visit = [start_node]
        node_count = 0
        edges = []
        
        for current_id in to_visit :
            self.np_buffer[node_count] = current_id
            node_count += 1
            
            parent_id = self.data[current_id].parent
            if parent_id != -1:  # Not root
                edges.append((parent_id, current_id))
            
            left_child, right_child = self.get_children(current_id)
            if left_child != -1:
                to_visit.append(left_child)
                to_visit.append(right_child)
        
        node_ids = np.array(self.np_buffer[:node_count])
        num_edges = len(edges)
        
        # Build incidence matrix
        incidence = np.zeros((node_count, num_edges), dtype=int)
        
        for edge_idx, (parent_id, child_id) in enumerate(edges) :
            parent_idx = np.where(node_ids == parent_id )[0][0]
            child_idx  = np.where(node_ids == child_id  )[0][0]
            
            incidence[ parent_idx, edge_idx] =  1   # Outgoing edge
            incidence[ child_idx,  edge_idx] = -1   # Incoming edge
        
        return {
            'incidence_matrix' : incidence,
            'node_ids'         : node_ids,
            'edge_list'        : edges
        }

    def distance_matrix( self,
                         nodes: list = None ) -> Dict[ str, Any ] :
        '''
        Build a distance matrix for specified nodes. Wraps pairwise_distances.
        
        Args
            nodes : List of nodes (IDs or names). If None, uses all leaves.
            
        Returns
            Dict[ str, Any ] : Dictionary with keys :
                - 'distance_matrix' : np.ndarray of pairwise distances
                - 'node_ids'        : np.ndarray of corresponding node IDs
                - 'node_names'      : List of node names (if applicable)
                
        Raises
            NodeNotFoundError : If any leaf name is not found
            InvalidNodeError  : If any node ID is out of bounds
        '''
        if nodes is None :
            # Use all leaf nodes
            node_ids   = self.leaf_node_ids
            node_names = [self.leaf_nodes[nid] for nid in node_ids]
        else :
            node_ids   = np.array( [ self._validate_node(node) for node in nodes ] )
            node_names = []
            for node_id in node_ids :
                if self._is_leaf(node_id) :
                    node_names.append( self.leaf_nodes[node_id] )
                else:
                    node_names.append( f'node_{node_id}' )
        
        dist_matrix = self.pairwise_distances( nodes )
        
        return {
            'distance_matrix' : dist_matrix,
            'node_ids'        : node_ids,
            'node_names'      : node_names
        }
    
    def degree_sequence( self,
                         from_node : Union[ int, str ] = None ) -> Dict[ str, Any ] :
        '''
        Compute the degree sequence of the tree.
        
        Args
            from_node : Root node for subtree (default: tree root)
        
        Returns
            Dict[ str, Any ] : Dictionary with keys :
                - 'degrees'    : np.ndarray of node degrees
                - 'node_ids'   : np.ndarray of corresponding node IDs
                - 'max_degree' : Maximum degree
                - 'min_degree' : Minimum degree
                
        Raises
            NodeNotFoundError : If from_node leaf name is not found
            InvalidNodeError  : If from_node ID is out of bounds
        '''
        adj_result = self.adjacency_matrix( from_node )
        adj_matrix = adj_result[ 'adjacency_matrix' ]
        node_ids   = adj_result[ 'node_ids' ]
        
        # Degree is sum of each row (or column, since symmetric)
        degrees = np.sum( adj_matrix > 0, axis=1 )  # Count non-zero entries
        
        return {
            'degrees'    : degrees,
            'node_ids'   : node_ids,
            'max_degree' : degrees.max(),
            'min_degree' : degrees.min()
        }
    
    # ====== SuchLinkedTree methods ======

    def link_leaf( self, unsigned int leaf_id, unsigned int col_id ) :
        '''
        Attaches a leaf node to SuchLinkedTrees link matrix column.
        '''
        if not self.data[leaf_id].left_child == -1 :
            raise Exception( 'Cannot link non-leaf node.', leaf_id )
        if not leaf_id in set( self.leafs.values() ) :
            raise Exception( 'Unknown leaf id.', leaf_id )
        # we only use the left child to identify a node as a leaf, so
        # the right child is avalable to store the column index
        self.data[leaf_id].right_child = col_id
        
    def get_links( self, leaf_ids ) :
        '''
        Returns an array of column ids for an array of leaf ids.
        '''
        if not set( leaf_ids ) <= set( self.leafs.values() ) :
            raise Exception( 'Unknown leaf id(s).', leaf_ids )
        col_ids = np.ndarray( len(leaf_ids), dtype=int )
        for n,leaf in enumerate( leaf_ids ) :
            col_ids[n] = self.data[ leaf ].right_child
        return col_ids
        
    # ====== Export and integration methods ======
 
    def to_networkx_nodes( self,
                           from_node : Union[ int, str ] = None ) -> Generator[ Tuple[ int,
                                                                                       Dict[ str, Any ] ],
                                                                                None,
                                                                                None ] :
        '''
        Generate node data compatible with NetworkX.
        
        Renamed from nodes_data().
        
        Args
            from_node : Root node for subtree (default: tree root)
            
        Yields
            Tuple[ int, Dict[ str, Any ] ] : ( node_id, attributes_dict) pairs
            
        Raises
            NodeNotFoundError : If from_node leaf name is not found
            InvalidNodeError  : If from_node ID is out of bounds
        '''
        if from_node is None :
            start_node = self.root_node
        else :
            start_node = self._validate_node(from_node)
        
        for node_id in self.get_descendants(start_node) :
            attributes = {}
            
            # Add node type
            if self._is_leaf(node_id) :
                attributes['type']  = 'leaf'
                attributes['label'] = self.leaf_nodes[node_id]
            else :
                attributes['type']  = 'internal'
                attributes['label'] = f'node_{node_id}'
            
            # Add support value if available
            support = self.data[node_id].support
            if support != -1 :
                attributes['support'] = support
            
            # Add distance to parent
            distance = self.data[node_id].distance
            if distance != -1 :
                attributes['distance_to_parent'] = distance
            
            # Add distance to root
            attributes['distance_to_root'] = self.distance_to_root(node_id)
            
            # Add depth
            depth = 0
            current = node_id
            while current != self.root_node and self.data[current].parent != -1 :
                current = self.data[current].parent
                depth += 1
            attributes['depth'] = depth
            
            yield (node_id, attributes)
    
    def nodes_data( self ) :
        '''
        Generator for the node data in the tree, compatible with networkx.
        '''
        warn( 'SuchTree.nodes_data is depricated in favor of SuchTree.to_networkx_nodes',
              category=DeprecationWarning, stacklevel=2 )
        return self.to_networkx_nodes()

    def to_networkx_edges( self,
                           from_node: Union[ int, str ] = None ) -> Generator[ Tuple[ int,
                                                                                      int,
                                                                                      Dict[ str,
                                                                                            Any ] ],
                                                                               None,
                                                                               None ] :
        '''
        Generate edge data compatible with NetworkX.
        
        Renamed from edges_data().
        
        Args
            from_node : Root node for subtree (default: tree root)
            
        Yields
            Tuple[ int, int, Dict[ str, Any ] ] : tuples like ( child_id,
                                                                parent_id,
                                                                attributes_dict)
        
        Raises
            NodeNotFoundError : If from_node leaf name is not found
            InvalidNodeError  : If from_node ID is out of bounds
        '''
        if from_node is None :
            start_node = self.root_node
        else :
            start_node = self._validate_node(from_node)
        
        for node_id in self.get_descendants(start_node) :
            parent_id = self.data[node_id].parent
            
            if parent_id == -1 :  # Root node has no parent
                continue
            
            attributes = {
                'weight': self.data[node_id].distance,
                'length': self.data[node_id].distance
            }
            
            # Add support value if this is an edge to an internal node
            if not self._is_leaf(node_id) :
                support = self.data[node_id].support
                if support != -1 :
                    attributes['support'] = support
            
            yield (node_id, parent_id, attributes)

    def to_networkx_graph( self,
                           from_node : Union[ int, str ] = None ) :
        '''
        Create a NetworkX Graph object from the tree.
        
        Args
            from_node : Root node for subtree (default: tree root)
            
        Returns
            networkx.Graph : NetworkX graph representation
            
        Raises
            ImportError       : If NetworkX is not installed
            NodeNotFoundError : If from_node leaf name is not found
            InvalidNodeError  : If from_node ID is out of bounds
        '''
        try :
            import networkx as nx
        except ImportError :
            raise ImportError( 'NetworkX is required for to_networkx_graph()' )
        
        G = nx.Graph()
        
        # Add nodes with attributes
        for node_id, attributes in self.to_networkx_nodes(from_node) :
            G.add_node(node_id, **attributes)
        
        # Add edges with attributes
        for child_id, parent_id, attributes in self.to_networkx_edges(from_node) :
            G.add_edge(child_id, parent_id, **attributes)
        
        return G

    def relationships( self ) -> pd.DataFrame :
        '''
        Return a Pandas DataFrame of describing the relationships among leafs in the tree.
        '''
        pairs        = [ sample([a,b],2) for a,b, in combinations( self.leafs.keys(), 2 ) ]
        distances    = self.distances_by_name( pairs )
        mrca         = [ self.mrca( a, b ) for a,b in pairs ]
        mrca_to_root = [ self.get_distance_to_root(m) for m in mrca ]
        a_to_root    = [ self.get_distance_to_root(a) for a in list( zip( *pairs ) )[0] ]
        b_to_root    = [ self.get_distance_to_root(b) for b in list( zip( *pairs ) )[1] ]
        a_to_mrca    = [ a2r-a2m for a2r,a2m in zip( a_to_root, mrca_to_root ) ]
        b_to_mrca    = [ b2r-b2m for b2r,b2m in zip( b_to_root, mrca_to_root ) ]
        
        return pd.DataFrame( { 'a'            : list( zip( *pairs ) )[0],
                               'b'            : list( zip( *pairs ) )[1],
                               'distance'     : distances,
                               'a_to_root'    : a_to_root,
                               'b_to_root'    : b_to_root,
                               'mrca'         : mrca,
                               'mrca_to_root' : mrca_to_root,
                               'a_to_mrca'    : a_to_mrca,
                               'b_to_mrca'    : b_to_mrca } )
    
    def to_newick( self,
                   from_node : Union[ int, str ] = None,
                                                   include_support   : bool = True, 
                                                   include_distances : bool = True ) -> str :
        '''
        Export tree or subtree to Newick format.
        
        Args
            from_node         : Root node for subtree (default: tree root)
            include_support   : Include support values in output
            include_distances : Include branch lengths in output
            
        Returns
            str : Newick format string
            
        Raises
            NodeNotFoundError : If from_node leaf name is not found
            InvalidNodeError  : If from_node ID is out of bounds
        '''
        if from_node is None :
            start_node = self.root_node
        else :
            start_node = self._validate_node(from_node)
        
        def _node_to_newick( node_id: int ) -> str :
            left_child, right_child = self.get_children(node_id)
            
            if left_child == -1 :  # Leaf node
                result = self.leaf_nodes[node_id]
            else :  # Internal node
                left_newick = _node_to_newick(left_child)
                right_newick = _node_to_newick(right_child)
                result = f'({left_newick},{right_newick})'
                
                # Add support value for internal nodes
                if include_support :
                    support = self.data[node_id].support
                    if support != -1 :
                        result += str(support)
            
            # Add branch length (distance to parent)
            if include_distances and node_id != start_node :  # Don't add distance for root
                distance = self.data[node_id].distance
                if distance != -1 :
                    result += f':{distance}'
            
            return result
        
        return _node_to_newick(start_node) + ';'

    def dump_array( self ) :
        '''
        Print the whole tree. (WARNING : may be huge and useless.)
        '''
        for n in range(self.length) :
            print( 'id : %d ->' % n )
            print( '   distance    : %0.3f' % self.data[n].distance    )
            print( '   parent      : %d'    % self.data[n].parent      )
            print( '   left child  : %d'    % self.data[n].left_child  )
            print( '   right child : %d'    % self.data[n].right_child )
        
   
    def edges_data( self ) :
        '''
        Generator for the edge (i.e. branch) data in the tree, compatible with networkx.
        '''
        for n in range(self.length) :
            # no edges beyond the root node
            if self.data[n].parent == -1 : continue
            yield ( n, self.data[n].parent, { 'weight' : self.data[n].distance } )
        

    # ====== Validation helper functions ======
    
    def _validate_node( self,
                        node : Union[ int, str ] ) -> int :
        '''
        Convert node reference to node ID with validation.
        
        Args
            node : Node ID (int) or leaf name (str)
            
        Returns
            int : Validated node ID
            
        Raises
            NodeNotFoundError : If leaf name is not found
            InvalidNodeError  : If node ID is out of bounds
            TypeError         : If node is not int or str
        '''

        if isinstance( node, str ) :
            if node not in self.leaves :
                raise NodeNotFoundError( node )
            return self.leaves[ node ]
        
        if not isinstance( node, Integral ) :
            raise TypeError( 'Node must be int or str, got {t}'.format( t=str( type(node) ) ) )
        
        node_id = int(node)
        if node_id < 0 or node_id >= self.size :
            raise InvalidNodeError( node_id, self.size )
        
        return node_id
    
    def _validate_node_pair( self,
                             a : Union[ int, str ],
                             b : Union[ int, str ] ) -> tuple[ int, int ] :
        '''
        Validate a pair of nodes and return their IDs.
        
        Args
            a : First node (ID or name)
            b : Second node (ID or name)
            
        Returns
            tuple[ int, int ] : Tuple of validated node IDs
        '''

        return self._validate_node(a), self._validate_node(b)
    
    def _validate_leaf_node( self,
                             node : Union[ int, str ] ) -> int :
        '''
        Validate that a node reference points to a leaf node.
        
        Args
            node: Node ID or leaf name
            
        Returns
            int : Validated leaf node ID
            
        Raises
            NodeNotFoundError : If leaf name is not found
            InvalidNodeError  : If node ID is out of bounds or not a leaf
        '''

        node_id = self._validate_node( node )
        
        if not self._is_leaf( node_id ) :
            raise InvalidNodeError( node_id, 
                                    message='Node {node_id} is not a leaf node'.format( node_id=str(node_id) ) )
        
        return node_id
    
    def _validate_internal_node( self,
                                 node : Union[int, str]) -> int :
        '''
        Validate that a node reference points to an internal node.
        
        Args
            node : Node ID or leaf name
            
        Returns
            int : Validated internal node ID
            
        Raises
            NodeNotFoundError : If leaf name is not found
            InvalidNodeError  : If node ID is out of bounds or not internal
        '''

        node_id = self._validate_node( node )
        
        if self._is_leaf( node_id ) :
            raise InvalidNodeError( node_id,
                                    message='Node {node_id} is not an internal node'.format( node_id=str(node_id) ) )
        
        return node_id
    
    def _convert_to_leaf_names( self,
                                node_ids : list ) -> list[str] :
        '''        
        Convert a list of leaf node IDs to leaf names.
        
        Args
            node_ids : List of node IDs
            
        Returns
            list[str] : List of leaf names
            
        Raises
            InvalidNodeError : If any node ID is not a leaf node
        '''        
        names = []
        for node_id in node_ids :
            if not self._is_leaf( node_id ) :
                raise InvalidNodeError( node_id,
                                        message='Node {node_id} is not a leaf'.format( node_id=str(node_id) ) )
            names.append( self.leaf_nodes[node_id] )
        return names
   
cdef struct Column :
    unsigned int length
    unsigned int leaf_id
    unsigned int* links

cdef class SuchLinkedTrees :
    cdef Column* table
    cdef unsigned int table_size
    
    cdef object TreeA
    cdef object TreeB
    
    cdef object row_ids
    cdef object row_names
    cdef object col_ids
    cdef object col_names
    
    cdef unsigned int n_rows
    cdef unsigned int n_cols
    cdef unsigned int n_links
    
    cdef object np_table
    cdef object np_linklist
    
    cdef object linked_leafsA
    cdef object linked_leafsB
    
    cdef unsigned int subset_a_root
    cdef unsigned int subset_b_root
    cdef object subset_columns
    cdef object subset_rows
    cdef object subset_a_leafs
    cdef object subset_b_leafs
    cdef unsigned int subset_a_size
    cdef unsigned int subset_b_size
    cdef unsigned int subset_n_links
    
    cdef object row_map
    
    cdef uint64_t seed
    cdef uint64_t modulus
    
    def __cinit__( self, tree_a, tree_b, link_matrix ) :
        cdef unsigned int i
        self.table_size = link_matrix.shape[1]
        self.table = <Column*> PyMem_Malloc( self.table_size * sizeof( Column ) )
        for i in xrange( self.table_size ) :
            self.table[i].length = 0
            self.table[i].leaf_id = 0
            self.table[i].links = NULL
        
        # initialize random number generator
        self.seed = np.random.randint( UINT64_MAX >> 1 )
        self.modulus = 2685821657736338717
        
    def __init__( self, tree_a, tree_b, link_matrix ) :
        
        # these objects are constructed only when first accessed
        self.np_table = None
        self.np_linklist = None
        self.linked_leafsA = None
        self.linked_leafsB = None
        self.col_ids = None
        self.row_ids = None
        self.col_names = None
        self.row_names = None
        self.subset_columns = None
        self.subset_a_leafs = None
        self.subset_b_leafs = None
        self.row_map = None
        
        # build trees from newick files, URLs to newick files or
        # from existing SuchTrees
        if isinstance( tree_a, basestring ):
            self.TreeA = SuchTree( tree_a )
        elif type( tree_a ) == SuchTree :
            self.TreeA = tree_a
        else :
            raise Exception( 'unknown input for tree', type(tree_a) )
        
        # build trees from newick files, URLs to newick files or
        # from existing SuchTrees
        if isinstance( tree_b, basestring ):
            self.TreeB = SuchTree( tree_b )
        elif type( tree_b ) == SuchTree :
            self.TreeB = tree_b
        else :
            raise Exception( 'unknown input for tree', type(tree_b) )
        
        # make sure the link matrix connects the trees
        if not link_matrix.shape == ( self.TreeA.n_leafs, self.TreeB.n_leafs ) :
            raise Exception( 'link_matrix shape must match tree leaf counts' )
        
        if not set(link_matrix.axes[0]) == set(self.TreeA.leafs.keys()) :
            raise Exception( 'axis[0] does not match TreeA leaf names' )
        
        if not set(link_matrix.axes[1]) == set(self.TreeB.leafs.keys()) :
            raise Exception( 'axis[1] does not match TreeB leaf names' )
        
        # set row and column indexes
        self.row_ids = np.array( list(self.TreeA.leafs.values()) )
        self.col_ids = np.array( list(self.TreeB.leafs.values()) )
        self.row_names = list(self.TreeA.leafs.keys())
        self.col_names = list(self.TreeB.leafs.keys())
        
        self.n_rows = self.TreeA.n_leafs
        self.n_cols = self.TreeB.n_leafs
        
        # reverse map for row ids
        self.row_map = np.zeros( self.TreeA.length, dtype=int )
        for n,i in enumerate(self.row_ids) :
            self.row_map[i] = n
        
        # populate the link table
        #print id(self), 'allocating columns in', <unsigned int> &self.table
        self.n_links = 0
        for i,(colname,s) in enumerate( link_matrix.T.reindex( self.col_names ).iterrows() ) :
            # attach leaf nodes in TreeB to corresponding column in
            # the link table
            self.TreeB.link_leaf( self.col_ids[i], i )
            l = []
            for rowname, value in s.items() :
                if value > 0 : l.append( self.TreeA.leafs[rowname] )
            col_size = len(l)
            if self.table[i].links == NULL :
                self.table[i].leaf_id = self.col_ids[i]
                self.n_links += col_size
                self.table[i].length = col_size
                self.table[i].links = <unsigned int*> PyMem_Malloc( col_size * sizeof( unsigned int ) )
                for j in xrange( col_size ) :
                    self.table[i].links[j] = l[j]
        
        # by default, the subset is the whole table
        #print 'bulding default subset.'
        self.subset_a_root = self.TreeA.root
        self.subset_b_root = self.TreeB.root
        self.subset_a_size = len( self.row_ids )
        self.subset_b_size = len( self.col_ids )
        self.subset_n_links = self.n_links
        self.subset_rows    = np.array( range( self.subset_a_size ) )
        self.subset_columns = np.array( range( self.subset_b_size ) )
        self.subset_a_leafs = self.row_ids
        self.subset_b_leafs = self.col_ids
        
        # make np_linklist
        #print 'bulding default link list.'
        self.np_linklist = np.ndarray( ( self.n_links, 2 ), dtype=int )
        self._build_linklist()
    
    def __dealloc__( self ) :
        
        #print id(self), 'freeing columns in', <unsigned int> &self.table
        
        for i in xrange( self.table_size ) :
            if not self.table[i].links == NULL :
                PyMem_Free( self.table[i].links )
        
        #print id(self), 'freeing table', <unsigned int> &self.table
        
        PyMem_Free( self.table )
    
    property TreeA :
        'first tree initialized by SuchLinkedTrees( TreeA, TreeB )'
        def __get__( self ) :
            return self.TreeA
    
    property TreeB :
        'second tree initialized by SuchLinkedTrees( TreeA, TreeB )'
        def __get__( self ) :
            return self.TreeB
    
    property n_links :
        'size of the link list'
        def __get__( self ) :
            return self.n_links
    
    property n_cols :
        'Number of columns in the link matrix.'
        def __get__( self ) :
            return self.n_cols
    
    property n_rows :
        'Number of rows in the link matrix.'
        def __get__( self ) :
            return self.n_rows
    
    property col_ids :
        'ids of the columns (TreeB) in the link matrix.'
        def __get__( self ) :
            if self.col_ids is None :
                self.col_ids = self.TreeB.leafs.values()
            return self.col_ids
    
    property row_ids :
        'ids of the rows (TreeA) in the link matrix.'
        def __get__( self ) :
            if self.row_ids is None :
                self.row_ids = self.TreeA.leafs.values()
            return self.row_ids
    
    property col_names :
        'Names of the columns (TreeB) in the link matrix.'
        def __get__( self ) :
            if self.col_names is None :
                self.col_names = self.TreeB.leafs.keys()
            return self.col_names
    
    property row_names :
        'Names of the rows (TreeA) in the link matrix.'
        def __get__( self ) :
            if self.col_ids is None :
                self.row_names = self.TreeA.leafs.keys()
            return self.row_names
    
    property subset_columns :
        'ids of the current subset columns.'
        def __get__( self ) :
            return self.subset_columns
    
    property subset_a_leafs :
        'ids of the current subset rows.'
        def __get__( self ) :
            return self.subset_a_leafs
    
    property subset_b_leafs :
        'ids of the current subset columns.'
        def __get__( self ) :
            return self.subset_b_leafs
    
    property subset_a_size :
        'Number of rows in the current subset.'
        def __get__( self ) :
            return self.subset_a_size
    
    property subset_b_size :
        'Number of columns in the current subset.'
        def __get__( self ) :
            return self.subset_b_size
    
    property subset_a_root :
        'ID of the current subset root in TreeA.'
        def __get__( self ) :
            return self.subset_a_root
    
    property subset_b_root :
        'ID of the current subset root in TreeB.'
        def __get__( self ) :
            return self.subset_b_root
    
    property subset_n_links :
        'Number of links in the current subset.'
        def __get__( self ) :
            return self.subset_n_links
        
    def get_column_leafs( self, col, as_row_ids=False ) :
        
        if isinstance(col, basestring) :
            col_id = self.col_names.index( col )
        else :
            col_id = col
        
        if col_id > self.n_cols :
            raise Exception( 'col_id out of bounds', col_id )
        
        length = self.table[ col_id ].length
        column = np.ndarray( self.table[ col_id ].length, dtype=int )
        for i in xrange( length ) :
            if as_row_ids :
                column[i] = self.row_map[ self.table[ col_id ].links[i] ]
            else :
                column[i] = self.table[ col_id ].links[i]
        
        return column
        
    def get_column_links( self, col ) :
        
        if isinstance(col, basestring) :
            col_id = self.col_names.index( col )
        else :
            col_id = col
        
        if col_id > self.n_cols :
            raise Exception( 'col_id out of bounds', col_id )
        
        length = self.table[ col_id ].length
        column = np.zeros( self.n_rows, dtype=bool )
        for i in xrange( length ) :
            column[ self.row_map[ self.table[ col_id ].links[i] ] ] = True
        
        return column
        
    property linkmatrix :
        'numpy representation of link matrix (generated only on access)'
        def __get__( self ) :
            self._build_linkmatrix()
            return self.np_table
    
    @cython.boundscheck(False)
    cdef _build_linkmatrix( self ) :
        
        ## FIXME : This seems to improperly index when subsetting
        
        cdef unsigned int i
        cdef unsigned int j
        cdef unsigned int l
        cdef unsigned int m
        cdef unsigned int row_id
        
        self.np_table = np.zeros( (self.subset_a_size, self.subset_b_size), dtype=bool )
        
        for col in self.subset_columns :
            for j in xrange( self.table[col].length ) :
                m = self.table[col].links[j]
                for l in self.subset_a_leafs :
                    if l == m :
                        row_id = self.row_map[ m ]
                        self.np_table[ row_id, col ] = True
                        continue
    
    property linklist :
        'numpy representation of link list'
        def __get__( self ) :
            # length will be shorter with subsetted link matrixes
            return self.np_linklist[:self.subset_n_links,:]
            
    @cython.boundscheck(False)
    cdef void _build_linklist( self ) :
        cdef unsigned int i
        cdef unsigned int j
        cdef unsigned int l
        cdef unsigned int m
        cdef unsigned int n
        cdef unsigned int col
        cdef unsigned int k = 0
        
        # Memoryviews into numpy arrays
        cdef long [:] col_ids        = self.col_ids
        cdef long [:] subset_columns = self.subset_columns
        cdef long [:] subset_a_leafs = self.subset_a_leafs
        cdef long [:] subset_b_leafs = self.subset_b_leafs
        cdef long [:,:] np_linklist  = self.np_linklist
        
        for i in xrange( self.subset_b_size ) :
            col = subset_columns[i]
            for j in xrange( self.table[col].length ) :
                m = self.table[col].links[j]
                for l in xrange( self.subset_a_size ) :
                    n = subset_a_leafs[l]
                    if n == m :
                        np_linklist[ k, 0 ] = col_ids[col]
                        np_linklist[ k, 1 ] = m
                        k += 1
                        continue
        
        self.subset_n_links = k
        
    def subset_b( self, node_id ) :
        'subset the link matrix to leafs desended from node_id in TreeB'
        
        if node_id > self.TreeB.length or node_id < 0 :
            raise Exception( 'Node ID out of bounds.', node_id )
        
        self.subset_b_leafs = self.TreeB.get_leafs( node_id )
        self.subset_columns = self.TreeB.get_links( self.subset_b_leafs )
        self.subset_b_size  = len( self.subset_columns )
        self.subset_b_root  = node_id
        self._build_linklist()
        
    def subset_a( self, node_id ) :
        'subset the link matrix to leafs desended from node_id in TreeA'
        
        if node_id > self.TreeA.length or node_id < 0 :
            raise Exception( 'Node ID out of bounds.', node_id )
        
        self.subset_a_leafs = self.TreeA.get_leafs( node_id )
        self.subset_rows    = self.TreeA.get_links( self.subset_a_leafs )
        self.subset_a_size  = len( self.subset_rows )
        self.subset_a_root  = node_id
        self._build_linklist()
    
    @cython.boundscheck(False)
    def linked_distances( self ) :
        '''
        Compute distances for all pairs of links. For large link
        tables, this will fail on memory allocation.
        '''
        cdef unsigned int i
        cdef unsigned int j
        cdef unsigned int k = 0
        cdef unsigned int size = ( self.subset_n_links * (self.subset_n_links-1) ) // 2
        
        ids_a = np.ndarray( ( size, 2 ), dtype=int )
        ids_b = np.ndarray( ( size, 2 ), dtype=int )
        
        cdef long [:,:] linklist = self.np_linklist
        cdef long [:,:] IDs_a = ids_a
        cdef long [:,:] IDs_b = ids_b
        
        with nogil :
            for i in xrange( self.subset_n_links ) :
                for j in xrange( i ) :
                    IDs_a[ k, 1 ] = linklist[ i, 1 ]
                    IDs_a[ k, 0 ] = linklist[ j, 1 ]
                    IDs_b[ k, 1 ] = linklist[ i, 0 ]
                    IDs_b[ k, 0 ] = linklist[ j, 0 ]
                    k += 1
        
        return { 'TreeA'       : self.TreeA.distances( ids_a ),
                 'TreeB'       : self.TreeB.distances( ids_b ),
                 'ids_A'       : ids_a,
                 'ids_B'       : ids_b,
                 'n_pairs'     : size,
                 'n_samples'   : size,
                 'deviation_a' : None,
                 'deviation_b' : None }
        
    @cython.boundscheck(False)
    cdef uint64_t _random_int( self, uint64_t n ) nogil :
        '''
        An implementation of the xorshift64star pseudorandom number
        generator, included here so that we can obtain random numbers
        outside python's Global Interpreter Lock.
        Marsaglia, G. (2003). Xorshift RNGs. Journal of Statistical
        Software, 8(14), 1 - 6.
        http://dx.doi.org/10.18637/jss.v008.i14
        '''
        self.seed ^= self.seed >> 12 # a
        self.seed ^= self.seed << 25 # b
        self.seed ^= self.seed >> 27 # c
        return ( self.seed * self.modulus ) % n
        
    @cython.boundscheck(False)
    def sample_linked_distances( self, float sigma=0.001,
                                       unsigned int buckets=64,
                                       unsigned int n=4096,
                                       unsigned int maxcycles=100 ) :
        
        np_query_a = np.ndarray( ( n, 2 ), dtype=int )
        np_query_b = np.ndarray( ( n, 2 ), dtype=int )
        
        np_distances_a = np.ndarray( ( buckets, n ), dtype=float )
        np_distances_b = np.ndarray( ( buckets, n ), dtype=float )
        
        np_dbuffer_a = np.ndarray( n, dtype=float )
        np_dbuffer_b = np.ndarray( n, dtype=float )
        
        np_sums_a = np.zeros( buckets, dtype=float )
        np_sums_b = np.zeros( buckets, dtype=float )
        
        np_sumsq_a = np.zeros( buckets, dtype=float )
        np_sumsq_b = np.zeros( buckets, dtype=float )
        
        np_samples_a = np.zeros( buckets, dtype=int )
        np_samples_b = np.zeros( buckets, dtype=int )
        
        np_deviations_a = np.ndarray( buckets, dtype=float )
        np_deviations_b = np.ndarray( buckets, dtype=float )
        
        np_all_distances_a = np.ndarray( buckets * n * maxcycles, dtype=float )
        np_all_distances_b = np.ndarray( buckets * n * maxcycles, dtype=float )
        
        cdef long [:,:] query_a = np_query_a
        cdef long [:,:] query_b = np_query_b
        
        cdef double [:,:] distances_a = np_distances_a
        cdef double [:,:] distances_b = np_distances_b
        
        cdef double [:] distances_a_mv = np_dbuffer_a
        cdef double [:] distances_b_mv = np_dbuffer_b
        
        cdef double [:] sums_a = np_sums_a
        cdef double [:] sums_b = np_sums_b
        
        cdef double [:] sumsq_a = np_sumsq_a
        cdef double [:] sumsq_b = np_sumsq_b
        
        cdef long [:] samples_a = np_samples_a
        cdef long [:] samples_b = np_samples_b
        
        cdef double [:] deviations_a = np_deviations_a
        cdef double [:] deviations_b = np_deviations_b
        
        cdef double [:] all_distances_a = np_all_distances_a
        cdef double [:] all_distances_b = np_all_distances_b
        
        cdef long [:,:] linklist = self.np_linklist
        
        cdef int i
        cdef int j
        cdef int l1
        cdef int l2
        cdef int a1
        cdef int a2
        cdef int b1
        cdef int b2
        
        cdef float deviation_a
        cdef float deviation_b
        cdef float sumsq_bucket_a
        cdef float sumsq_bucket_b
        
        cdef unsigned int cycles = 0
        
        while True :
            for i in xrange( buckets ) :
                with nogil :
                    for j in xrange( n ) :
                        l1 = self._random_int( self.subset_n_links )
                        l2 = self._random_int( self.subset_n_links )
                        #l1 = np.random.randint( self.subset_n_links )
                        #l2 = np.random.randint( self.subset_n_links )
                        a1 = linklist[ l1, 1 ]
                        b1 = linklist[ l1, 0 ]
                        a2 = linklist[ l2, 1 ]
                        b2 = linklist[ l2, 0 ]
                        query_a[ j, 0 ] = a1
                        query_a[ j, 1 ] = a2
                        query_b[ j, 0 ] = b1
                        query_b[ j, 1 ] = b2
                distances_a_mv = self.TreeA.distances( query_a )
                distances_b_mv = self.TreeB.distances( query_b )
                distances_a[ i, : ] = distances_a_mv
                distances_b[ i, : ] = distances_b_mv
                all_distances_a[ n * i + cycles * n * buckets : n * i + cycles * n * buckets + n ] = distances_a_mv
                all_distances_b[ n * i + cycles * n * buckets : n * i + cycles * n * buckets + n ] = distances_b_mv
                with nogil :
                    for j in xrange( n ) :
                        sums_a[i]  += distances_a[ i, j ]
                        sums_b[i]  += distances_b[ i, j ]
                        sumsq_a[i] += distances_a[ i, j ]**2
                        sumsq_b[i] += distances_b[ i, j ]**2
                samples_a[i] += n
                samples_b[i] += n
                deviations_a[i] = ( sumsq_a[i] / samples_a[i]
                                - ( sums_a[i]  / samples_a[i] )**2 )**(0.5)
                deviations_b[i] = ( sumsq_b[i] / samples_b[i]
                                - ( sums_b[i]  / samples_b[i] )**2 )**(0.5)
            deviation_a = 0
            deviation_b = 0
            sumsq_bucket_a = 0
            sumsq_bucket_b = 0
            for i in xrange( buckets ) :
                deviation_a += deviations_a[i]
                deviation_b += deviations_b[i]
                sumsq_bucket_a += deviations_a[i]**2
                sumsq_bucket_b += deviations_b[i]**2
            deviation_a = ( sumsq_bucket_a / buckets - ( deviation_a / buckets )**2 )**(0.5)
            deviation_b = ( sumsq_bucket_b / buckets - ( deviation_b / buckets )**2 )**(0.5)
            
            cycles += 1
            
            if deviation_a < sigma and deviation_b < sigma : break
            if cycles >= maxcycles : return None
        
        return { 'TreeA'       : np_all_distances_a[ : n * buckets * cycles ],
                 'TreeB'       : np_all_distances_b[ : n * buckets * cycles ],
                 'n_pairs'     : ( self.subset_n_links * ( self.subset_n_links - 1 ) ) / 2,
                 'n_samples'   : n * buckets * cycles,
                 'deviation_a' : deviation_a,
                 'deviation_b' : deviation_b }
        
    def adjacency( self, deletions=0, additions=0, swaps=0 ) :
        '''
        Build the graph adjacency matrix of the current subsetted
        trees, applying the specified random permutaitons.
        '''
        TA = self.TreeA.adjacency( node = self.subset_a_root )
        TB = self.TreeB.adjacency( node = self.subset_b_root )
        ta_aj = TA['adjacency_matrix']
        tb_aj = TB['adjacency_matrix']
        ta_node_ids = TA['node_ids'].tolist()
        tb_node_ids = TB['node_ids'].tolist()
        
        # apply random permutations
        ll = np.array( self.linklist )
        for i in xrange( 1, deletions ) :
            ll = np.delete( ll, np.random.randint(len(ll)), axis=0 )
        for i in xrange( 1, swaps ) :
            x, y = np.random.choice( xrange( len(ll) ), size=2, replace=False )
            X, Y = ll[x,1], ll[y,1]
            ll[x,1] = Y
            ll[y,1] = X
        for i in xrange( 1, additions ) :
            a = np.random.choice( self.TreeA.leafs.values() )
            b = np.random.choice( self.TreeB.leafs.values() )
            ll = np.concatenate( (ll, np.array([[b,a]])), axis=0 )
        
        # map node ids to matrix coordinates
        ta_links = map( lambda x : ta_node_ids.index(x), ll[:,1] )
        tb_links = map( lambda x : tb_node_ids.index(x) + ta_aj.shape[0], ll[:,0] )
        
        # build empty graph adjacency matrix
        aj = np.zeros( ( ta_aj.shape[0] + tb_aj.shape[0],
                         ta_aj.shape[1] + tb_aj.shape[1] ) )
        
        # place the tree adjacency matrixes into the empty graph matrix
        aj[ 0:ta_aj.shape[0] , 0:ta_aj.shape[1]  ] = ta_aj / ta_aj.max()
        aj[   ta_aj.shape[0]:,   ta_aj.shape[1]: ] = tb_aj / tb_aj.max()
        
        # compute means of all the non-zero-length edges of the trees
        ta_mean = np.mean( ta_aj.flatten()[ ta_aj.flatten() > self.TreeA.polytomy_distance ] )
        tb_mean = np.mean( tb_aj.flatten()[ tb_aj.flatten() > self.TreeB.polytomy_distance ] )
        link_mean = ( ta_mean / ta_aj.max() + tb_mean / tb_aj.max() ) / 2.0
        
        # place the link edges into graph adjacency matrix,
        # normalizing their edge weights to the average weight of the
        # tree edges
        for i,j in zip( tb_links, ta_links ) :
            aj[i,j] = link_mean
            aj[j,i] = link_mean
        
        return aj
        
    def laplacian( self, deletions=0, additions=0, swaps=0 ) :
        '''
        The graph Laplacian matrix of the current subsetted trees.
        '''
        
        aj = self.adjacency( deletions=deletions,
                             additions=additions,
                             swaps=swaps )
        lp = np.zeros( aj.shape )
        np.fill_diagonal( lp, aj.sum( axis=0 ) )
        lp = lp - aj
        
        return lp
        
    def spectrum( self, deletions=0, additions=0, swaps=0 ) :
        '''
        The eigenvalues of the graph Laplacian matrix of the current
        subsetted trees.
        '''
        lp = self.laplacian( deletions, additions, swaps )
        
        cdef int N     = lp.shape[0]
        cdef int nb    = 4
        cdef int lwork = (nb+2)*N
        
        np_work = np.ndarray( lwork )
        np_w    = np.ndarray( N )
        
        cdef double[:,::1] a = lp
        cdef double[:] work  = np_work
        cdef double[:] w     = np_w
        
        cdef double * b = &a[0,0]
        cdef int info   = 0
        
        dsyev( 'N', 'U', &N, b, &N, &w[0], &work[0], &lwork, &info )
        
        if info == 0 :
            return np_w
        else :
            return info
    
    def to_igraph( self, deletions=0, additions=0, swaps=0 ) :
        '''
        Return the current SuchLinkedTrees subgraph as a weighted,
        labled igraph object. The igraph package must be installed.
        '''
        if not with_igraph :
            raise Exception( 'igraph package not installed.' )
        
        g = Graph.Weighted_Adjacency( self.adjacency( deletions=deletions, 
                                                      additions=additions, 
                                                      swaps=swaps ).tolist(),
                                      mode=ADJ_UNDIRECTED )
        
        subset_a_length = len( list( self.TreeA.get_descendant_nodes( self.subset_a_root ) ) )
        subset_b_length = len( list( self.TreeB.get_descendant_nodes( self.subset_b_root ) ) )
        
        g.vs['color'] = ['#e1e329ff'] * subset_a_length + \
                        ['#24878dff'] * subset_b_length
        g.vs['label'] = [ 'h' + str(i) for i in range( subset_a_length ) ] + \
                        [ 'g' + str(i) for i in range( subset_b_length ) ]
        g.vs['tree'] = [ 0 ]  * subset_a_length + [ 1 ] * subset_b_length
        
        return g
 

    def dump_table( self ) :
        'Print the link matrix (WARNING : may be huge and useless)'
        for i in xrange( self.n_cols ) :
            col = []
            for j in xrange( self.table[i].length ) :
                #row_id = np.where( self.row_ids == self.table[i].links[j] )[0][0]
                row_id = self.table[i].links[j]
                col.append( row_id )
            print( 'column', i, ':', ','.join( map( str, col ) ) )




