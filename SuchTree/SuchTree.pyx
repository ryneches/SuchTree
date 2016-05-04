import cython
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from dendropy import Tree
import numpy as np
cimport numpy as np

cdef struct Node :
    int parent
    int left_child
    int right_child
    float distance

cdef float _get_distance_to_root( Node* data, id ) :
    cdef float d = 0.0
    cdef float d_i = 0.0
    cdef int i = id
    while True :
        d_i = data[i].distance
        if d_i == -1 : break
        d = d + d_i
        i = data[i].parent
    return d

@cython.boundscheck(False)
cdef void _distances( Node* data, long[:,:] ids, double[:] result ) :
    cdef int a
    cdef int b
    cdef int i
    cdef float d_a = 0.0
    cdef float d_b = 0.0
    cdef float d_i = 0.0
    
    with nogil :
           
        for i in range( ids.shape[0] ) :
            a = ids[i][0]
            b = ids[i][1]
            d_a = 0.0
            d_b = 0.0
            while True :
                d_i = data[a].distance
                if d_i == -1 : break
                d_a += d_i
                a = data[a].parent
            while True :
                d_i = data[b].distance
                if d_i == -1 : break
                d_b += d_i
                b = data[b].parent
            if d_a > d_b :
                result[i] = d_a - d_b
            else :
                result[i] = d_b - d_a

cdef class SuchTree :

    cdef Node* data
    cdef int length
    cdef object leafs
    
    def __init__( self, tree_file ) :
        tree = Tree.get( file=open(tree_file), 
                         schema='newick',
                         preserve_underscores=True )
        tree.resolve_polytomies()
        size = len( tree.nodes() )
        # allocate some memory
        self.data = <Node*> PyMem_Malloc( size * sizeof(Node) )
        self.length = size
        if not self.data :
            raise MemoryError()
        
        self.leafs = {}
        for id,node in enumerate( tree.inorder_node_iter() ) :
            node.label = id
            if node.taxon :
                self.leafs[ node.taxon.label ] = id
                
        for id,node in enumerate( tree.inorder_node_iter() ) :
            if not node.parent_node :
                distance = -1
                parent   = -1
            else :
                distance = node.edge_length
                parent   = node.parent_node.label
            if node.taxon :
                left_child, right_child = -1, -1
            else :
                l_child, r_child = node.child_nodes()
                left_child  = l_child.label
                right_child = r_child.label
            
            self.data[id].parent      = parent
            self.data[id].left_child  = left_child
            self.data[id].right_child = right_child
            self.data[id].distance    = distance
   
    property length :
        def __get__( self ) :
            return self.length

    property leafs :
        def __get__( self ) :
            return self.leafs
    
    def get_children( self, id ) :
        return ( self.data[id].left_child, self.data[id].right_child )

    def get_distance_to_root( self, id ) :
        return _get_distance_to_root( self.data, id )
        
    def distance( self, a, b ) :
        cdef float d_a = self.get_distance_to_root( a )
        cdef float d_b = self.get_distance_to_root( b )
        return abs( d_a - d_b ) 

    def distances( self, long[:,:] ids ) :
        if not ids.shape[1] == 2 : 
            raise Exception( 'expected (n,2) array', 
                             ids.shape[0], ids.shape[1] )
        
        result = np.zeros( ids.shape[0], dtype=float )
        _distances( self.data, ids, result )
        return result
         
    def distances_by_name( self, id_pairs ) :
        shape = ( len(id_pairs), len(id_pairs[0]) )
        ids = np.zeros( shape, dtype=float )
        for n,(a,b) in enumerate(id_pairs) :
            ids[n][0] = self.leafs[a]
            ids[n][1] = self.leafs[b]
        return self.distances( ids )

    def dump_array( self ) :
        for n in range(self.length) :
            print 'id : %d ->' % n
            print '   distance    : %0.3f' % self.data[n].distance
            print '   parent      : %d'    % self.data[n].parent
            print '   left child  : %d'    % self.data[n].left_child
            print '   right child : %d'    % self.data[n].right_child
    
    def __dealloc__(self):
        PyMem_Free(self.data)     # no-op if self.data is NULL
