import cython
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from dendropy import Tree
import numpy as np
cimport numpy as np

# Trees are built from arrays of Node structs. 'parent', 'left_child'
# and 'right_child' attributes represent integer offsets within the
# array that specify other Node structs.
cdef struct Node :
    int parent
    int left_child
    int right_child
    float distance

cdef float _get_distance_to_root( Node* data, id ) :
    """
    Calculate the distance from a node of a given id to the root node.
    Will work for both leaf and internal nodes. Private cdef method.
    """
    cdef float d = 0.0
    cdef float d_i = 0.0
    cdef int i = id
    cdef int a_depth = 0
    cdef int mrca = -1
    
    while True :
        d_i = data[i].distance
        if d_i == -1 : break
        d = d + d_i
        i = data[i].parent
    return d

cdef int _mrca( Node* data, int depth, int a, int b ) :
    cdef int n
    cdef int i
    cdef int mrca = -1
    cdef int a_depth
    # allocate some memory for visited node array
    visited = <int*> PyMem_Malloc( depth * sizeof(int) )
    
    if visited == NULL :
        raise Exception( '_mrca could not allocate memmory' )
     
    with nogil :
        n = a
        i = 0
        while True :
            visited[i] = n
            n = data[n].parent
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
            n = data[n].parent
            if n == -1 :
                mrca = n
                break
    
    # free the visited node array (no-op if NULL)
    PyMem_Free(visited)
    return mrca

cdef float _distance( Node* data, int depth, int a, int b ) :
    cdef int mrca
    cdef float d = 0
    cdef int n
    
    mrca = _mrca( data, depth, a, b )
   
    n = a
    while n != mrca :
        d += data[n].distance
        n = data[n].parent
    n = b
    while n != mrca :
        d += data[n].distance
        n = data[n].parent
    return d

@cython.boundscheck(False)
cdef void _distances( Node* data, int length, int depth, long[:,:] ids, double[:] result ) :
    """
    For each pair of node ids in the given (n,2) array, calculate the
    distance to the root node for each pair and store their differece
    in the given (1,n) result array. Calculations are performed within
    a 'nogil' context, allowing the interpreter to perform other tasks
    concurrently if desired. Private cdef method.
    """
    cdef int a
    cdef int b
    cdef int j
    cdef int n
    cdef int i
    cdef int mrca
    cdef int a_depth
    cdef float d = 0
    cdef bint fail = False
    # allocate some memory for visited node array
    visited = <int*> PyMem_Malloc( depth * sizeof(int) )
    
    if visited == NULL :
        raise Exception( '_distances could not allocate memory' )
    
    with nogil : 
        for j in range( ids.shape[0] ) :
            d = 0
            a = ids[j][0]
            b = ids[j][1]
            if a >= length or b >= length :
                fail = True
                break
            if a == b :
                result[j] = 0.0
                continue
            n = a
            i = 0
            while True :
                if n == -1 : break
                visited[i] = n
                n = data[n].parent
                i += 1
            a_depth = i
            
            mrca = -1
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
                d += data[n].distance
                n = data[n].parent
                if n == -1 :
                    mrca = n
                    break
            
            n = b
            while n != mrca :
                d += data[n].distance
                n = data[n].parent
            
            result[j] = d
     
    # free the visited node array (no-op if NULL)
    PyMem_Free(visited)
    
    if fail :
        raise Exception( 'query contains out of bounds id' )

cdef class SuchTree :

    cdef Node* data
    cdef int length
    cdef int depth
    cdef int root
    cdef object leafs   
    
    def __init__( self, tree_file ) :
        """
        Initialize a new SuchTree extention type. The constructor
        accepts a filesystem path or URL to a file that describes
        the tree in NEWICK format. For now, SuchTree uses dendropy
        to parse the NEWICK file.
        
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
        """
        cdef int n
        cdef int id
        
        url_strings = [ 'http://', 'https://', 'ftp://' ]
        
        if filter( lambda x : tree_file.startswith(x), url_strings ) :
            t = Tree.get( url=tree_file,
                          schema='newick',
                          preserve_underscores=True,
                          suppress_internal_node_taxa=True )
        else :
            t = Tree.get( file=open(tree_file),
                          schema='newick',
                          preserve_underscores=True,
                          suppress_internal_node_taxa=True )
        t.resolve_polytomies()
        size = len( t.nodes() )
        # allocate some memory
        self.data = <Node*> PyMem_Malloc( size * sizeof(Node) )
        if self.data == NULL :
            raise Exception( 'SuchTree could not allocate memory' )
        
        self.length = size
        if not self.data :
            raise MemoryError()
        
        self.leafs = {}
        for id,node in enumerate( t.inorder_node_iter() ) :
            node.label = id
            if id >= size :
                raise Exception( 'node label out of bounds : ' + str(id) ) 
            if node.taxon :
                self.leafs[ node.taxon.label ] = id
                
        for id,node in enumerate( t.inorder_node_iter() ) :
            if not node.parent_node :
                distance = -1
                parent   = -1
                self.root = id
            else :
                distance = node.edge_length
                parent   = node.parent_node.label
            if node.taxon :
                left_child, right_child = -1, -1
            else :
                l_child, r_child = node.child_nodes()
                left_child  = l_child.label
                right_child = r_child.label
            
            if id >= size :
                raise Exception( 'node label out of bounds : ' + str(id) ) 
        
            self.data[id].parent      = parent
            self.data[id].left_child  = left_child
            self.data[id].right_child = right_child
            self.data[id].distance    = distance
            
        for id in self.leafs.values() :
            n = 1
            while True :
                if self.data[id].parent == -1 : break
                id = self.data[id].parent
                n += 1
            if n > self.depth :
                self.depth = n
                
    property length :
        def __get__( self ) :
            return self.length
    
    property depth :
        def __get__( self ) :
            return self.depth
    
    property leafs :
        def __get__( self ) :
            return self.leafs
    
    property root :
        def __get__( self ) :
            return self.root
    
    def get_parent( self, id ) :
        """
        Return the id of the parent of a given node. Will accept node
        id or leaf name.
        """
        if type(id) is str :
            try :
                id = self.leafs[ id ]
            except KeyError :
                raise Exception( 'Leaf name not found : ' + id )
        return self.data[id].parent
    
    def get_children( self, id ) :
        """
        Return the ids of child nodes of given node. Will accept node
        id or a leaf name.
        """
        if type(id) is str :
            try :
                id = self.leafs[ id ]
            except KeyError :
                raise Exception( 'Leaf name not found : ' + id )
        return ( self.data[id].left_child, self.data[id].right_child )

    def get_leafs( self, id ) :
        """
        Return the ids of all leaf nodes descendent from a given node.
        """
        to_visit = [id]
        for i in to_visit :
            l,r = self.get_children( i )
            if l == -1 :
                yield i
            else :
                to_visit.append( l )
                to_visit.append( r )   
 
    def get_distance_to_root( self, id ) :
        """
        Return distance to root for a given node. Will accept node id
        or a leaf name.
        """
        if type(id) is str :
            try :
                id = self.leafs[ id ]
            except KeyError :
                raise Exception( 'Leaf name not found : ' + id )
        return _get_distance_to_root( self.data, id )
    
    def mrca( self, a, b ) :
        """
        Return the id of the most recent common ancestor of two nodes
        if given ids.
        """
        return _mrca( self.data, self.depth, a, b )
        
    def distance( self, a, b ) :
        """
        Return distnace between a pair of nodes. Will accelt node ids
        or leaf names.
        """
        if type(a) is str :
            try :
                a = self.leafs[a]
            except KeyError :
                raise Exception( 'Leaf name not found : ' + a )
        if type(b) is str :
            try :
                b = self.leafs[b]
            except KeyError :
                raise Exception( 'Leaf name not found : ' + b )
        return _distance( self.data, self.depth, a, b ) 

    def distances( self, long[:,:] ids ) :
        """
        Returns an array of distances between pairs of node ids in a
        given (n,2) array. Accepts only node ids.
        """
        if not ids.shape[1] == 2 : 
            raise Exception( 'expected (n,2) array', 
                             ids.shape[0], ids.shape[1] )
        
        result = np.zeros( ids.shape[0], dtype=float )
        _distances( self.data, self.length, self.depth, ids, result )
        return result
         
    def distances_by_name( self, id_pairs ) :
        """
        Returns an array of distances between pairs of leaf names in a
        given (n,2) list of lists. Accepts only leaf names.
        """
        shape = ( len(id_pairs), len(id_pairs[0]) )
        ids = np.zeros( shape, dtype=int )
        for n,(a,b) in enumerate(id_pairs) :
            ids[n][0] = self.leafs[a]
            ids[n][1] = self.leafs[b]
        return self.distances( ids )

    def dump_array( self ) :
        """
        Print the whole array. WARNING : may be huge and useless.
        """
        for n in range(self.length) :
            print 'id : %d ->' % n
            print '   distance    : %0.3f' % self.data[n].distance
            print '   parent      : %d'    % self.data[n].parent
            print '   left child  : %d'    % self.data[n].left_child
            print '   right child : %d'    % self.data[n].right_child
    
    def __dealloc__(self):
        PyMem_Free(self.data)     # no-op if self.data is NULL
