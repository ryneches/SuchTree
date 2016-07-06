import cython
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from dendropy import Tree
import numpy as np
cimport numpy as np
import pandas as pd

cdef extern from 'stdint.h' :
    ctypedef unsigned long int uint64_t
    uint64_t UINT64_MAX

# Trees are built from arrays of Node structs. 'parent', 'left_child'
# and 'right_child' attributes represent integer offsets within the
# array that specify other Node structs.
cdef struct Node :
    int parent
    int left_child
    int right_child
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
    cdef int fail_id_a
    cdef int fail_id_b
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
                fail_id_a = a
                fail_id_b = b
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
        raise Exception( 'query contains out of bounds id', (fail_id_a, fail_id_b) )

@cython.no_gc_clear
cdef class SuchTree :
    """
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
    """

    cdef Node* data
    cdef unsigned int length
    cdef unsigned int depth
    cdef unsigned int n_leafs
    cdef unsigned int root
    cdef object leafs   
    cdef object np_buffer   
    
    def __init__( self, tree_file ) :
        """
        SuchTree constructor.
        """
        cdef unsigned int n
        cdef int id
        self.np_buffer = None
        self.n_leafs = 0
        
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
                self.n_leafs += 1
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
        'The number of nodes in the tree.'
        def __get__( self ) :
            return self.length
    
    property depth :
        'The maximum depth of the tree.'
        def __get__( self ) :
            return self.depth
    
    property n_leafs :
        'The number of leafs in the tree.'
        def __get__( self ) :
            return self.n_leafs
    
    property leafs :
        'A dictionary mapping leaf names to leaf node ids.'
        def __get__( self ) :
            return self.leafs
    
    property root :
        'The id of the root node.'
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
        Return an array of ids of all leaf nodes descendent from a given node.
        """
        cdef unsigned int i
        cdef int l
        cdef int r
        cdef unsigned int n = 0
        if self.np_buffer is None :
            self.np_buffer = np.ndarray( self.n_leafs, dtype=int )
        to_visit = [id]
        for i in to_visit :
            l,r = self.get_children( i )
            if l == -1 :
                self.np_buffer[n] = i
                n += 1
            else :
                to_visit.append( l )
                to_visit.append( r )   
        return np.array(self.np_buffer[:n])

    def get_internal_nodes( self ) :
        """
        Return an array of the ids of all internal nodes.
        """
        cdef unsigned int i
        cdef int l
        cdef int r
        cdef unsigned int n = 0
        if self.np_buffer is None :
            self.np_buffer = np.ndarray( self.n_leafs, dtype=int ) 
        to_visit = [self.root]
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
    
    def link_leaf( self, unsigned int leaf_id, unsigned int col_id ) :
        """
        Attaches a leaf node to SuchLinkedTrees link matrix column.
        """
        if not self.data[leaf_id].left_child == -1 :
            raise Exception( 'Cannot link non-leaf node.', leaf_id )
        if not leaf_id in set( self.leafs.values() ) :
            raise Exception( 'Unknown leaf id.', leaf_id )
        # we only use the left child to identify a node as a leaf, so
        # the right child is avalable to store the column index
        self.data[leaf_id].right_child = col_id
    
    def get_links( self, leaf_ids ) :
        """
        Returns an array of column ids for an array of leaf ids.
        """
        if not set( leaf_ids ) <= set( self.leafs.values() ) :
            raise Exception( 'Unknown leaf id(s).', leaf_ids )
        col_ids = np.ndarray( len(leaf_ids), dtype=int )
        for n,leaf in enumerate( leaf_ids ) :
            col_ids[n] = self.data[ leaf ].right_child
        return col_ids
    
    def dump_array( self ) :
        """
        Print the whole tree. (WARNING : may be huge and useless.)
        """
        for n in range(self.length) :
            print 'id : %d ->' % n
            print '   distance    : %0.3f' % self.data[n].distance
            print '   parent      : %d'    % self.data[n].parent
            print '   left child  : %d'    % self.data[n].left_child
            print '   right child : %d'    % self.data[n].right_child
    
    def __dealloc__( self ) :
        PyMem_Free( self.data )     # no-op if self.data is NULL


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
    
    cdef object subset_columns
    cdef object subset_leafs
    cdef unsigned int subset_size
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
        self.subset_leafs = None
        self.row_map = None        
        
        # build trees from newick files, URLs to newick files or 
        # from existing SuchTrees
        if type( tree_a ) == str : 
            self.TreeA = SuchTree( tree_a )
        elif type( tree_a ) == SuchTree :
            self.TreeA = tree_a
        else :
            raise Exception( 'unknown input for tree', type(tree_a) )
        
        # build trees from newick files, URLs to newick files or 
        # from existing SuchTrees
        if type( tree_b ) == str : 
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
        self.row_ids = np.array( self.TreeA.leafs.values() )
        self.col_ids = np.array( self.TreeB.leafs.values() )
        self.row_names = self.TreeA.leafs.keys()
        self.col_names = self.TreeB.leafs.keys()
        
        self.n_rows = self.TreeA.n_leafs
        self.n_cols = self.TreeB.n_leafs
        
        # reverse map for row ids
        self.row_map = np.zeros( self.TreeA.length, dtype=int )
        for n,i in enumerate(self.row_ids) :
            self.row_map[i] = n
        
        print id(self), 'allocating columns in', <unsigned int> &self.table
        self.n_links = 0
        for i in xrange( self.table_size ) :
        #for i,s in enumerate( link_matrix.iteritems() ) :
            self.TreeB.link_leaf( self.col_ids[i], i )
            s = link_matrix[ link_matrix.columns[i] ]
            l = map( lambda x : self.TreeA.leafs[x], s[ s > 0 ].to_dict().keys() )
            col_size = len(l)
            if self.table[i].links == NULL :
                self.table[i].leaf_id = self.col_ids[i]
                self.n_links += col_size
                self.table[i].length = col_size
                self.table[i].links = <unsigned int*> PyMem_Malloc( col_size * sizeof( unsigned int ) )
                for j in xrange( col_size ) :
                    self.table[i].links[j] = l[j]
         
        # by default, the subset is the whole table
        print 'bulding default subset.'
        self.subset_size = len( self.col_ids )
        self.subset_n_links = self.n_links
        self.subset_columns = np.array( range( self.subset_size ) )
        self.subset_leafs = self.col_ids
        
        # make np_linklist
        print 'bulding default link list.'
        self.np_linklist = np.ndarray( ( self.n_links, 2 ), dtype=int )
        self._build_linklist()
    
    def __dealloc__( self ) :
        
        print id(self), 'freeing columns in', <unsigned int> &self.table
        
        for i in xrange( self.table_size ) :
            if not self.table[i].links == NULL :
                PyMem_Free( self.table[i].links ) 
        
        print id(self), 'freeing table', <unsigned int> &self.table
        
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
    
    property subset_leafs :
        'ids of the current subset columns.'
        def __get__( self ) :
            return self.subset_leafs
    
    property subset_size :
        'Number of columns in the current subset.'
        def __get__( self ) :
            return self.subset_size
        
    property subset_n_links :
        'Number of columns in the current subset.'
        def __get__( self ) :
            return self.subset_n_links
    
    def get_column_leafs( self, col, as_row_ids=False ) :
        
        if type(col) is str :
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
        
        if type(col) is str :
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
    
    cdef _build_linkmatrix( self ) :
        cdef unsigned int i
        cdef unsigned int j
        cdef unsigned int row_id
        
        self.np_table = np.zeros( (self.n_rows, self.subset_size), dtype=bool )
        
        for i in xrange( self.subset_size ) :
            col = self.subset_columns[i] 
            for j in xrange( self.table[col].length ) :
                row_id = self.row_map[ self.table[col].links[j] ]
                self.np_table[ col, row_id ] = True
        
    property linklist :
        'numpy representation of link list (generated only on access)'
        def __get__( self ) :
            # actual length will be shorter when with subsetted link matrixes
            return self.np_linklist[:self.subset_n_links-1,:]
    
    @cython.boundscheck(False)
    cdef void _build_linklist( self ) nogil :
            cdef unsigned int i
            cdef unsigned int j
            cdef unsigned int col
            cdef unsigned int k = 0
           
            # Memoryviews into numpy arrays
            cdef long [:] subset_columns = self.subset_columns
            cdef long [:] subset_leafs   = self.subset_leafs
            cdef long [:,:] np_linklist  = self.np_linklist
            
            for i in xrange( self.subset_size ) :
                col = subset_columns[i]
                for j in xrange( self.table[col].length ) :
                    np_linklist[ k, 0 ] = subset_leafs[i]
                    np_linklist[ k, 1 ] = self.table[col].links[j]
                    k += 1
            
            self.subset_n_links = k
            
    def subset( self, node_id ) :
        'subset the link matrix to leafs desended from node_id'
        self.subset_leafs = self.TreeB.get_leafs( node_id )
        self.subset_columns = self.TreeB.get_links( self.subset_leafs )
        self.subset_size = len( self.subset_columns )
        self._build_linklist()
    
    @cython.boundscheck(False)
    def linked_distances( self ) :
        """
        Compute distances for all pairs of links. For large link
        tables, this will fail on memory allocation.
        """
        cdef unsigned int i
        cdef unsigned int j
        cdef unsigned int k = 0
        cdef unsigned int size = ( self.subset_n_links * (self.subset_n_links-1) ) / 2 
        
        ids_a = np.ndarray( ( size, 2 ), dtype=int )
        ids_b = np.ndarray( ( size, 2 ), dtype=int )
        
        cdef long [:,:] linklist = self.np_linklist
        cdef long [:,:] IDs_a = ids_a
        cdef long [:,:] IDs_b = ids_b
         
        with nogil :
            for i in xrange( self.subset_n_links ) :
                for j in xrange( i ) :
                    IDs_a[ k, 0 ] = linklist[ i, 1 ]
                    IDs_a[ k, 1 ] = linklist[ j, 1 ]
                    IDs_b[ k, 0 ] = linklist[ i, 0 ]
                    IDs_b[ k, 1 ] = linklist[ j, 0 ]
                    k += 1
        
        return { 'TreeA' : self.TreeA.distances( ids_a ), 
                 'TreeB' : self.TreeB.distances( ids_b ) }
    
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
        
        cdef double [:] distances_a_mv
        cdef double [:] distances_b_mv
         
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
            cycles += 1
            for i in xrange( buckets ) :
                with nogil :
                    for j in xrange( n ) :
                        l1 = self._random_int( self.subset_n_links )
                        l2 = self._random_int( self.subset_n_links )
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
                all_distances_a[ n * i * cycles : n * i * cycles + n ] = distances_a_mv
                all_distances_b[ n * i * cycles : n * i * cycles + n ] = distances_b_mv
                with nogil :
                    for j in xrange( n ) :
                        sums_a[i] += distances_a[ i, j ]
                        sums_b[i] += distances_b[ i, j ]
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
            
            print deviation_a, deviation_b
            
            if deviation_a < sigma and deviation_b < sigma : break
            if cycles >= maxcycles : return None
 
        return { 'r' : _pearson( all_distances_a[ : n * buckets * cycles ], 
                                 all_distances_b[ : n * buckets * cycles ], n*cycles ),
                 'n' : n * buckets * cycles }
 
    def dump_table( self ) :
        'Print the link matrix (WARNING : may be huge and useless)'
        for i in xrange( self.n_cols ) :
            col = []
            for j in xrange( self.table[i].length ) :
                #row_id = np.where( self.row_ids == self.table[i].links[j] )[0][0]
                row_id = self.table[i].links[j]
                col.append( row_id )
            print 'column', i, ':', ','.join( map( str, col ) )
