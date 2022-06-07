#cython: language_level=3
import cython
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from dendropy import Tree
import numpy as np
cimport numpy as np
import pandas as pd
from scipy.linalg.cython_lapack cimport dsyev

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
    cdef np.float64_t epsilon
    cdef object leafs
    cdef object leafnodes
    cdef object np_buffer
    
    def __init__( self, tree_file ) :
        """
        SuchTree constructor.
        """
        cdef unsigned int n
        cdef int node_id
        self.np_buffer = None
        self.n_leafs = 0
        
        # tiny nonzero distance for representing polytomies
        self.epsilon = np.finfo( np.float64 ).eps
        
        url_strings = [ 'http://', 'https://', 'ftp://' ]

        if any( [ tree_file.startswith(x) for x in url_strings ] ) :
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
        self.leafnodes = {}
        for node_id,node in enumerate( t.inorder_node_iter() ) :
            node.label = node_id
            if node_id >= size :
                raise Exception( 'node label out of bounds : ' + str(node_id) )
            if node.taxon :
                self.leafs[ node.taxon.label ] = node_id
                self.leafnodes[ node_id ] = node.taxon.label

        for node_id,node in enumerate( t.inorder_node_iter() ) :
            if not node.parent_node :
                distance = -1.0
                parent   = -1
                self.root = node_id
            else :
                if not node.edge_length :
                    distance = 0.0
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
            
            if node_id >= size :
                raise Exception( 'node label out of bounds : ' + str(node_id) )
            
            self.data[node_id].parent      = parent
            self.data[node_id].left_child  = left_child
            self.data[node_id].right_child = right_child
            self.data[node_id].distance    = distance
            
        for node_id in self.leafs.values() :
            n = 1
            while True :
                if self.data[node_id].parent == -1 : break
                node_id = self.data[node_id].parent
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
    
    property leafnodes :
        'A dictionary mapping leaf node ids to leaf names.'
        def __get__( self ) :
            return self.leafnodes

    property root :
        'The id of the root node.'
        def __get__( self ) :
            return self.root
            
    property polytomy_distance :
        'Tiny, nonzero distance for polytomies in the adjacency matrix.'
        def __get__( self ) :
            return self.epsilon
        def __set__( self, np.float64_t new_epsilon ) :
            self.epsilon = new_epsilon
            
    def get_parent( self, query ) :
        """
        Return the id of the parent of a given node. Will accept node
        id or leaf name.
        """
        if type(query) is str :
            try :
                node_id = self.leafs[ query ]
            except KeyError :
                raise Exception( 'leaf name not found : ' + query )
        else :
            node_id = int( query )
        if node_id < 0 or node_id >= self.length :
            raise Exception( 'node id out of bounds : ', node_id )
            
        return self.data[node_id].parent
        
    def get_children( self, node_id ) :
        """
        Return the ids of child nodes of given node. Will accept node
        id or a leaf name.
        """
        if type(node_id) is str :
            try :
                node_id = self.leafs[ node_id ]
            except KeyError :
                raise Exception( 'Leaf name not found : ' + node_id )
        return ( self.data[node_id].left_child, self.data[node_id].right_child )
        
    def get_leafs( self, node_id ) :
        """
        Return an array of ids of all leaf nodes descendent from a given node.
        """
        cdef unsigned int i
        cdef int l
        cdef int r
        cdef unsigned int n = 0
        if self.np_buffer is None :
            self.np_buffer = np.ndarray( self.n_leafs, dtype=int )
        to_visit = [node_id]
        for i in to_visit :
            l,r = self.get_children( i )
            if l == -1 :
                self.np_buffer[n] = i
                n += 1
            else :
                to_visit.append( l )
                to_visit.append( r )
        return np.array(self.np_buffer[:n])
    
    def get_descendant_nodes( self, node_id ) :
        """
        Generator for ids of all nodes descendent from a given node,
        starting with the given node.
        """
        cdef unsigned int i
        cdef int l
        cdef int r
        cdef unsigned int n = 0
        
        to_visit = [node_id]
        for i in to_visit :
            l,r = self.get_children( i )
            if l == -1 :
                yield i
                continue
            else :
                to_visit.append( l )
                to_visit.append( r )
                yield i
    
    def get_internal_nodes( self, from_node=-1 ) :
        """
        Return an array of the ids of all internal nodes.
        """
        cdef unsigned int i
        cdef int l
        cdef int r
        cdef unsigned int n = 0
        
        if from_node == -1 : from_node = self.root
        
        if self.np_buffer is None :
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
        """
        Return an array of the ids of all internal nodes.
        """
        cdef unsigned int i
        cdef int l
        cdef int r
        cdef unsigned int n = 0
        
        if from_node == -1 : from_node = self.root
        
        if self.np_buffer is None :
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
 
    def get_distance_to_root( self, node_id ) :
        """
        Return distance to root for a given node. Will accept node id
        or a leaf name.
        """
        if type(node_id) is str :
            try :
                id = self.leafs[ node_id ]
            except KeyError :
                raise Exception( 'Leaf name not found : ' + node_id )
        return self._get_distance_to_root( node_id )
        
    @cython.boundscheck(False)
    cdef float _get_distance_to_root( self, node_id ) :
        """
        Calculate the distance from a node of a given id to the root node.
        Will work for both leaf and internal nodes. Private cdef method.
        """
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
    
    def is_ancestor( self, a, b ) :
        """
        Tristate : returns 1 if a is an ancestor of b, -1 if b is an
        ancestor of a, or 0 otherwise. Accepts node ids.
        """
        return self._is_ancestor( a, b )
        
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

    def mrca( self, a, b ) :
        """
        Return the id of the most recent common ancestor of two nodes
        if given ids.
        """
        visited = np.zeros( self.depth, dtype=int )
        
        return self._mrca( visited, a, b )
        
    @cython.boundscheck(False)
    cdef int _mrca( self, long[:] visited, int a, int b ) nogil :
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
        for i in xrange( self.depth ) :
            visited[i] = -1
            
        return mrca
        
    def distance( self, a, b ) :
        """
        Return distnace between a pair of nodes. Will treat strings as
        leaf names and integers as node ids. Either argument can be a
        leaf name or an integer.
        """
        if isinstance( a, str ) :
            try :
                a = self.leafs[a]
            except KeyError :
                raise Exception( 'Leaf name not found : ' + a )
        if isinstance( b, str ) :
            try :
                b = self.leafs[b]
            except KeyError :
                raise Exception( 'Leaf name not found : ' + b )
        
        if a < 0 or a >= self.length :
            raise Exception( 'node id out of bounds :', a )
        if b < 0 or b >= self.length :
            raise Exception( 'node id out of bounds :', b )
        
        return self._distance( a, b )
    
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
    
    def distances( self, long[:,:] ids ) :
        """
        Returns an array of distances between pairs of node ids,
        which are expected as an (n,2) array of type int.
        """
        if not ids.shape[1] == 2 :
            raise Exception( 'expected (n,2) array',
                             ids.shape[0], ids.shape[1] )
        
        visited = np.zeros( self.depth, dtype=int )
        result = np.zeros( ids.shape[0], dtype=float )
        self._distances( ids.shape[0], visited, ids, result )
        return result
        
    @cython.boundscheck(False)
    cdef void _distances( self, unsigned int length,
                                long[:] visited,
                                long[:,:] ids,
                                double[:] result ) nogil :
        """
        For each pair of node ids in the given (n,2) array, calculate the
        distance to the root node for each pair and store their differece
        in the given (1,n) result array. Calculations are performed within
        a 'nogil' context, allowing the interpreter to perform other tasks
        concurrently if desired. Private cdef method.
        """
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
        
    def adjacency( self, int node=-1 ) :
        """
        The graph adjacency matrix of the tree. If parameter 
        node is given, return graph adjacency matrix of the
        subtree descendent from node_id.
        """
        cdef unsigned int i
        cdef unsigned int j
        cdef unsigned int k
        cdef unsigned int node_id
        cdef unsigned int parent
        cdef float distance
        cdef int l
        cdef int r
        cdef unsigned int n = 0
        
        # by default, start from the root node
        if node == -1 :
            node = self.root
        
        # bail if the node isn't in our tree
        if node > self.length or node < -1 :
            raise Exception( 'Node id out of range.', node )
        
        self.np_buffer = np.ndarray( self.length, dtype=int )
        
        to_visit = [ node ]
        for i in to_visit :
            self.np_buffer[n] = i
            n += 1
            l,r = self.get_children( i )
            if l != -1 :
                to_visit.append( l )
                to_visit.append( r )
        
        ajmatrix = np.zeros( (n,n), dtype=float )
        
        for i in xrange( n ) :
            node_id  = self.np_buffer[i]
            parent   = self.data[node_id].parent
            if parent == -1 : continue
            distance = self.data[node_id].distance
            if distance == 0 : distance += self.epsilon
            for j,k in enumerate( self.np_buffer[:n] ) :
                if k == parent :
                    ajmatrix[ i,j ] = distance
                    ajmatrix[ j,i ] = distance
        
        return { 'adjacency_matrix' : ajmatrix,
                 'node_ids' : self.np_buffer[:n] }
        
    def laplacian( self, int node=-1 ) :
        """
        The graph Laplacian matrix of the tree, or if the parameter
        node is given, return the graph Laplacian matrix of the 
        subtree decendent from node.
        """
        if node == -1 :
            node = self.root
        
        aj, node_ids = self.adjacency( node=node ).values()
        lp = np.zeros( aj.shape )
        np.fill_diagonal( lp, aj.sum( axis=0 ) )
        lp = lp - aj
        
        return { 'laplacian' : lp,
                 'node_ids' : node_ids }
        
    def dump_array( self ) :
        """
        Print the whole tree. (WARNING : may be huge and useless.)
        """
        for n in range(self.length) :
            print( 'id : %d ->' % n )
            print( '   distance    : %0.3f' % self.data[n].distance    )
            print( '   parent      : %d'    % self.data[n].parent      )
            print( '   left child  : %d'    % self.data[n].left_child  )
            print( '   right child : %d'    % self.data[n].right_child )
        
    def nodes_data( self ) :
        """
        Generator for the node data in the tree, compatible with networkx.
        """
        for n in range(self.length) :
            if self.data[n].left_child == -1 :
                leaf_name = self.leafnodes[n]
            else :
                leaf_name = ''
            yield ( n, { 'label' : leaf_name } )
    
    def edges_data( self ) :
        """
        Generator for the edge (i.e. branch) data in the tree, compatible with networkx.
        """
        for n in range(self.length) :
            # no edges beyond the root node
            if self.data[n].parent == -1 : continue
            yield ( n, self.data[n].parent, { 'weight' : self.data[n].distance } )

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
            for rowname, value in s.iteritems() :
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
        """
        Compute distances for all pairs of links. For large link
        tables, this will fail on memory allocation.
        """
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
        """
        Build the graph adjacency matrix of the current subsetted
        trees, applying the specified random permutaitons.
        """
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
        """
        The graph Laplacian matrix of the current subsetted trees.
        """
        
        aj = self.adjacency( deletions=deletions,
                             additions=additions,
                             swaps=swaps )
        lp = np.zeros( aj.shape )
        np.fill_diagonal( lp, aj.sum( axis=0 ) )
        lp = lp - aj
        
        return lp
        
    def spectrum( self, deletions=0, additions=0, swaps=0 ) :
        """
        The eigenvalues of the graph Laplacian matrix of the current
        subsetted trees.
        """
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
        """
        Return the current SuchLinkedTrees subgraph as a weighted,
        labled igraph object. The igraph package must be installed.
        """
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
