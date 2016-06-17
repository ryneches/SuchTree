import cython
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from dendropy import Tree
import numpy as np
cimport numpy as np
import pandas as pd

# Trees are built from arrays of Node structs. 'parent', 'left_child'
# and 'right_child' attributes represent integer offsets within the
# array that specify other Node structs.
cdef struct Node :
    int parent
    int left_child
    int right_child
    float distance

@cython.boundscheck(False)
cdef double _pearson( double[:] x, double[:] y ) :
    cdef unsigned int n = len(x)
    cdef unsigned long j
    cdef float yt,xt,t,df
    cdef float syy=0.0,sxy=0.0,sxx=0.0,ay=0.0,ax=0.0
    with nogil :
        for j in xrange(n) :
            ax += x[j]
            ay += y[j]
        ax /= n
        ay /= n
        for j in xrange(n) :
            xt=x[j]-ax
            yt=y[j]-ay
            sxx += xt*xt
            syy += yt*yt
            sxy += xt*yt
        return sxy/((sxx*syy)+1.0e-20)**(0.5)

def pearson( double[:] x, double[:] y ) :
    try :
        return _pearson( x, y )
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
        if not self.data[leaf_id].right_child == -1 :
            raise Exception( 'Cannot link non-leaf node.', leaf_id )
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

cdef struct Link :
    unsigned int a
    unsigned int b

cdef struct LinkColumn :
    unsigned int length
    unsigned int* links

@cython.boundscheck(False)
cdef unsigned int _subset_guest_tree( long[:] col_ids, 
                                      LinkColumn* linkmatrix, 
                                      Link* linklist ) :
    cdef unsigned int n = 0
    cdef unsigned int n_leafs = len( col_ids )
    cdef unsigned int i
    cdef unsigned int j
    cdef unsigned int col
    print 'address check 1 :', <unsigned int>&linkmatrix
    with nogil :
        for i in xrange( n_leafs ) :
            col = col_ids[i]
            for j in xrange( linkmatrix[col].length ) :
                linklist[n].a = linkmatrix[col].links[j]
                linklist[n].b = col 
                n += 1
    return n
  
cdef class SuchLinkedTrees :
    """
    The links argument can be an (n,2) numpy array of ids or a (n,2)
    list of tuples of leaf names.
    
    The leafs of TreeA correspond to axis 0 of the link matrix (row
    names), and the leafs of TreeB correspond to axis 1 of the link 
    matrix (column names).
    """
    cdef LinkColumn* linkmatrix
    cdef Link* linklist
    cdef unsigned int n_cols
    cdef unsigned int n_rows
    cdef unsigned int n_links
    cdef unsigned int subset_node
    cdef unsigned int subset_n_leafs
    cdef unsigned int subset_n_links
    cdef object np_linkmatrix
    cdef object np_linklist
    cdef object TreeA
    cdef object TreeB
    cdef object linked_leafsA
    cdef object linked_leafsB
    cdef object col_ids
    cdef object row_ids
    cdef object col_names
    cdef object row_names    
    
    def __init__( self, tree_file_a, tree_file_b, link_matrix ) :
        
        cdef unsigned int i
        cdef unsigned int j
        cdef unsigned int link       
        
        # these objects are constructed only when first accessed
        self.np_linkmatrix = None
        self.np_linklist = None
        self.linked_leafsA = None
        self.linked_leafsB = None
        self.col_ids = None
        self.row_ids = None
        self.col_names = None
        self.row_names = None    
    
        if not type(link_matrix) is pd.DataFrame :
            raise Exception( 'unsupported type for link matrix', type(link_matrix) )
               
        # build trees from newick files, URLs to newick files or 
        # from existing SuchTrees
        if type( tree_file_a ) == str : 
            self.TreeA = SuchTree( tree_file_a )
        elif type( tree_file_a ) == SuchTree :
            self.TreeA = tree_file_a
        else :
            raise Exception( 'unknown input for tree', type(tree_file_a) )
        
        # build trees from newick files, URLs to newick files or 
        # from existing SuchTrees
        if type( tree_file_b ) == str : 
            self.TreeB = SuchTree( tree_file_b )
        elif type( tree_file_b ) == SuchTree :
            self.TreeB = tree_file_b
        else :
            raise Exception( 'unknown input for tree', type(tree_file_b) )
        
        # make sure the link matrix connects the trees
        if not link_matrix.shape == ( self.TreeA.n_leafs, self.TreeB.n_leafs ) :
            raise Exception( 'link_matrix shape must match tree leaf counts' )
        
        if not set(link_matrix.axes[0]) == set(self.TreeA.leafs.keys()) :
            raise Exception( 'axis[0] does not match TreeA leaf names' )
        
        if not set(link_matrix.axes[1]) == set(self.TreeB.leafs.keys()) :
            raise Exception( 'axis[1] does not match TreeB leaf names' )
        
        # set row and column indexes
        self.row_ids = self.TreeA.leafs.values()
        self.row_names = self.TreeA.leafs.keys()
        self.col_ids = self.TreeB.leafs.values()
        self.col_names = self.TreeB.leafs.keys()
        
        # allocate memory for link columns
        self.n_rows, self.n_cols = self.TreeA.n_leafs, self.TreeB.n_leafs
        self.n_links = 0
        self.linkmatrix = <LinkColumn*> PyMem_Malloc( self.n_cols * sizeof( LinkColumn ) )
        for i,col in enumerate( self.col_names ) :
            # link the guest tree leaf to the link matrix column
            col_id = self.TreeB.leafs[ col ]
            self.TreeB.link_leaf( col_id, i )
            s = link_matrix[ col ]
            l = map( lambda x : self.TreeA.leafs[x], s[ s > 0 ].to_dict().keys() )
            size = len(l)
            self.n_links += size
            # allocate memory for links in this column
            self.linkmatrix[i].length = size
            self.linkmatrix[i].links = <unsigned int*> PyMem_Malloc( size * sizeof( unsigned int ) )
            for j,link in enumerate(l) :
                self.linkmatrix[i].links[j] = link
        
        print 'address check 0 :', <unsigned int>&self.linkmatrix
        # allocate memory for link list
        self.linklist = <Link*> PyMem_Malloc( self.n_links * sizeof( Link ) )
        self.subset_guest_tree( self.TreeB.root )
    
    def __dealloc__( self ) :
        print 'address check 3 :', <unsigned int>&self.linkmatrix
        print 'check 0'
        for i in xrange( self.n_cols ) :
            print 'check 1', i
            PyMem_Free( self.linkmatrix[i].links )
        print 'check 2'
        PyMem_Free( self.linkmatrix )
        print 'check 3'
        PyMem_Free( self.linklist )
        print 'check 4'
        
    property TreeA :
        'first tree initialized by SuchLinkedTrees( TreeA, TreeB )'
        def __get__( self ) :
            return self.TreeA
    
    property TreeB :
        'second tree initialized by SuchLinkedTrees( TreeA, TreeB )'
        def __get__( self ) :
            return self.TreeB
    
    property linkmatrix :
        'numpy representation of link list (generated only on access)'
        def __get__( self ) :
            cdef unsigned int i
            cdef unsigned int j
            cdef unsigned int l
            if self.np_linkmatrix is None :
                self.np_linkmatrix = np.zeros( (self.n_rows,self.n_cols), dtype=bool )
                row_ids = self.TreeA.leafs.values()
                for i in xrange( self.n_cols ) :
                    for j in xrange( self.linkmatrix[i].length ) :
                        l = self.linkmatrix[i].links[j]
                        self.np_linkmatrix[ row_ids.index( l ), i ] = True
            return self.np_linkmatrix
    
    property linklist :
        def __get__( self ) :
            cdef unsigned int i
            if self.np_linklist is None :
                self.np_linklist = np.ndarray( (self.subset_n_links, 2), dtype=int )
                for i in xrange( self.subset_n_links ) :
                    self.np_linklist[ i, : ] = self.linklist[i].a, self.linklist[i].b
            return self.np_linklist
            
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
    
    property subset_node :
        'id of the current subsettng node (default is root).'
        def __get__( self ) :
            return self.subset_node
    
    property subset_n_leafs :
        'Number of leafs in the current subset.'
        def __get__( self ) :
            return self.subset_n_leafs
    
    property subset_n_links :
        'Number of links in the current subset.'
        def __get__( self ) :
            return self.subset_n_links
    
    def subset_guest_tree( self, node_id ) :
        print 'address check 2 :', <unsigned int>&self.linkmatrix
        leafs = self.TreeA.get_leafs( node_id )
        col_ids = self.TreeA.get_links( leafs )
        n = _subset_guest_tree( col_ids, self.linkmatrix, self.linklist )
        self.subset_node = node_id
        self.subset_n_leafs = len(leafs)
        self.subset_n_links = n
        self.np_linklist = None
        
    #broken
    def linked_distances( self ) :
        """
        Compute distances for all pairs of links. For large link
        tables, this will fail on memory allocation.
        """
        cdef unsigned int i
        cdef unsigned int j
        cdef unsigned int k = 0
        cdef unsigned int size = ( self.n_links * (self.n_links-1) ) / 2
        #if self.linklist == NULL :
            
            
        ids_a = np.ndarray( ( size, 2 ), dtype=int )
        ids_b = np.ndarray( ( size, 2 ), dtype=int )
        for i in xrange( self.n_cols ) :
            b = self.col_ids[i]
            for j in xrange( self.linkmatrix[i].length ) :
                ids_a[ k, : ] = self.links[ i ].a, self.links[ j ].a
                ids_b[ k, : ] = self.links[ i ].b, self.links[ j ].b
                k += 1
        return { 'TreeA' : self.TreeA.distances( ids_a ), 
                 'TreeB' : self.TreeB.distances( ids_b ) }
    
    #broken
    def sample_linked_distances( self, sigma=0.001, buckets=64, n=4096 ) :
        ids_a = np.ndarray( (n,2), dtype=int )
        ids_b = np.ndarray( (n,2), dtype=int )
        a_buckets = []
        b_buckets = []
        for i in xrange(buckets) :
            a_buckets.append( np.array([]) )
            b_buckets.append( np.array([]) )
        s_a = 10e10
        s_b = 10e10
        a_sigmas = []
        b_sigmas = []
        while True :
            for i in xrange(buckets) :
                l1 = np.random.randint( 0, self.n_links, n )
                l2 = np.random.randint( 0, self.n_links, n )
                for j in xrange(n) :
                    ids_a[ j, : ] = self.links[ l1[j] ].a, self.links[ l2[j] ].a
                    ids_b[ j, : ] = self.links[ l1[j] ].b, self.links[ l2[j] ].b
                a_result = self.TreeA.distances( ids_a )
                b_result = self.TreeB.distances( ids_b )
                a_buckets[i] = np.append( a_buckets[i], a_result )
                b_buckets[i] = np.append( b_buckets[i], b_result )
            a_sigmas.append( np.std( a_buckets ) )
            b_sigmas.append( np.std( b_buckets ) )
            if len( a_sigmas ) == 1 : continue
            s_a = np.std( a_sigmas )
            s_b = np.std( b_sigmas )
            if s_a < sigma and s_b < sigma : break
        return { 'TreeA' : reduce( np.append, a_buckets ),
                 'TreeB' : reduce( np.append, b_buckets ),
                 'sigma_a' : s_a,
                 'sigma_b' : s_b }
