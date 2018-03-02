from nose.tools import assert_equal, assert_almost_equal, assert_true
from SuchTree import SuchTree
from dendropy import Tree
from itertools import combinations
import numpy

test_tree = 'SuchTree/tests/test.tree'
dpt = Tree.get( file=open(test_tree), schema='newick' )
dpt.resolve_polytomies()
for n,node in enumerate( dpt.inorder_node_iter() ) :
    node.label = n

def test_init() :
    T = SuchTree( test_tree )
    assert_equal( type(T), SuchTree )

def test_get_children() :
    T = SuchTree( test_tree )
    for node in dpt.inorder_node_iter() :
        if not node.taxon :
            left, right = [ n.label for n in node.child_nodes() ]
        else :
            left, right = -1, -1
        L,R = T.get_children( node.label )
        assert_equal( L, left )
        assert_equal( R, right )

def test_get_distance_to_root() :
    T = SuchTree( test_tree )
    for leaf in dpt.leaf_node_iter() :
        assert_almost_equal( T.get_distance_to_root( leaf.label ),
                             leaf.distance_from_root(),
                             places=4 )

def test_distance() :
    T = SuchTree( test_tree )
    for line in open( 'SuchTree/tests/test.matrix' ) :
        a,b,d1 = line.split()
        d1 = float(d1)
        d2 = T.distance( a, b )
        assert_almost_equal( d1, d2, places=4 )   

def test_distances() :
    T = SuchTree( test_tree )
    ids = []
    d1 = []
    for line in open( 'SuchTree/tests/test.matrix' ) :
        a,b,d = line.split()
        d1.append( float(d) )
        A = T.leafs[a]
        B = T.leafs[b]
        ids.append( (A,B) )
    result = T.distances( numpy.array( ids, dtype=numpy.int64 ) )
    for D1,D2 in zip( d1,result ) :
        assert_almost_equal( D1, D2, places=4 )

def test_distances_by_name() :
    T = SuchTree( test_tree )
    ids = []
    d1 = []
    for line in open( 'SuchTree/tests/test.matrix' ) :
        a,b,d = line.split()
        d1.append( float(d) )
        ids.append( (a,b) )
    result = T.distances_by_name( ids )
    for D1,D2 in zip( d1,result ) :
        assert_almost_equal( D1, D2, places=4 )

def test_get_leafs() :
    T = SuchTree( test_tree )
    assert_equal( set( list(T.get_leafs( T.root )) ), set( T.leafs.values() ) )

def test_hierarchy() :
    T = SuchTree( test_tree )
    all_leafs = set( T.get_leafs( T.root ) )
    for i in T.get_internal_nodes() :
        some_leafs = set( T.get_leafs( i ) )
        assert_true( some_leafs <= all_leafs )

def test_adjacency() :
    T = SuchTree( test_tree )
    aj, leaf_ids = T.adjacency( T.root ).values()
    leaf_ids = list( leaf_ids )
    for node in T.leafs.values() + list(T.get_internal_nodes() ):
        if node == T.root : continue # skip the root node
        parent = T.get_parent( node )
        distance = T.distance( node, parent )
        i,j = leaf_ids.index( node ), leaf_ids.index( parent )
        print node, parent, ':', i, j, ' :: ', aj[i,j], distance

def test_get_descendant_nodes() :
    T = SuchTree( test_tree )
    A = set( T.get_descendant_nodes( T.root ) )
    B = set( T.get_leafs( T.root ) )
    C = set( T.get_internal_nodes() )
    assert_equal( A, B | C )

