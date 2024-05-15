from __future__ import print_function

import pytest
from pytest import approx
from SuchTree import SuchTree
from dendropy import Tree
from itertools import combinations, chain
import numpy

try :
    import networkx
    has_networkx = True
except ImportError :
    has_networkx = False

test_tree            = 'SuchTree/tests/test.tree'
support_tree_int     = 'SuchTree/tests/support_int.tree'
support_tree_float   = 'SuchTree/tests/support_float.tree'
support_tree_comment = 'SuchTree/tests/support_comment.tree'

dpt = Tree.get( file=open(test_tree), schema='newick' )
dpt.resolve_polytomies()
for n,node in enumerate( dpt.inorder_node_iter() ) :
    node.label = n

def test_init() :
    T = SuchTree( test_tree )
    assert type(T) == SuchTree

def test_get_children() :
    T = SuchTree( test_tree )
    for node in dpt.inorder_node_iter() :
        if not node.taxon :
            left, right = [ n.label for n in node.child_nodes() ]
        else :
            left, right = -1, -1
        L,R = T.get_children( node.label )
        assert L == left
        assert R == right

def test_get_distance_to_root() :
    T = SuchTree( test_tree )
    for leaf in dpt.leaf_node_iter() :
        assert T.get_distance_to_root( leaf.label ) == approx( leaf.distance_from_root(), 0.001 )

def test_get_distance_to_root_inputs() :
    T = SuchTree( test_tree )
    for leaf in T.leafs.keys() :
        assert T.get_distance_to_root( leaf ) == T.get_distance_to_root( T.leafs[leaf] )

def test_distance() :
    T = SuchTree( test_tree )
    for line in open( 'SuchTree/tests/test.matrix' ) :
        a,b,d1 = line.split()
        d1 = float(d1)
        d2 = T.distance( a, b )
        assert d1 == approx( d2, 0.001 )

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
        assert D1 == approx( D2, 0.001 )

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
        assert D1 == approx( D2, 0.001 )

def test_get_int_support_by_id() :
    T = SuchTree( support_tree_int )
    for node_id in T.get_nodes() :
        assert T.get_support( node_id ) != 0

def test_get_int_support_by_name() :
    T = SuchTree( support_tree_int )
    for leaf in T.leafs.keys() :
        assert T.get_support( leaf ) < 0

def test_get_float_support_by_id() :
    T = SuchTree( support_tree_float )
    for node_id in T.get_nodes() :
        assert T.get_support( node_id ) != 0

def test_get_comment_support_by_id() :
    T = SuchTree( support_tree_comment )
    for node_id in T.get_nodes() :
        assert T.get_support( node_id ) != 0

def test_get_leafs() :
    T = SuchTree( test_tree )
    assert set( list(T.get_leafs( T.root )) ) == set( T.leafs.values() )

def test_hierarchy() :
    T = SuchTree( test_tree )
    all_leafs = set( T.get_leafs( T.root ) )
    for i in T.get_internal_nodes() :
        some_leafs = set( T.get_leafs( i ) )
        assert some_leafs <= all_leafs

def test_adjacency() :
    T = SuchTree( test_tree )
    aj, leaf_ids = T.adjacency( T.root ).values()
    leaf_ids = list( leaf_ids )
    for node in chain(T.leafs.values(), list(T.get_internal_nodes() )):
        if node == T.root : continue # skip the root node
        parent = T.get_parent( node )
        distance = T.distance( node, parent )
        i,j = leaf_ids.index( node ), leaf_ids.index( parent )
        print( node, parent, ':', i, j, ' :: ', aj[i,j], distance )

def test_get_descendant_nodes() :
    T = SuchTree( test_tree )
    A = set( T.get_descendant_nodes( T.root ) )
    B = set( T.get_leafs( T.root ) )
    C = set( T.get_internal_nodes() )
    assert A == B | C

def test_is_ancestor() :
    T = SuchTree( test_tree )
    assert T.length - 1 == sum( map( lambda x : T.is_ancestor( T.root, x ), 
                                T.get_descendant_nodes( T.root ) ) )
    assert 1 - T.length == sum( map( lambda x : T.is_ancestor( x, T.root ),
                                T.get_descendant_nodes( T.root ) ) )

def test_is_leaf() :
    T = SuchTree( test_tree )
    for leaf in T.leafs.values() :
        assert T.is_leaf( leaf )

def test_is_internal_node() :
    T = SuchTree( test_tree )
    for node in T.get_internal_nodes() :
        assert T.is_internal_node( node )

def test_mrca_by_id() :
    T = SuchTree( test_tree )
    leaf_ids = T.leafs.values()
    for a,b in combinations( leaf_ids, 2 ) :
        assert T.mrca( a, b ) == T.mrca( b, a )
        mrca = T.mrca( a, b )
        assert T.is_ancestor( mrca, a )
        assert T.is_ancestor( mrca, b )
        assert a in T.get_descendant_nodes( mrca )
        assert b in T.get_descendant_nodes( mrca )

def test_mrca_by_name() :
    T = SuchTree( test_tree )
    leafs = T.leafs.keys()
    for a,b in combinations( leafs, 2 ) :
        assert T.mrca( a, b ) == T.mrca( b, a )
        mrca = T.mrca( a, b )
        assert T.is_ancestor( mrca, a )
        assert T.is_ancestor( mrca, b )
        assert T.leafs[a] in T.get_descendant_nodes( mrca )
        assert T.leafs[b] in T.get_descendant_nodes( mrca )

def test_bipartitions() :
    # a tree should have no conflicting bipartitions with itself
    T = SuchTree( test_tree )
    S = []
    for A,B in combinations( T.bipartitions(), 2 ) :
        A0,A1 = A
        B0,B1 = B
        S.append( ( not bool( A0 & B0 ) ) \
                | ( not bool( A1 & B0 ) ) \
                | ( not bool( A0 & B1 ) ) \
                | ( not bool( A1 & B1 ) ) )
        assert sum(S)/len(S) == 1.0

def test_in_order() :
    T = SuchTree( test_tree )
    traversal = set( node for node,d in T.in_order() ) 
    assert traversal == set( T.get_nodes() )

@pytest.mark.skipif(not has_networkx, reason="networkx not installed")
def test_networkx() :
    T = SuchTree( test_tree )
    g = networkx.graph.Graph()
    
    g.add_nodes_from( T.nodes_data() )
    g.add_edges_from( T.edges_data() )
    
    assert set( g.nodes() ) == set( T.get_nodes() )
