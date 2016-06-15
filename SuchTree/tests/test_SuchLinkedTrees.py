from nose.tools import assert_equal, assert_almost_equal
from SuchTree import SuchTree, SuchLinkedTrees, pearson
from dendropy import Tree
from itertools import combinations
import numpy
import pandas as pd

test_tree = 'SuchTree/tests/test.tree'
dpt = Tree.get( file=open(test_tree), schema='newick' )
for n,node in enumerate( dpt.inorder_node_iter() ) :
    node.label = n

def test_init_link_by_name() :
    T = SuchTree( test_tree )
    links = pd.DataFrame( numpy.random.random_integers( 0, 3, size=(14,14)),
                          columns=T.leafs.keys(), 
                          index=T.leafs.keys() )

    SLT = SuchLinkedTrees( T, T, links )
    assert_equal( type(SLT), SuchLinkedTrees )

def test_init_link_by_id() :
    T = SuchTree( test_tree )
    links = pd.DataFrame( numpy.random.random_integers( 0, 3, size=(14,14)),
                          columns=T.leafs.keys(), 
                          index=T.leafs.keys() )
    SLT = SuchLinkedTrees( T, T, links )
    assert_equal( type(SLT), SuchLinkedTrees )

def test_init_one_tree_by_file() :
    T = SuchTree( test_tree )
    links = pd.DataFrame( numpy.random.random_integers( 0, 3, size=(14,14)),
                          columns=T.leafs.keys(), 
                          index=T.leafs.keys() )
    SLT = SuchLinkedTrees( test_tree, T, links )
    assert_equal( type(SLT), SuchLinkedTrees )

def test_init_both_trees_by_file() :
    T = SuchTree( test_tree )
    links = pd.DataFrame( numpy.random.random_integers( 0, 3, size=(14,14)),
                          columns=T.leafs.keys(), 
                          index=T.leafs.keys() )
    SLT = SuchLinkedTrees( test_tree, test_tree, links )
    assert_equal( type(SLT), SuchLinkedTrees )

# test properties

def test_col_names() :
    T = SuchTree( test_tree )
    links = pd.DataFrame( numpy.random.random_integers( 0, 3, size=(14,14)),
                          columns=T.leafs.keys(), 
                          index=T.leafs.keys() )

    SLT = SuchLinkedTrees( T, T, links )
    assert_equal( SLT.col_names, T.leafs.keys() )

def test_row_names() :
    T = SuchTree( test_tree )
    links = pd.DataFrame( numpy.random.random_integers( 0, 3, size=(14,14)),
                          columns=T.leafs.keys(), 
                          index=T.leafs.keys() )

    SLT = SuchLinkedTrees( T, T, links )
    assert_equal( SLT.row_names, T.leafs.keys() )

def test_col_ids() :
    T = SuchTree( test_tree )
    links = pd.DataFrame( numpy.random.random_integers( 0, 3, size=(14,14)),
                          columns=T.leafs.keys(), 
                          index=T.leafs.keys() )

    SLT = SuchLinkedTrees( T, T, links )
    assert_equal( SLT.col_ids, T.leafs.values() )

def test_row_ids() :
    T = SuchTree( test_tree )
    links = pd.DataFrame( numpy.random.random_integers( 0, 3, size=(14,14)),
                          columns=T.leafs.keys(), 
                          index=T.leafs.keys() )

    SLT = SuchLinkedTrees( T, T, links )
    assert_equal( SLT.row_ids, T.leafs.values() )

def test_n_cols() :
    T = SuchTree( test_tree )
    links = pd.DataFrame( numpy.random.random_integers( 0, 3, size=(14,14)),
                          columns=T.leafs.keys(), 
                          index=T.leafs.keys() )

    SLT = SuchLinkedTrees( T, T, links )
    assert_equal( SLT.n_cols, T.n_leafs )

def test_n_rows() :
    T = SuchTree( test_tree )
    links = pd.DataFrame( numpy.random.random_integers( 0, 3, size=(14,14)),
                          columns=T.leafs.keys(), 
                          index=T.leafs.keys() )

    SLT = SuchLinkedTrees( T, T, links )
    assert_equal( SLT.n_rows, T.n_leafs )

def test_linkmatrix_property() :
    T = SuchTree( test_tree )
    links = pd.DataFrame( numpy.random.random_integers( 0, 3, size=(14,14)),
                          columns=T.leafs.keys(), 
                          index=T.leafs.keys() )
    SLT = SuchLinkedTrees( T, T, links )
    L = SLT.linkmatrix
    LL = pd.DataFrame( L, columns=SLT.col_names, index=SLT.row_names )
    b_links = links.applymap( lambda x : bool(x) )
    for row in SLT.row_names :
        for col in SLT.col_names :
            assert_equal( LL[ row ][ col ], b_links[ row ][ col ] )
