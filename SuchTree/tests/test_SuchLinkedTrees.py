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
    col_ids = SLT.col_ids
    leaf_ids = T.leafs.values()
    assert_equal( len(col_ids), len(leaf_ids) )
    for i,j in zip( col_ids, leaf_ids ) :
        assert_equal( i,j )

def test_row_ids() :
    T = SuchTree( test_tree )
    links = pd.DataFrame( numpy.random.random_integers( 0, 3, size=(14,14)),
                          columns=T.leafs.keys(), 
                          index=T.leafs.keys() )

    SLT = SuchLinkedTrees( T, T, links )
    row_ids = SLT.row_ids
    leaf_ids = T.leafs.values()
    assert_equal( len(row_ids), len(leaf_ids) )
    for i,j in zip( row_ids, leaf_ids ) :
        assert_equal( i,j )

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

def test_get_column_leafs() :
    T = SuchTree( test_tree )
    links = pd.DataFrame( numpy.random.random_integers( 0, 3, size=(14,14)),
                          columns=T.leafs.keys(), 
                          index=T.leafs.keys() )
    SLT = SuchLinkedTrees( T, T, links )
    for n,colname in enumerate( links.columns ) :
        s = links.applymap(bool)[ colname ]
        leafs1 = set( map( lambda x : T.leafs[x],  s[ s > 0 ].index ) )
        leafs2 = set( SLT.get_column_leafs(n) )
        assert_equal( leafs1, leafs2 )

def test_get_column_leafs_by_name() :
    T = SuchTree( test_tree )
    links = pd.DataFrame( numpy.random.random_integers( 0, 3, size=(14,14)),
                          columns=T.leafs.keys(), 
                          index=T.leafs.keys() )
    SLT = SuchLinkedTrees( T, T, links )
    for colname in links.columns :
        s = links.applymap(bool)[ colname ]
        leafs1 = set( map( lambda x : T.leafs[x],  s[ s > 0 ].index ) )
        leafs2 = set( SLT.get_column_leafs( colname ) )
        assert_equal( leafs1, leafs2 )

def test_get_column_leafs_as_row_ids() :
    T = SuchTree( test_tree )
    links = pd.DataFrame( numpy.random.random_integers( 0, 3, size=(14,14)),
                          columns=T.leafs.keys(), 
                          index=T.leafs.keys() )
    SLT = SuchLinkedTrees( T, T, links )
    for n,colname in enumerate( links.columns ) :
        s = links.applymap(bool)[ colname ]
        leafs1 = set( map( list(SLT.col_ids).index, map( lambda x : T.leafs[x],  s[ s > 0 ].index ) ) )
        leafs2 = set( SLT.get_column_leafs(n, as_row_ids=True) )
        assert_equal( leafs1, leafs2 )

def test_get_column_leafs_by_name_as_row_ids() :
    T = SuchTree( test_tree )
    links = pd.DataFrame( numpy.random.random_integers( 0, 3, size=(14,14)),
                          columns=T.leafs.keys(), 
                          index=T.leafs.keys() )
    SLT = SuchLinkedTrees( T, T, links )
    for colname in links.columns :
        s = links.applymap(bool)[ colname ]
        leafs1 = set( map( list(SLT.col_ids).index, map( lambda x : T.leafs[x],  s[ s > 0 ].index ) ) )
        leafs2 = set( SLT.get_column_leafs( colname, as_row_ids=True ) )
        assert_equal( leafs1, leafs2 )

def test_get_column_links() :
    T = SuchTree( test_tree )
    row_names = T.leafs.keys()
    numpy.random.shuffle(row_names)
    links = pd.DataFrame( numpy.random.random_integers( 0, 3, size=(14,14)),
                          columns=T.leafs.keys(), 
                          index=row_names )
    SLT = SuchLinkedTrees( T, T, links )
    for n,colname in enumerate( links.columns ) :
        s = links.applymap(bool)[ colname ]
        c = SLT.get_column_links(n)
        for m,rowname in enumerate( SLT.row_names ) :
            assert_equal( s[rowname], c[m] )
 
def test_linkmatrix_property() :
    T = SuchTree( test_tree )
    row_names = T.leafs.keys()
    numpy.random.shuffle(row_names)
    links = pd.DataFrame( numpy.random.random_integers( 0, 3, size=(14,14)),
                          columns=T.leafs.keys(), 
                          index=row_names )
    SLT = SuchLinkedTrees( T, T, links )
    for col in SLT.col_names :
        for row in SLT.row_names :
            col_id = SLT.col_names.index(col)
            row_id = SLT.row_names.index(row)
            assert_equal( bool(links[col][row]), SLT.linkmatrix[col_id][row_id] )
