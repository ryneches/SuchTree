from __future__ import print_function

from functools import reduce
import pytest
from SuchTree import SuchTree, SuchLinkedTrees
from dendropy import Tree
from itertools import combinations
import numpy
import pandas as pd
import tempfile

try :
    import igraph
    has_igraph = True
except ImportError :
    has_igraph = False

test_tree = 'SuchTree/tests/test.tree'
dpt = Tree.get( file=open(test_tree), schema='newick' )
dpt.resolve_polytomies()
N = 0
for n,node in enumerate( dpt.inorder_node_iter() ) :
    node.label = n
    if node.taxon :
        N += 1

# gopher/louse dataset
gopher_tree = 'SuchTree/tests/gopher.tree'
lice_tree   = 'SuchTree/tests/lice.tree'
gl_links    = 'SuchTree/tests/links.csv'

def test_init_link_by_name() :
    T = SuchTree( test_tree )
    links = pd.DataFrame( numpy.random.randint( 0, 3, size=(N,N) ),
                          columns=list(T.leafs.keys()), 
                          index=list(T.leafs.keys()) )

    SLT = SuchLinkedTrees( T, T, links )
    assert type(SLT) == SuchLinkedTrees

def test_init_link_by_id() :
    T = SuchTree( test_tree )
    links = pd.DataFrame( numpy.random.randint( 0, 3, size=(N,N) ),
                          columns=list(T.leafs.keys()), 
                          index=list(T.leafs.keys()) )
    SLT = SuchLinkedTrees( T, T, links )
    assert type(SLT) == SuchLinkedTrees

def test_init_one_tree_by_file() :
    T = SuchTree( test_tree )
    links = pd.DataFrame( numpy.random.randint( 0, 3, size=(N,N) ),
                          columns=list(T.leafs.keys()), 
                          index=list(T.leafs.keys()) )
    SLT = SuchLinkedTrees( test_tree, T, links )
    assert type(SLT) == SuchLinkedTrees

def test_init_both_trees_by_file() :
    T = SuchTree( test_tree )
    links = pd.DataFrame( numpy.random.randint( 0, 3, size=(N,N) ),
                          columns=list(T.leafs.keys()), 
                          index=list(T.leafs.keys()) )
    SLT = SuchLinkedTrees( test_tree, test_tree, links )
    assert type(SLT) == SuchLinkedTrees

# test properties

def test_col_names() :
    T = SuchTree( test_tree )
    links = pd.DataFrame( numpy.random.randint( 0, 3, size=(N,N) ),
                          columns=list(T.leafs.keys()), 
                          index=list(T.leafs.keys()) )

    SLT = SuchLinkedTrees( T, T, links )
    assert SLT.col_names == list(T.leafs.keys())

def test_row_names() :
    T = SuchTree( test_tree )
    links = pd.DataFrame( numpy.random.randint( 0, 3, size=(N,N) ),
                          columns=list(T.leafs.keys()), 
                          index=list(T.leafs.keys()) )

    SLT = SuchLinkedTrees( T, T, links )
    assert SLT.row_names == list(T.leafs.keys())

def test_col_ids() :
    T = SuchTree( test_tree )
    links = pd.DataFrame( numpy.random.randint( 0, 3, size=(N,N) ),
                          columns=list(T.leafs.keys()), 
                          index=list(T.leafs.keys()) )

    SLT = SuchLinkedTrees( T, T, links )
    col_ids = SLT.col_ids
    leaf_ids = T.leafs.values()
    assert len(col_ids) == len(leaf_ids)
    for i,j in zip( col_ids, leaf_ids ) :
        assert i == j

def test_row_ids() :
    T = SuchTree( test_tree )
    links = pd.DataFrame( numpy.random.randint( 0, 3, size=(N,N) ),
                          columns=list(T.leafs.keys()), 
                          index=list(T.leafs.keys()) )

    SLT = SuchLinkedTrees( T, T, links )
    row_ids = SLT.row_ids
    leaf_ids = T.leafs.values()
    assert len(row_ids) == len(leaf_ids)
    for i,j in zip( row_ids, leaf_ids ) :
        assert i == j

def test_n_cols() :
    T = SuchTree( test_tree )
    links = pd.DataFrame( numpy.random.randint( 0, 3, size=(N,N) ),
                          columns=list(T.leafs.keys()), 
                          index=list(T.leafs.keys()) )

    SLT = SuchLinkedTrees( T, T, links )
    assert SLT.n_cols == T.n_leafs

def test_n_rows() :
    T = SuchTree( test_tree )
    links = pd.DataFrame( numpy.random.randint( 0, 3, size=(N,N) ),
                          columns=list(T.leafs.keys()), 
                          index=list(T.leafs.keys()) )

    SLT = SuchLinkedTrees( T, T, links )
    assert SLT.n_rows == T.n_leafs

def test_get_column_leafs() :
    T = SuchTree( test_tree )
    links = pd.DataFrame( numpy.random.randint( 0, 3, size=(N,N) ),
                          columns=list(T.leafs.keys()), 
                          index=list(T.leafs.keys()) )
    SLT = SuchLinkedTrees( T, T, links )
    for n,colname in enumerate( links.columns ) :
        s = links.map(bool)[ colname ]
        leafs1 = set( map( lambda x : T.leafs[x],  s[ s > 0 ].index ) )
        leafs2 = set( SLT.get_column_leafs(n) )
        assert leafs1 == leafs2

def test_get_column_leafs_by_name() :
    T = SuchTree( test_tree )
    links = pd.DataFrame( numpy.random.randint( 0, 3, size=(N,N) ),
                          columns=list(T.leafs.keys()), 
                          index=list(T.leafs.keys()) )
    SLT = SuchLinkedTrees( T, T, links )
    for colname in links.columns :
        s = links.map(bool)[ colname ]
        leafs1 = set( map( lambda x : T.leafs[x],  s[ s > 0 ].index ) )
        #print( colname )
        leafs2 = set( SLT.get_column_leafs( colname ) )
        assert leafs1 == leafs2

def test_get_column_leafs_as_row_ids() :
    T = SuchTree( test_tree )
    links = pd.DataFrame( numpy.random.randint( 0, 3, size=(N,N) ),
                          columns=list(T.leafs.keys()), 
                          index=list(T.leafs.keys()) )
    SLT = SuchLinkedTrees( T, T, links )
    for n,colname in enumerate( links.columns ) :
        s = links.map(bool)[ colname ]
        leafs1 = set( map( list(SLT.col_ids).index, map( lambda x : T.leafs[x],  s[ s > 0 ].index ) ) )
        leafs2 = set( SLT.get_column_leafs(n, as_row_ids=True) )
        assert leafs1 == leafs2

def test_get_column_leafs_by_name_as_row_ids() :
    T = SuchTree( test_tree )
    links = pd.DataFrame( numpy.random.randint( 0, 3, size=(N,N) ),
                          columns=list(T.leafs.keys()), 
                          index=list(T.leafs.keys()) )
    SLT = SuchLinkedTrees( T, T, links )
    for colname in links.columns :
        s = links.map(bool)[ colname ]
        leafs1 = set( map( list(SLT.col_ids).index, map( lambda x : T.leafs[x],  s[ s > 0 ].index ) ) )
        leafs2 = set( SLT.get_column_leafs( colname, as_row_ids=True ) )
        assert leafs1 == leafs2

def test_get_column_links() :
    T = SuchTree( test_tree )
    row_names = list(T.leafs.keys())
    numpy.random.shuffle(row_names)
    links = pd.DataFrame( numpy.random.randint( 0, 3, size=(N,N) ),
                          columns=list(T.leafs.keys()), 
                          index=row_names )
    SLT = SuchLinkedTrees( T, T, links )
    for n,colname in enumerate( links.columns ) :
        s = links.map(bool)[ colname ]
        c = SLT.get_column_links(n)
        for m,rowname in enumerate( SLT.row_names ) :
            assert s[rowname] == c[m]
 
def test_linkmatrix_property() :
    T = SuchTree( test_tree )
    row_names = list(T.leafs.keys())
    numpy.random.shuffle(row_names)
    links = pd.DataFrame( numpy.random.randint( 0, 3, size=(N,N) ),
                          columns=list(T.leafs.keys()), 
                          index=row_names )
    SLT = SuchLinkedTrees( T, T, links )
    for col in SLT.col_names :
        for row in SLT.row_names :
            col_id = SLT.col_names.index(col)
            row_id = SLT.row_names.index(row)
            assert bool(links.T[row][col]) == SLT.linkmatrix[row_id][col_id]

def test_linklist_property() :
    T = SuchTree( test_tree )
    row_names = list(T.leafs.keys())
    numpy.random.shuffle(row_names)
    links = pd.DataFrame( numpy.random.randint( 0, 3, size=(N,N) ),
                          columns=list(T.leafs.keys()), 
                          index=row_names )
    SLT = SuchLinkedTrees( T, T, links )
    l = links.unstack()
    A = set(map( lambda x : (SLT.TreeB.leafs[x[0]], SLT.TreeA.leafs[x[1]]), 
        list( l[l>0].index ) ))
    B = set(map( lambda x : (x[0], x[1]),  SLT.linklist ) )
    assert A == B

def test_link_identities() :
    with tempfile.NamedTemporaryFile() as f1 :
        f1.file.write( b'(A:1,(B:1,(C:1,D:1)E:1)F:1)G:1;' )
        f1.file.close()
        T1 = SuchTree( f1.name )
    with tempfile.NamedTemporaryFile() as f2 :
        f2.file.write( b'((a:1,b:1)e:1,(c:1,d:1)f:1)g:1;' )
        f2.file.close()
        T2 = SuchTree( f2.name )
    
    ll = ( ('A','a'), ('B','c'), ('B','d'), ('C','d'), ('D','d') )
    
    links = pd.DataFrame( numpy.zeros( (4,4), dtype=int ), 
                          index=list(T1.leafs.keys()), 
                          columns=list(T2.leafs.keys()) )
    for i,j in ll :
            links.at[i,j] = 1
    
    SLT = SuchLinkedTrees( T1, T2, links )
    
    t1_sfeal = dict( zip( T1.leafs.values(), T1.leafs.keys() ) )
    t2_sfeal = dict( zip( T2.leafs.values(), T2.leafs.keys() ) )
    
    lll = set( (t1_sfeal[j], t2_sfeal[i] ) for i,j in SLT.linklist.tolist() )
    
    assert set(ll) == lll

# test subsetting

def test_subset_a() :
    T = SuchTree( test_tree )
    row_names = list(T.leafs.keys())
    numpy.random.shuffle(row_names)
    links = pd.DataFrame( numpy.random.randint( 0, 3, size=(N,N) ),
                          columns=list(T.leafs.keys()), 
                          index=row_names )
    SLT = SuchLinkedTrees( T, T, links )
    sfeal = dict( zip( SLT.TreeA.leafs.values(), SLT.TreeA.leafs.keys() ) )
    subset_links = links.loc[ list(map( lambda x: sfeal[x], SLT.TreeA.get_leafs(1) )) ]
    l = subset_links.unstack()
    SLT.subset_a(1)
    A = set(map( lambda x : (SLT.TreeB.leafs[x[0]], SLT.TreeA.leafs[x[1]]), 
        list( l[l>0].index ) ))
    B = set(map( lambda x : (x[0], x[1]),  SLT.linklist ) )
    assert A == B

def test_subset_b() :
    T = SuchTree( test_tree )
    row_names = list(T.leafs.keys())
    numpy.random.shuffle(row_names)
    links = pd.DataFrame( numpy.random.randint( 0, 3, size=(N,N) ),
                          columns=list(T.leafs.keys()), 
                          index=row_names )
    SLT = SuchLinkedTrees( T, T, links )
    sfeal = dict( zip( SLT.TreeB.leafs.values(), SLT.TreeB.leafs.keys() ) )
    subset_links = links[ list(map( lambda x: sfeal[x], SLT.TreeB.get_leafs(1) )) ]
    l = subset_links.unstack()
    SLT.subset_b(1)
    A = set(map( lambda x : (SLT.TreeB.leafs[x[0]], SLT.TreeA.leafs[x[1]]), 
        list( l[l>0].index ) ))
    B = set(map( lambda x : (x[0], x[1]),  SLT.linklist ) )
    assert A == B

# test igraph output
@pytest.mark.skipif(not has_igraph, reason="igraph not installed")
def test_to_igraph() :
    #Make sure the igraph output has correct same structure 
    
    T1 = SuchTree( gopher_tree )
    T2 = SuchTree( lice_tree   )
    links = pd.read_csv( gl_links, index_col=0 )
    
    SLT = SuchLinkedTrees( T1, T2, links )
    
    g = SLT.to_igraph()
    
    # igraph returns an unweighted adjacency matrix,
    # so we'll convert SuchLinkedTrees weighted
    # adjacency matrix to an unweighted form.
    saj = numpy.ceil( SLT.adjacency() )
    
    # For some reason, igraph invented its own Matrix
    # class that doesn't implement a standard numpy 
    # interface. :-/
    iaj = numpy.array( list( map( list, g.get_adjacency() ) ) )
    
    # matrixes must be the same shape
    assert saj.shape == iaj.shape
    
    # all matrix elements must be equal
    assert reduce( lambda a,b:a and b, (saj == iaj).flatten() )

