from nose.tools import assert_equal, assert_almost_equal
from SuchTree import SuchTree, SuchLinkedTrees, pearson
from dendropy import Tree
from itertools import combinations
import numpy

test_tree = 'SuchTree/tests/test.tree'
dpt = Tree.get( file=open(test_tree), schema='newick' )
for n,node in enumerate( dpt.inorder_node_iter() ) :
    node.label = n

def test_init_link_by_name() :
    T = SuchTree( test_tree )
    names_a = T.leafs.keys()
    names_b = T.leafs.keys()
    numpy.random.shuffle( names_b )
    links = zip( names_a, names_b )
    SLT = SuchLinkedTrees( T, T, links )
    assert_equal( type(SLT), SuchLinkedTrees )

def test_init_link_by_id() :
    T = SuchTree( test_tree )
    names_a = T.leafs.values()
    names_b = T.leafs.values()
    numpy.random.shuffle( names_b )
    links = numpy.array( zip( names_a, names_b ), dtype=int )
    SLT = SuchLinkedTrees( T, T, links )
    assert_equal( type(SLT), SuchLinkedTrees )

def test_init_one_tree_by_file() :
    T = SuchTree( test_tree )
    names_a = T.leafs.keys()
    names_b = T.leafs.keys()
    numpy.random.shuffle( names_b )
    links = zip( names_a, names_b )
    SLT = SuchLinkedTrees( test_tree, T, links )
    assert_equal( type(SLT), SuchLinkedTrees )

def test_init_both_trees_by_file() :
    T = SuchTree( test_tree )
    names_a = T.leafs.keys()
    names_b = T.leafs.keys()
    numpy.random.shuffle( names_b )
    links = zip( names_a, names_b )
    SLT = SuchLinkedTrees( test_tree, test_tree, links )
    assert_equal( type(SLT), SuchLinkedTrees )

