"""
Test suite for the new SuchTree API.

This test file validates all the new methods and properties introduced
in the API refactoring, using the existing test structure as a template.
"""

from __future__ import print_function

import pytest
from pytest import approx
import numpy as np
import pandas as pd
from itertools import combinations, chain
from collections import deque

from SuchTree import SuchTree

from SuchTree import SuchTreeError, NodeNotFoundError, InvalidNodeError, TreeStructureError

try:
    import networkx
    has_networkx = True
except ImportError:
    has_networkx = False

try:
    import igraph
    has_igraph = True
except ImportError:
    has_igraph = False

# Test data
test_tree_str = '(A,B,(C,D));'
test_tree = 'SuchTree/tests/test.tree'
support_tree_int = 'SuchTree/tests/support_int.tree'
support_tree_float = 'SuchTree/tests/support_float.tree'
support_tree_comment = 'SuchTree/tests/support_comment.tree'


class TestNewProperties:
    """Test new and renamed properties."""
    
    def setup_method(self):
        self.tree = SuchTree(test_tree)
        self.simple_tree = SuchTree(test_tree_str)
    
    def test_size_property(self):
        """Test size property (renamed from length)."""
        assert isinstance(self.tree.size, int)
        assert self.tree.size > 0
        assert self.tree.size == self.tree.length  # Backward compatibility
    
    def test_leaves_property(self):
        """Test leaves property (renamed from leafs)."""
        leaves = self.tree.leaves
        assert isinstance(leaves, dict)
        assert all(isinstance(k, str) for k in leaves.keys())
        assert all(isinstance(v, int) for v in leaves.values())
        assert leaves == self.tree.leafs  # Backward compatibility
    
    def test_leaf_nodes_property(self):
        """Test leaf_nodes property (renamed from leafnodes)."""
        leaf_nodes = self.tree.leaf_nodes
        assert isinstance(leaf_nodes, dict)
        assert all(isinstance(k, int) for k in leaf_nodes.keys())
        assert all(isinstance(v, str) for v in leaf_nodes.values())
        assert leaf_nodes == self.tree.leafnodes  # Backward compatibility
    
    def test_num_leaves_property(self):
        """Test num_leaves property (renamed from n_leafs)."""
        assert isinstance(self.tree.num_leaves, int)
        assert self.tree.num_leaves > 0
        assert self.tree.num_leaves == self.tree.n_leafs  # Backward compatibility
    
    def test_root_node_property(self):
        """Test root_node property (renamed from root)."""
        assert isinstance(self.tree.root_node, int)
        assert self.tree.root_node == self.tree.root  # Backward compatibility
    
    def test_polytomy_epsilon_property(self):
        """Test polytomy_epsilon property (renamed from polytomy_distance)."""
        epsilon = self.tree.polytomy_epsilon
        assert isinstance(epsilon, float)
        assert epsilon > 0
        assert epsilon == self.tree.polytomy_distance  # Backward compatibility
        
        # Test setter
        new_epsilon = 1e-10
        self.tree.polytomy_epsilon = new_epsilon
        assert self.tree.polytomy_epsilon == new_epsilon
    
    def test_internal_nodes_property(self):
        """Test new internal_nodes property."""
        internal_nodes = self.tree.internal_nodes
        assert isinstance(internal_nodes, np.ndarray)
        assert len(internal_nodes) > 0
        assert all(not self.tree.is_leaf(nid) for nid in internal_nodes)
    
    def test_all_nodes_property(self):
        """Test new all_nodes property."""
        all_nodes = self.tree.all_nodes
        assert isinstance(all_nodes, np.ndarray)
        assert len(all_nodes) == self.tree.size
    
    def test_leaf_node_ids_property(self):
        """Test new leaf_node_ids property."""
        leaf_ids = self.tree.leaf_node_ids
        assert isinstance(leaf_ids, np.ndarray)
        assert len(leaf_ids) == self.tree.num_leaves
        assert set(leaf_ids) == set(self.tree.leaves.values())
    
    def test_leaf_names_property(self):
        """Test new leaf_names property."""
        names = self.tree.leaf_names
        assert isinstance(names, list)
        assert len(names) == self.tree.num_leaves
        assert set(names) == set(self.tree.leaves.keys())


class TestValidationMethods:
    """Test the new validation helper methods."""
    
    def setup_method(self):
        self.tree = SuchTree(test_tree)
    
    def test_validate_node_with_valid_id(self):
        """Test _validate_node with valid node ID."""
        node_id = list(self.tree.leaves.values())[0]
        validated = self.tree._validate_node(node_id)
        assert validated == node_id
    
    def test_validate_node_with_valid_name(self):
        """Test _validate_node with valid leaf name."""
        leaf_name = list(self.tree.leaves.keys())[0]
        validated = self.tree._validate_node(leaf_name)
        assert validated == self.tree.leaves[leaf_name]
    
    def test_validate_node_with_invalid_name(self):
        """Test _validate_node with invalid leaf name."""
        with pytest.raises(NodeNotFoundError):
            self.tree._validate_node("invalid_name")
    
    def test_validate_node_with_invalid_id(self):
        """Test _validate_node with out of bounds node ID."""
        with pytest.raises(InvalidNodeError):
            self.tree._validate_node(-1)
        
        with pytest.raises(InvalidNodeError):
            self.tree._validate_node(self.tree.size + 100)
    
    def test_validate_node_with_invalid_type(self):
        """Test _validate_node with invalid type."""
        with pytest.raises(TypeError):
            self.tree._validate_node(3.14)
        
        with pytest.raises(TypeError):
            self.tree._validate_node(['invalid'])


class TestNodeQueryMethods:
    """Test refactored node query methods."""
    
    def setup_method(self):
        self.tree = SuchTree(test_tree)
        self.simple_tree = SuchTree(test_tree_str)
    
    def test_get_parent_with_id(self):
        """Test get_parent with node ID."""
        leaf_id = list(self.tree.leaves.values())[0]
        parent_id = self.tree.get_parent(leaf_id)
        assert isinstance(parent_id, int)
        assert parent_id != leaf_id or parent_id == -1  # Root has parent -1
    
    def test_get_parent_with_name(self):
        """Test get_parent with leaf name."""
        leaf_name = list(self.tree.leaves.keys())[0]
        parent_id = self.tree.get_parent(leaf_name)
        assert isinstance(parent_id, int)
        
        # Should be same as using node ID
        leaf_id = self.tree.leaves[leaf_name]
        assert parent_id == self.tree.get_parent(leaf_id)
    
    def test_get_children_consistency(self):
        """Test get_children maintains consistency with original."""
        for node_id in self.tree.all_nodes:
            left, right = self.tree.get_children(node_id)
            # Backward compatibility
            assert (left, right) == self.tree.get_children(node_id)
    
    def test_get_ancestors(self):
        """Test get_ancestors (renamed from get_lineage)."""
        leaf_name = list(self.tree.leaves.keys())[0]
        ancestors = list(self.tree.get_ancestors(leaf_name))
        
        # Each ancestor should be ancestor of the leaf
        for ancestor in ancestors:
            assert self.tree.is_ancestor(ancestor, leaf_name) == 1
        
        # Should be same as old get_lineage
        old_lineage = list(self.tree.get_lineage(leaf_name))
        assert ancestors == old_lineage
    
    def test_get_descendants(self):
        """Test get_descendants (renamed from get_descendant_nodes)."""
        root = self.tree.root_node
        descendants = set(self.tree.get_descendants(root))
        
        # Should include all nodes
        assert descendants == set(self.tree.all_nodes)
        
        # Should be same as old method
        old_descendants = set(self.tree.get_descendant_nodes(root))
        assert descendants == old_descendants
    
    def test_get_leaves(self):
        """Test get_leaves (renamed from get_leafs)."""
        root = self.tree.root_node
        leaves = self.tree.get_leaves(root)
        
        assert isinstance(leaves, np.ndarray)
        assert set(leaves) == set(self.tree.leaves.values())
        
        # Should be same as old method
        old_leaves = self.tree.get_leafs(root)
        np.testing.assert_array_equal(leaves, old_leaves)
    
    def test_get_support_with_support_values(self):
        """Test get_support with trees that have support values."""
        support_tree = SuchTree(support_tree_int)
        for node_id in support_tree.internal_nodes:
            # skip nodes that can't have a support value
            if node_id == support_tree.root_node : continue
            parent = support_tree.get_parent( node_id )
            distance = support_tree.distance( node_id, parent )
            if distance == pytest.approx( support_tree.polytomy_epsilon ) : continue

            support = support_tree.get_support(node_id)
            assert isinstance(support, float)
            # Internal nodes should have support values
            assert support != -1.0 
    
    def test_get_internal_nodes_consistency(self):
        """Test get_internal_nodes maintains consistency."""
        internal_nodes = self.tree.get_internal_nodes()
        assert isinstance(internal_nodes, np.ndarray)
        assert all(not self.tree.is_leaf(nid) for nid in internal_nodes)
        
        # Should match property
        np.testing.assert_array_equal(
            np.sort(internal_nodes),
            np.sort(self.tree.internal_nodes)
        )


class TestNodeTestMethods:
    """Test refactored node test methods."""
    
    def setup_method(self):
        self.tree = SuchTree(test_tree)
        self.simple_tree = SuchTree(test_tree_str)
    
    def test_is_leaf_consistency(self):
        """Test is_leaf maintains consistency."""
        for leaf_id in self.tree.leaves.values():
            assert self.tree.is_leaf(leaf_id)
        
        for internal_id in self.tree.internal_nodes:
            assert not self.tree.is_leaf(internal_id)
    
    def test_is_internal(self):
        """Test is_internal (renamed from is_internal_node)."""
        for leaf_id in self.tree.leaves.values():
            assert not self.tree.is_internal(leaf_id)
        
        for internal_id in self.tree.internal_nodes:
            assert self.tree.is_internal(internal_id)
        
        # Should be same as old method
        for node_id in self.tree.all_nodes:
            assert self.tree.is_internal(node_id) == self.tree.is_internal_node(node_id)
    
    def test_is_ancestor_consistency(self):
        """Test is_ancestor maintains consistency."""
        root = self.tree.root_node
        
        # Root should be ancestor of all other nodes
        for node_id in self.tree.all_nodes:
            if node_id != root:
                assert self.tree.is_ancestor(root, node_id) == 1
                assert self.tree.is_ancestor(node_id, root) == -1
    
    def test_is_descendant(self):
        """Test new is_descendant method."""
        root = self.tree.root_node
        
        for node_id in self.tree.all_nodes:
            if node_id != root:
                # Should be complement of is_ancestor
                assert self.tree.is_descendant(node_id, root)
                assert not self.tree.is_descendant(root, node_id)
    
    def test_is_root(self):
        """Test new is_root method."""
        root = self.tree.root_node
        assert self.tree.is_root(root)
        
        for node_id in self.tree.all_nodes:
            if node_id != root:
                assert not self.tree.is_root(node_id)
    
    def test_is_sibling(self):
        """Test new is_sibling method."""
        # In simple tree (A,B,(C,D)), A and the internal node (C,D) are siblings
        simple = SuchTree(test_tree_str)
        
        # Find siblings by checking same parent
        node_pairs_with_same_parent = []
        for node1 in simple.all_nodes:
            for node2 in simple.all_nodes:
                if (node1 != node2 and 
                    simple.get_parent(node1) == simple.get_parent(node2) and
                    simple.get_parent(node1) != -1):
                    node_pairs_with_same_parent.append((node1, node2))
        
        # Test siblings
        for node1, node2 in node_pairs_with_same_parent:
            assert simple.is_sibling(node1, node2)
    
    def test_has_children(self):
        """Test new has_children method."""
        for node_id in self.tree.all_nodes:
            has_children = self.tree.has_children(node_id)
            is_internal = self.tree.is_internal(node_id)
            assert has_children == is_internal
    
    def test_has_parent(self):
        """Test new has_parent method."""
        root = self.tree.root_node
        assert not self.tree.has_parent(root)
        
        for node_id in self.tree.all_nodes:
            if node_id != root:
                assert self.tree.has_parent(node_id)


class TestDistanceMethods:
    """Test refactored distance methods."""
    
    def setup_method(self):
        self.tree = SuchTree(test_tree)
    
    def test_distance_to_root(self):
        """Test distance_to_root (renamed from get_distance_to_root)."""
        # Test with leaf names
        for leaf_name in self.tree.leaves.keys():
            dist_new = self.tree.distance_to_root(leaf_name)
            dist_old = self.tree.get_distance_to_root(leaf_name)
            assert dist_new == approx(dist_old, abs=0.001)
    
    def test_distance_consistency(self):
        """Test distance method maintains consistency."""
        # Compare with test matrix
        with open('SuchTree/tests/test.matrix') as f:
            for line in f:
                a, b, expected_dist = line.strip().split()
                expected_dist = float(expected_dist)
                actual_dist = self.tree.distance(a, b)
                assert actual_dist == approx(expected_dist, abs=0.001)
    
    def test_distances_bulk(self):
        """Test distances_bulk (renamed from distances)."""
        # Prepare test data
        pairs = []
        expected = []
        with open('SuchTree/tests/test.matrix') as f:
            for line in f:
                a, b, dist = line.strip().split()
                pairs.append((self.tree.leaves[a], self.tree.leaves[b]))
                expected.append(float(dist))
        
        pairs_array = np.array(pairs, dtype=np.int64)
        result = self.tree.distances_bulk(pairs_array)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(expected)
        
        for exp, res in zip(expected, result):
            assert res == approx(exp, abs=0.001)
    
    def test_distances_by_name_consistency(self):
        """Test distances_by_name maintains consistency."""
        pairs = []
        expected = []
        with open('SuchTree/tests/test.matrix') as f:
            for line in f:
                a, b, dist = line.strip().split()
                pairs.append((a, b))
                expected.append(float(dist))
        
        result = self.tree.distances_by_name(pairs)
        
        assert isinstance(result, list)
        assert len(result) == len(expected)
        
        for exp, res in zip(expected, result):
            assert res == approx(exp, abs=0.001)
    
    def test_pairwise_distances(self):
        """Test new pairwise_distances method."""
        # Test with subset of leaves
        leaf_names = list(self.tree.leaves.keys())[:5]
        dist_matrix = self.tree.pairwise_distances(leaf_names)
        
        assert isinstance(dist_matrix, np.ndarray)
        assert dist_matrix.shape == (len(leaf_names), len(leaf_names))
        
        # Should be symmetric
        np.testing.assert_array_almost_equal(dist_matrix, dist_matrix.T)
        
        # Diagonal should be zero
        np.testing.assert_array_almost_equal(np.diag(dist_matrix), np.zeros(len(leaf_names)))
    
    def test_nearest_neighbors(self):
        """Test new nearest_neighbors method."""
        leaf_names = list(self.tree.leaves.keys())
        query_leaf = leaf_names[0]
        
        neighbors = self.tree.nearest_neighbors(query_leaf, k=3)
        
        assert isinstance(neighbors, list)
        assert len(neighbors) == 3
        
        # Each neighbor should be (name, distance) tuple
        for neighbor, distance in neighbors:
            assert isinstance(neighbor, str)
            assert isinstance(distance, float)
            assert distance >= 0
            assert neighbor != query_leaf
        
        # Distances should be in ascending order
        distances = [dist for _, dist in neighbors]
        assert distances == sorted(distances)


class TestTopologyMethods:
    """Test refactored topology methods."""
    
    def setup_method(self):
        self.tree = SuchTree(test_tree)
        self.simple_tree = SuchTree(test_tree_str)
    
    def test_common_ancestor(self):
        """Test common_ancestor (renamed from mrca)."""
        # Test with leaf names
        leaves = list(self.tree.leaves.keys())
        for a, b in combinations(leaves[:5], 2):
            mrca_new = self.tree.common_ancestor(a, b)
            mrca_old = self.tree.mrca(a, b)
            assert mrca_new == mrca_old
            
            # Should be symmetric
            assert self.tree.common_ancestor(a, b) == self.tree.common_ancestor(b, a)
            
            # MRCA should be ancestor of both nodes
            assert self.tree.is_ancestor(mrca_new, a) == 1
            assert self.tree.is_ancestor(mrca_new, b) == 1
    
    def test_bipartition(self):
        """Test bipartition (renamed from get_bipartition)."""
        for internal_node in self.tree.internal_nodes:
            # Test by names (default)
            bipart_names = self.tree.bipartition(internal_node, by_id=False)
            old_bipart_names = self.tree.get_bipartition(internal_node, by_id=False)
            assert bipart_names == old_bipart_names
            
            # Test by IDs
            bipart_ids = self.tree.bipartition(internal_node, by_id=True)
            old_bipart_ids = self.tree.get_bipartition(internal_node, by_id=True)
            assert bipart_ids == old_bipart_ids
            
            # Should be frozenset of two frozensets
            assert isinstance(bipart_names, frozenset)
            assert len(bipart_names) == 2
            assert all(isinstance(part, frozenset) for part in bipart_names)
    
    def test_bipartitions_generator(self):
        """Test bipartitions generator."""
        biparts_new = list(self.tree.bipartitions(by_id=False))
        biparts_old = list(self.tree.bipartitions(by_id=False))
        
        assert len(biparts_new) == len(biparts_old)
        assert len(biparts_new) == len(self.tree.internal_nodes)
    
    def test_quartet_topology(self):
        """Test quartet_topology (renamed from get_quartet_topology)."""
        # Test with simple tree where we know the topology
        simple = SuchTree('(A,B,(C,D));')
        topology = simple.quartet_topology('A', 'B', 'C', 'D')
        expected = frozenset((
            frozenset(('A', 'B')),
            frozenset(('C', 'D'))
        ))
        assert topology == expected
        
        # Should be same as old method
        old_topology = simple.get_quartet_topology('A', 'B', 'C', 'D')
        assert topology == old_topology
    
    def test_quartet_topologies_bulk(self):
        """Test quartet_topologies_bulk (renamed from quartet_topologies)."""
        if self.tree.num_leaves >= 4:
            # Create array of quartets
            leaf_ids = list(self.tree.leaves.values())
            quartets = np.array(list(combinations(leaf_ids, 4))[:5], dtype=np.int64)
            
            topologies_new = self.tree.quartet_topologies_bulk(quartets)
            topologies_old = self.tree.quartet_topologies(quartets)
            
            np.testing.assert_array_equal(topologies_new, topologies_old)
            assert topologies_new.shape == (len(quartets), 4)
    
    def test_quartet_topologies_by_name(self):
        """Test quartet_topologies_by_name method."""
        if self.tree.num_leaves >= 4:
            leaf_names = list(self.tree.leaves.keys())
            quartets = [tuple(combo) for combo in combinations(leaf_names, 4)][:3]
            
            topologies = self.tree.quartet_topologies_by_name(quartets)
            
            assert isinstance(topologies, list)
            assert len(topologies) == len(quartets)
            
            for topology in topologies:
                assert isinstance(topology, frozenset)
                assert len(topology) == 2
    
    def test_path_between_nodes(self):
        """Test new path_between_nodes method."""
        leaves = list(self.tree.leaves.keys())[:2]
        path = self.tree.path_between_nodes(leaves[0], leaves[1])
        
        assert isinstance(path, list)
        assert len(path) >= 2  # At least the two nodes
        assert all(isinstance(node_id, int) for node_id in path)
        
        # Path should start and end with the specified nodes
        leaf1_id = self.tree.leaves[leaves[0]]
        leaf2_id = self.tree.leaves[leaves[1]]
        assert path[0] == leaf1_id
        assert path[-1] == leaf2_id


class TestTraversalMethods:
    """Test refactored traversal methods."""
    
    def setup_method(self):
        self.tree = SuchTree(test_tree)
        self.simple_tree = SuchTree(test_tree_str)
    
    def test_traverse_inorder(self):
        """Test traverse_inorder (renamed from in_order)."""
        # With distances
        traversal_with_dist = list(self.tree.traverse_inorder(include_distances=True))
        old_traversal = list(self.tree.in_order(distances=True))
        
        assert len(traversal_with_dist) == len(old_traversal)
        assert len(traversal_with_dist) == self.tree.size
        
        # Without distances
        traversal_no_dist = list(self.tree.traverse_inorder(include_distances=False))
        assert len(traversal_no_dist) == self.tree.size
        assert all(isinstance(node_id, int) for node_id in traversal_no_dist)
    
    def test_traverse_preorder(self):
        """Test traverse_preorder (renamed from pre_order)."""
        traversal_new = list(self.tree.traverse_preorder())
        traversal_old = list(self.tree.pre_order())
        
        assert traversal_new == traversal_old
        assert len(traversal_new) == self.tree.size
        
        # Should start with root
        assert traversal_new[0] == self.tree.root_node
    
    def test_traverse_postorder(self):
        """Test new traverse_postorder method."""
        traversal = list(self.tree.traverse_postorder())
        
        assert len(traversal) == self.tree.size
        assert all(isinstance(node_id, int) for node_id in traversal)
        
        # Root should be last in postorder
        assert traversal[-1] == self.tree.root_node
    
    def test_traverse_levelorder(self):
        """Test new traverse_levelorder method."""
        traversal = list(self.tree.traverse_levelorder())
        
        assert len(traversal) == self.tree.size
        assert all(isinstance(node_id, int) for node_id in traversal)
        
        # Root should be first in level order
        assert traversal[0] == self.tree.root_node
    
    def test_traverse_leaves_only(self):
        """Test new traverse_leaves_only method."""
        leaf_traversal = list(self.tree.traverse_leaves_only())
        
        assert len(leaf_traversal) == self.tree.num_leaves
        assert set(leaf_traversal) == set(self.tree.leaves.values())
        assert all(self.tree.is_leaf(nid) for nid in leaf_traversal)
    
    def test_traverse_internal_only(self):
        """Test new traverse_internal_only method."""
        internal_traversal = list(self.tree.traverse_internal_only())
        
        assert len(internal_traversal) == len(self.tree.internal_nodes)
        assert set(internal_traversal) == set(self.tree.internal_nodes)
        assert all(self.tree.is_internal(nid) for nid in internal_traversal)
    
    def test_traverse_with_depth(self):
        """Test new traverse_with_depth method."""
        depth_traversal = list(self.tree.traverse_with_depth())
        
        assert len(depth_traversal) == self.tree.size
        
        for node_id, depth in depth_traversal:
            assert isinstance(node_id, int)
            assert isinstance(depth, int)
            assert depth >= 0
        
        # Root should have depth 0
        root_depth = next(depth for nid, depth in depth_traversal if nid == self.tree.root_node)
        assert root_depth == 0
    
    def test_traverse_with_distances(self):
        """Test new traverse_with_distances method."""
        dist_traversal = list(self.tree.traverse_with_distances())
        
        assert len(dist_traversal) == self.tree.size
        
        for node_id, dist_to_parent, dist_to_root in dist_traversal:
            assert isinstance(node_id, int)
            assert isinstance(dist_to_parent, float)
            assert isinstance(dist_to_root, float)
            assert dist_to_root >= 0


class TestGraphMatrixMethods:
    """Test refactored graph and matrix methods."""
    
    def setup_method(self):
        self.tree = SuchTree(test_tree)
    
    def test_adjacency_matrix(self):
        """Test adjacency_matrix (renamed from adjacency)."""
        result_new = self.tree.adjacency_matrix()
        result_old = self.tree.adjacency()
        
        # Should have same structure
        assert set(result_new.keys()) == set(result_old.keys())
        assert 'adjacency_matrix' in result_new
        assert 'node_ids' in result_new
        
        # Matrices should be same
        np.testing.assert_array_almost_equal(
            result_new['adjacency_matrix'],
            result_old['adjacency_matrix']
        )
    
    def test_laplacian_matrix(self):
        """Test laplacian_matrix (renamed from laplacian)."""
        result_new = self.tree.laplacian_matrix()
        result_old = self.tree.laplacian()
        
        # Should have same structure
        assert set(result_new.keys()) == set(result_old.keys())
        
        # Matrices should be same
        np.testing.assert_array_almost_equal(
            result_new['laplacian'],
            result_old['laplacian']
        )
    
    def test_incidence_matrix(self):
        """Test new incidence_matrix method."""
        result = self.tree.incidence_matrix()
        
        assert 'incidence_matrix' in result
        assert 'node_ids' in result
        assert 'edge_list' in result
        
        incidence = result['incidence_matrix']
        node_ids = result['node_ids']
        edges = result['edge_list']
        
        # Should be nodes × edges matrix
        assert incidence.shape[0] == len(node_ids)
        assert incidence.shape[1] == len(edges)
        
        # Each edge should connect exactly two nodes
        for col in range(incidence.shape[1]):
            nonzero_count = np.count_nonzero(incidence[:, col])
            assert nonzero_count == 2  # One +1 and one -1
    
    def test_distance_matrix(self):
        """Test new distance_matrix method."""
        # Test with subset of leaves
        leaf_names = list(self.tree.leaves.keys())[:5]
        result = self.tree.distance_matrix(leaf_names)
        
        assert 'distance_matrix' in result
        assert 'node_ids' in result
        assert 'node_names' in result
        
        dist_matrix = result['distance_matrix']
        
        # Should be symmetric
        np.testing.assert_array_almost_equal(dist_matrix, dist_matrix.T)
        
        # Diagonal should be zero
        np.testing.assert_array_almost_equal(
            np.diag(dist_matrix), 
            np.zeros(len(leaf_names))
        )
    
    def test_degree_sequence(self):
        """Test new degree_sequence method."""
        result = self.tree.degree_sequence()
        
        assert 'degrees' in result
        assert 'node_ids' in result
        assert 'max_degree' in result
        assert 'min_degree' in result
        
        degrees = result['degrees']
        
        # Degrees should be positive integers
        assert all(deg >= 0 for deg in degrees)
        assert result['max_degree'] == degrees.max()
        assert result['min_degree'] == degrees.min()


class TestExportIntegrationMethods:
    """Test refactored export and integration methods."""
    
    def setup_method(self):
        self.tree = SuchTree(test_tree)
    
    def test_to_networkx_nodes(self):
        """Test to_networkx_nodes (renamed from nodes_data)."""
        nodes_new = list(self.tree.to_networkx_nodes())
        nodes_old = list(self.tree.nodes_data())
        
        assert len(nodes_new) == len(nodes_old)
        assert len(nodes_new) == self.tree.size
        
        # Each should be (node_id, attributes) tuple
        for node_id, attrs in nodes_new:
            assert isinstance(node_id, int)
            assert isinstance(attrs, dict)
            assert 'type' in attrs
            assert attrs['type'] in ['leaf', 'internal']
    
    def test_to_networkx_edges(self):
        """Test to_networkx_edges (renamed from edges_data)."""
        edges_new = list(self.tree.to_networkx_edges())
        edges_old = list(self.tree.edges_data())
        
        assert len(edges_new) == len(edges_old)
        
        # Each should be (child, parent, attributes) tuple
        for child_id, parent_id, attrs in edges_new:
            assert isinstance(child_id, int)
            assert isinstance(parent_id, int)
            assert isinstance(attrs, dict)
            assert 'weight' in attrs
    
    @pytest.mark.skipif(not has_networkx, reason="networkx not installed")
    def test_to_networkx_graph(self):
        """Test new to_networkx_graph method."""
        import networkx as nx
        
        graph = self.tree.to_networkx_graph()
        
        assert isinstance(graph, nx.Graph)
        assert len(graph.nodes) == self.tree.size
        assert len(graph.edges) == self.tree.size - 1  # Tree has n-1 edges
    
    def test_to_newick(self):
        """Test new to_newick method."""
        newick_str = self.tree.to_newick()
        
        assert isinstance(newick_str, str)
        assert newick_str.endswith(';')
        assert '(' in newick_str and ')' in newick_str
        
        # Should be parseable by SuchTree
        reconstructed = SuchTree(newick_str)
        assert reconstructed.num_leaves == self.tree.num_leaves
    
class TestDeprecationWarnings:
    """Test that deprecated methods emit appropriate warnings."""
    
    def setup_method(self):
        self.tree = SuchTree(test_tree)
    
    def test_deprecated_properties(self):
        """Test that deprecated properties emit warnings."""
        with pytest.warns(DeprecationWarning):
            _ = self.tree.length
        
        with pytest.warns(DeprecationWarning):
            _ = self.tree.leafs
        
        with pytest.warns(DeprecationWarning):
            _ = self.tree.leafnodes
    
    def test_deprecated_methods(self):
        """Test that deprecated methods emit warnings."""
        leaf_name = list(self.tree.leaves.keys())[0]
        
        with pytest.warns(DeprecationWarning):
            _ = self.tree.get_distance_to_root(leaf_name)
        
        with pytest.warns(DeprecationWarning):
            _ = self.tree.get_lineage(leaf_name)
        
        with pytest.warns(DeprecationWarning):
            _ = self.tree.mrca(leaf_name, leaf_name)


class TestErrorHandling:
    """Test proper error handling with custom exceptions."""
    
    def setup_method(self):
        self.tree = SuchTree(test_tree)
    
    def test_node_not_found_error(self):
        """Test NodeNotFoundError is raised appropriately."""
        with pytest.raises(NodeNotFoundError):
            self.tree.distance_to_root("nonexistent_leaf")
        
        with pytest.raises(NodeNotFoundError):
            self.tree.common_ancestor("leaf1", "nonexistent_leaf")
    
    def test_invalid_node_error(self):
        """Test InvalidNodeError is raised appropriately."""
        with pytest.raises(InvalidNodeError):
            self.tree.distance_to_root(-1)
        
        with pytest.raises(InvalidNodeError):
            self.tree.distance_to_root(self.tree.size + 100)
    
    def test_value_error_for_malformed_inputs(self):
        """Test ValueError for malformed inputs."""
        with pytest.raises(ValueError):
            # Wrong shape for distances_bulk
            self.tree.distances_bulk(np.array([[1, 2, 3]]))  # Should be (n,2)
    
    def test_type_error_for_wrong_types(self):
        """Test TypeError for wrong input types."""
        with pytest.raises(TypeError):
            self.tree.distance_to_root(3.14)  # Float not allowed


# Integration tests using both old and new APIs
class TestBackwardCompatibility:
    """Test that old and new APIs produce identical results."""
    
    def setup_method(self):
        self.tree = SuchTree(test_tree)
    
    def test_property_equivalence(self):
        """Test that new properties match old ones."""
        assert self.tree.size == self.tree.length
        assert self.tree.leaves == self.tree.leafs
        assert self.tree.leaf_nodes == self.tree.leafnodes
        assert self.tree.num_leaves == self.tree.n_leafs
        assert self.tree.root_node == self.tree.root
    
    def test_method_equivalence(self):
        """Test that new methods match old ones."""
        leaf_name = list(self.tree.leaves.keys())[0]
        
        # Suppress warnings for this test
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Distance methods
            assert (self.tree.distance_to_root(leaf_name) == 
                    self.tree.get_distance_to_root(leaf_name))
            
            # Topology methods
            leaves = list(self.tree.leaves.keys())
            if len(leaves) >= 2:
                assert (self.tree.common_ancestor(leaves[0], leaves[1]) ==
                        self.tree.mrca(leaves[0], leaves[1]))
            
            # Traversal methods
            traversal_new = list(self.tree.traverse_preorder())
            traversal_old = list(self.tree.pre_order())
            assert traversal_new == traversal_old


if __name__ == "__main__":
    pytest.main([__file__])
