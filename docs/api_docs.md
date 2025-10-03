---
title: SuchTree's API
subtitle: A somewhat consice description of the core features of SuchTree.
icon: material/file-document-outline
---

# SuchTree Class API Reference

## Overview
The `SuchTree` class provides high-performance phylogenetic tree manipulation using Cython. It supports:
- Fast tree traversal and node queries
- Patristic distance calculations
- Topological analysis
- Multiple tree formats (Newick, URL, file path)
- Integration with NetworkX and igraph

## Initialization
```python
class SuchTree(tree_input: Union[str, Path])
```
Construct from:
- Newick string
- File path
- URL (http/https/ftp)

Example:
```python
tree = SuchTree("(A:0.1,B:0.2,(C:0.3,D:0.4));")
tree = SuchTree("https://example.com/tree.newick")
```

## Core Properties

### Tree Structure
| Property | Type | Description | Example |
|----------|------|-------------|---------|
| `size` | `int` | Total nodes | `tree.size` → 7 |
| `depth` | `int` | Max depth | `tree.depth` → 3 |
| `num_leaves` | `int` | Leaf count | `tree.num_leaves` → 4 |
| `root_node` | `int` | Root ID | `tree.root_node` → 0 |
| `polytomy_epsilon` | `float` | Polytomy resolution | `1e-20` |

### Node Collections
| Property | Type | Description | Contains |
|----------|------|-------------|----------|
| `leaves` | `Dict[str, int]` | Name → ID | `{'A': 1, 'B': 2}` |
| `leaf_nodes` | `Dict[int, str]` | ID → Name | `{1: 'A', 2: 'B'}` |
| `internal_nodes` | `np.ndarray` | Internal IDs | `[0, 3, 4]` |
| `all_nodes` | `np.ndarray` | All IDs | `[0, 1, 2, 3, 4, 5, 6]` |
| `leaf_node_ids` | `np.ndarray` | Leaf IDs | `[1, 2, 5, 6]` |
| `leaf_names` | `list` | Leaf names | `['A', 'B', 'C', 'D']` |

## Core Methods

### Node Relationships
```python
get_parent(node: Union[int, str]) -> int
```
Get immediate parent node for a given node.

Args:
    node: Node identifier as either integer ID or leaf name string
    
Returns:
    Integer ID of parent node. Returns -1 if called on root node.
    
Raises:
    NodeNotFoundError: If leaf name doesn't exist in the tree
    InvalidNodeError: If node ID is out of valid range (0 <= id < tree.size)

Example:
```python
parent_id = tree.get_parent("A")
parent_id = tree.get_parent(5)
```

```python
get_children(node: Union[int, str]) -> Tuple[int, int]
```
Get direct children of a node. 

Args:
    node: Node identifier as either integer ID or leaf name string
    
Returns:
    Tuple of (left_child, right_child) node IDs. Returns (-1, -1) for leaf nodes.
    
Raises:
    NodeNotFoundError: If leaf name doesn't exist
    InvalidNodeError: If node ID is invalid

Note:
    For multifurcating trees, only the first two children are returned. Use 
    `traverse_children()` method for complete child iteration.

```python
get_ancestors(node: Union[int, str]) -> Generator[int, None, None]
```
Generate ancestor node IDs from node to root. Yields parent IDs in ascending order from immediate parent to root.

```python
get_descendants(node_id: int) -> Generator[int, None, None]
```
Generate all descendant node IDs in depth-first order. Includes the starting node in the output.

```python
get_leaves(node: Union[int, str]) -> np.ndarray
```
Get array of leaf node IDs descended from a given node. Uses efficient buffer reuse for performance.

```python
get_support(node: Union[int, str]) -> float
```
Retrieve node support value. Returns -1 if no support available. Works for both internal nodes and leaves.

### Tree Navigation
```python
is_leaf(node: Union[int, str]) -> bool
```
Check if node is a leaf. Uses optimized Cython implementation for fast checking.

```python
is_internal(node: Union[int, str]) -> bool
```
Check if node is internal. Simply returns negation of `is_leaf` but provides clearer intent.

```python
is_ancestor(ancestor: Union[int, str], descendant: Union[int, str]) -> int
```
Test ancestral relationship. Returns:  
- `1` if ancestor of descendant  
- `-1` if descendant is ancestor  
- `0` if no direct relationship

```python
is_descendant(descendant: Union[int, str], ancestor: Union[int, str]) -> bool
```
Convenience method that returns True if descendant is indeed a descendant of ancestor.

```python
is_root(node: Union[int, str]) -> bool
```
Check if node is the tree root. Uses direct comparison with stored root node ID.

```python
is_sibling(node1: Union[int, str], node2: Union[int, str]) -> bool
```
Check if two nodes share the same parent. Automatically returns False if either node is root.

```python
has_children(node: Union[int, str]) -> bool
```
Determine if node has any children. Equivalent to `is_internal` but may be more intuitive for some users.

```python
has_parent(node: Union[int, str]) -> bool
```
Check if node has a parent (i.e., is not root). Returns negation of `is_root`.

```python
common_ancestor(a: Union[int, str], b: Union[int, str]) -> int
```
Find most recent common ancestor of two nodes. Uses optimized MRCA algorithm with visited node tracking.

```python
path_between_nodes(a: Union[int, str], b: Union[int, str]) -> List[int]
```
Get node IDs forming the path between two nodes through their common ancestor. Returns list from a -> MRCA -> b.

### Distance Analysis
```python
distance(a: Union[int, str], b: Union[int, str]) -> float
```
Calculate patristic distance between two nodes along the tree.

Args:
    a: First node identifier (ID or name)
    b: Second node identifier (ID or name)
    
Returns:
    Sum of branch lengths along the path between nodes via their most recent 
    common ancestor (MRCA)
    
Raises:
    NodeNotFoundError: If either node name doesn't exist
    InvalidNodeError: If either node ID is invalid

Complexity:
    O(h) where h is the height of the tree. Uses cached ancestor paths for
    optimal performance.

Example:
```python
dist = tree.distance("A", "B")
dist = tree.distance(2, 5)
```

```python
distance_to_root(node: Union[int, str]) -> float
```
Calculate total branch length from node to root. Optimized with cumulative distance caching.

```python
distances_bulk(pairs: np.ndarray) -> np.ndarray
```
Efficiently compute distances for multiple node pairs. Accepts (n, 2) array of node IDs. Uses Cython nogil implementation.

```python
distances_by_name(pairs: List[Tuple[str, str]]) -> List[float]
```
Convenience wrapper for bulk distance calculation using leaf names instead of IDs.

```python
pairwise_distances(nodes: list = None) -> np.ndarray
```
Generate full distance matrix for specified nodes (all leaves by default). Returns symmetric numpy array.

```python
nearest_neighbors(node: Union[int, str], k=1) -> List[Tuple[Union[int, str], float]]
```
Find k nearest neighbors to a node. Can search among specific nodes or all leaves by default.

### Tree Traversal
```python
traverse_inorder(include_distances: bool = True) -> Generator
```
In-order traversal (left, root, right). Yields node IDs or (ID, distance) tuples.

```python
traverse_preorder(from_node: Union[int, str] = None) -> Generator
"""
Iterate through nodes in pre-order traversal (parent before children).

Args:
    from_node: Starting node (default: root). Can be ID or name.
    
Yields:
    Node IDs in traversal order
    
Raises:
    NodeNotFoundError: If from_node name doesn't exist
    InvalidNodeError: If from_node ID is invalid

Memory:
    O(h) space complexity due to stack implementation, where h is tree height

Example:
```python
for node_id in tree.traverse_preorder():
    print(f"Visiting node {node_id}")
```

```python
traverse_postorder(from_node: Union[int, str] = None) -> Generator
```
Post-order traversal (left, right, root). Useful for dependency resolution.

```python
traverse_levelorder(from_node: Union[int, str] = None) -> Generator
```
Breadth-first level order traversal. Yields nodes by depth level.

```python
traverse_leaves_only(from_node: Union[int, str] = None) -> Generator
```
Efficient traversal that only yields leaf nodes. Skips internal nodes.

```python
traverse_internal_only(from_node: Union[int, str] = None) -> Generator
```
Traversal that skips leaf nodes. Useful for operations only on internal nodes.

```python
traverse_with_depth(from_node: Union[int, str] = None) -> Generator[Tuple[int, int], None, None]
```
Traversal yielding (node ID, depth) pairs. Depth starts at 0 for root.

```python
traverse_with_distances(from_node: Union[int, str] = None) -> Generator[Tuple[int, float, float], None, None]
```
Traversal yielding (node ID, distance to parent, cumulative distance to root).

### Topological Analysis
```python
bipartition(node: Union[int, str], by_id=False) -> frozenset
bipartitions(by_id=False) -> Generator[frozenset, None, None]
quartet_topology(a: Union[int, str], b: Union[int, str], c: Union[int, str], d: Union[int, str]) -> frozenset
quartet_topologies_bulk(quartets: Union[list, np.ndarray]) -> np.ndarray
quartet_topologies_by_name(quartets: List[Tuple[str, str, str, str]]) -> List[frozenset]
```

### Graph Operations
```python
adjacency_matrix(from_node: Union[int, str] = None) -> Dict[str, Any]
laplacian_matrix(from_node: Union[int, str] = None) -> Dict[str, Any]
incidence_matrix(from_node: Union[int, str] = None) -> Dict[str, Any]
degree_sequence(from_node: Union[int, str] = None) -> Dict[str, Any]
```

### Export & Conversion
```python
to_networkx_graph(from_node: Union[int, str] = None) -> 'networkx.Graph'
to_newick(include_support=True, include_distances=True) -> str
relative_evolutionary_divergence() -> Dict[int, float]
```

## Example Usage

```python
# Initialize and basic properties
tree = SuchTree("(A:0.1,B:0.2,(C:0.3,D:0.4)E:0.5)F;")
print(f"Tree depth: {tree.depth}")
print(f"Leaf names: {tree.leaf_names}")

# Node relationships
node_id = tree.leaves['A']
parent_id = tree.get_parent(node_id)
children = tree.get_children(parent_id)

# Distance analysis
dist = tree.distance('A', 'C')
dist_matrix = tree.pairwise_distances(['A', 'B', 'C', 'D'])

# Advanced features
red_values = tree.relative_evolutionary_divergence()
nx_graph = tree.to_networkx_graph()
```

## Error Handling
Raises specific exceptions:
- `NodeNotFoundError`: Invalid leaf name
- `InvalidNodeError`: Invalid node ID
- `TreeStructureError`: Invalid tree operations
- `ValueError`: Invalid input format
- `TypeError`: Incorrect argument type

## Best Practices
1. Use node IDs for performance-critical code
2. Prefer bulk methods (`distances_bulk`) for multiple calculations
3. Cache frequently-used properties (like RED values)
4. Use traversal generators for memory efficiency
