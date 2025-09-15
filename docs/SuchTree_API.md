# SuchTree Class API Reference

## Overview
The `SuchTree` class represents phylogenetic trees and provides efficient tree manipulation and analysis capabilities. It is implemented in Cython for high performance.

## Initialization
```python
class SuchTree(tree_input: Union[str, Path])
```
Construct a tree from:
- Newick string
- File path
- URL pointing to a Newick file

Example:
```python
tree = SuchTree("(A,B,(C,D));")
tree = SuchTree("path/to/tree.newick")
```

## Properties

### Tree Structure
| Property | Type | Description |
|----------|------|-------------|
| `size` | `int` | Total number of nodes |
| `depth` | `int` | Maximum depth of the tree |
| `num_leaves` | `int` | Number of leaf nodes |
| `root_node` | `int` | ID of the root node |
| `polytomy_epsilon` | `float` | Minimum branch length for polytomies |

### Node Collections
| Property | Type | Description |
|----------|------|-------------|
| `leaves` | `Dict[str, int]` | Leaf name → node ID mapping |
| `leaf_nodes` | `Dict[int, str]` | Node ID → leaf name mapping |
| `internal_nodes` | `np.ndarray` | Array of internal node IDs |
| `all_nodes` | `np.ndarray` | Array of all node IDs |

## Key Methods

### Node Relationships
```python
get_parent(node: Union[int, str]) -> int
get_children(node: Union[int, str]) -> Tuple[int, int]
get_ancestors(node: Union[int, str]) -> Generator[int, None, None]
get_descendants(node: int) -> Generator[int, None, None]
get_leaves(node: Union[int, str]) -> np.ndarray
```

### Tree Navigation
```python
is_leaf(node: Union[int, str]) -> bool
is_internal(node: Union[int, str]) -> bool
is_ancestor(ancestor: Union[int, str], descendant: Union[int, str]) -> int
is_descendant(descendant: Union[int, str], ancestor: Union[int, str]) -> bool
common_ancestor(a: Union[int, str], b: Union[int, str]) -> int
```

### Distance Calculations
```python
distance(a: Union[int, str], b: Union[int, str]) -> float
distance_to_root(node: Union[int, str]) -> float
distances_bulk(pairs: np.ndarray) -> np.ndarray
pairwise_distances(nodes: list = None) -> Dict[str, Any]
```

### Traversal Methods
```python
traverse_inorder(include_distances: bool = True) -> Generator
traverse_preorder(from_node: Union[int, str] = None) -> Generator  
traverse_postorder(from_node: Union[int, str] = None) -> Generator
traverse_levelorder(from_node: Union[int, str] = None) -> Generator
```

### Advanced Features
```python
relative_evolutionary_divergence() -> Dict[int, float]
adjacency_matrix(from_node: Union[int, str] = None) -> Dict[str, Any]
laplacian_matrix(from_node: Union[int, str] = None) -> Dict[str, Any]
to_networkx_graph(from_node: Union[int, str] = None) -> 'networkx.Graph'
to_newick(include_support: bool = True, include_distances: bool = True) -> str
```

## Example Usage
```python
# Basic tree analysis
tree = SuchTree("(A:0.1,B:0.2,(C:0.3,D:0.4)E:0.5)F;")
print(f"Tree has {tree.num_leaves} leaves and {tree.size} total nodes")

# Get distances
dist = tree.distance("A", "C")
path = tree.path_between_nodes("B", "D")

# Process subtree
subtree_root = tree.common_ancestor("C", "D")
subtree_leaves = tree.get_leaves(subtree_root)

# Export formats
newick_str = tree.to_newick()
nx_graph = tree.to_networkx_graph()
```

## Error Handling
The class raises specific exceptions:
- `NodeNotFoundError`: Invalid leaf name
- `InvalidNodeError`: Invalid node ID
- `TreeStructureError`: Invalid tree operations
