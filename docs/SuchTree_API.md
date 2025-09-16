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
get_children(node: Union[int, str]) -> Tuple[int, int]
get_ancestors(node: Union[int, str]) -> Generator[int, None, None]
get_descendants(node_id: int) -> Generator[int, None, None]
get_leaves(node: Union[int, str]) -> np.ndarray
get_support(node: Union[int, str]) -> float
```

### Tree Navigation
```python
is_leaf(node: Union[int, str]) -> bool
is_internal(node: Union[int, str]) -> bool
is_ancestor(ancestor: Union[int, str], descendant: Union[int, str]) -> int
is_descendant(descendant: Union[int, str], ancestor: Union[int, str]) -> bool
is_root(node: Union[int, str]) -> bool
is_sibling(node1: Union[int, str], node2: Union[int, str]) -> bool
has_children(node: Union[int, str]) -> bool
has_parent(node: Union[int, str]) -> bool
common_ancestor(a: Union[int, str], b: Union[int, str]) -> int
path_between_nodes(a: Union[int, str], b: Union[int, str]) -> List[int]
```

### Distance Analysis
```python
distance(a: Union[int, str], b: Union[int, str]) -> float
distance_to_root(node: Union[int, str]) -> float
distances_bulk(pairs: np.ndarray) -> np.ndarray
distances_by_name(pairs: List[Tuple[str, str]]) -> List[float]
pairwise_distances(nodes: list = None) -> np.ndarray
nearest_neighbors(node: Union[int, str], k=1) -> List[Tuple[Union[int, str], float]]
```

### Tree Traversal
```python
traverse_inorder(include_distances: bool = True) -> Generator
traverse_preorder(from_node: Union[int, str] = None) -> Generator
traverse_postorder(from_node: Union[int, str] = None) -> Generator
traverse_levelorder(from_node: Union[int, str] = None) -> Generator
traverse_leaves_only(from_node: Union[int, str] = None) -> Generator
traverse_internal_only(from_node: Union[int, str] = None) -> Generator
traverse_with_depth(from_node: Union[int, str] = None) -> Generator[Tuple[int, int], None, None]
traverse_with_distances(from_node: Union[int, str] = None) -> Generator[Tuple[int, float, float], None, None]
```

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
