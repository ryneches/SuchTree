---
title: Release notes
subtitle: What's new in this release? What have we been up to?
icon: material/bullhorn-outline
---

#### New for SuchTree v1.3
- Automatic URL/file/NEWICK input no longer broken
- Custom exception classes
- Improved naming conventions of methods & properties
- Refactored node query methods
- Regularized naming and arguments for distance methods
- Refactored topology analysis methods
- More traversal methods
- Refactored graph theory and matrix operations
- Refactored export methods
- Stubs for backward compatibility with deprecation warnings
- Improved test coverage
- Various bug fixes
- Verified commits for GitHub, digital attestations for PiPy
- Renamed `master` to `main`
- New logo, website and documentation 

#### New for SuchTree v1.2

- Quartet topology tests provided by `SuchTree.get_quartet_topology( a, b, c, d )`
- Optimized, thread-safe bulk quartet topology tests provided by
  `SuchTree.quartet_topologies( [N,4] )`
- SuchTree now automatically detects and uses NEWICK strings as for initialization

#### New for SuchTree v1.1

- Basic support for support values provided by `SuchTree.get_support( node_id )`
- Relative evolutionary divergence (RED)
- Bipartitions
- Node generators for in-order and preorder traversal
- Summary of leaf relationships via `SuchTree.relationships()`
