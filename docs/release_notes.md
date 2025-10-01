---
title: Release notes
subtitle: What's new in this release? What have we been up to?
icon: material/bullhorn-outline
---

#### New for SuchTree v1.2

* Quartet topology tests provided by `SuchTree.get_quartet_topology( a, b, c, d )`
* Optimized, thread-safe bulk quartet topology tests provided by
  `SuchTree.quartet_topologies( [N,4] )`
* SuchTree now automatically detects and uses NEWICK strings as for initialization

#### New for SuchTree v1.1

* Basic support for support values provided by `SuchTree.get_support( node_id )`
* Relative evolutionary divergence (RED)
* Bipartitions
* Node generators for in-order and preorder traversal
* Summary of leaf relationships via `SuchTree.relationships()`
