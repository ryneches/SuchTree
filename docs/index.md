---
title: SuchTree Documentation
subtitle: A Python library for doing fast, thread-safe computations with phylogenetic trees.
icon: material/code-block-braces
---

# SuchTree

A Python library for doing fast, thread-safe computations with
phylogenetic trees.

[![Actions Status](https://github.com/ryneches/SuchTree/workflows/Build%20wheels/badge.svg)](https://github.com/ryneches/SuchTree/actions) [![codecov](https://codecov.io/gh/ryneches/SuchTree/branch/master/graph/badge.svg)](https://codecov.io/gh/ryneches/SuchTree) [![License](https://img.shields.io/badge/license-BSD--3-blue.svg)](https://raw.githubusercontent.com/ryneches/SuchTree/master/LICENSE) [![JOSS](http://joss.theoj.org/papers/23bac1ae69cfaf201203dd52d7dd5610/status.svg)](http://joss.theoj.org/papers/23bac1ae69cfaf201203dd52d7dd5610) [![GitHub all releases](https://img.shields.io/github/downloads/ryneches/SuchTree/total?label=downloads&logo=github)](https://github.com/ryneches/SuchTree/graphs/traffic) [![PyPI - Downloads](https://img.shields.io/pypi/dd/SuchTree?logo=PyPI)](https://pypistats.org/packages/suchtree) [![Conda Downloads](https://img.shields.io/conda/d/bioconda/suchtree)](https://anaconda.org/bioconda/suchtree)


### High-performance sampling of very large trees

So, you have a phylogenetic tree, and you want to do some statistics with it.
There are lots of packages in Python that let you manipulate
phylogenies, like [`dendropy`](http://www.dendropy.org/), the tree model
included in [`scikit-bio`](http://scikit-bio.org/docs/latest/tree.html),
[`ete3`](http://etetoolkit.org/) and the awesome, shiny new 
[`toytree`](https://github.com/eaton-lab/toytree). If your tree isn't *too*
big and your statistical tests doesn't require *too* many traversals, there 
a lot of great options. If you're working with about a thousand taxa or less,
you should be able to use any of those packages for your tree.

However, if you are working with trees that include tens of thousands, or
maybe even millions of taxa, you are going to run into problems. `ete3`,
`dendropy`, `toytree`, and`scikit-bio`'s `TreeNode` are all designed to give
you lots of flexibility. You can re-root trees, use different traversal
schemes, attach metadata to nodes, attach and detach nodes, splice sub-trees
into or out of the main tree, plot trees for publication figures and do lots
of other useful things. That power and flexibility comes with a price -- speed.

For trees of moderate size, it is possible to solve the speed issue by
working with matrix representations of the tree. Unfortunately, these
representations scale quadratically with the number of taxa in the tree.
A distance matrix for a tree of 100,000 taxa will consume about 20GB 
of RAM. If your method performs sampling, then almost every operation
will be a cache miss. Unless you are very clever about access patterns and
matrix layout, the performance will be limited by RAM latency, leaving the
CPU mostly idle.

### Sampling linked trees

Suppose you have more than one group of organisms, and you want to study
the way their interactions have influenced their evolution. Now, you have
several trees that link together to form a generalized graph.

`SuchLinkedTrees` has you covered. At the moment, `SuchLinkedTrees` supports
trees of two interacting groups. Like `SuchTree`, `SuchLinkedTrees` is not
intended to be a general-purpose graph theory package. Instead, it leverages
`SuchTree` to efficiently handle the problem-specific tasks of working with
co-phylogeny systems. It will load your datasets. It will build the graphs. It
will let you subset the graphs using their phylogenetic or ecological
properties. It will generate weighted adjacency and Laplacian matrixes of the
whole graph or of subgraphs you have selected. It will generate spectral
decompositions of subgraphs if spectral graph theory is your thing.

And, if that doesn't solve your problem, it will emit sugraphs as `Graph`
objects for use with the [`igraph`](http://igraph.org/) network analysis
package, or node and edge data for building graphs in 
[`networkx`](https://networkx.github.io/). Now you can do even more things. 
Maybe you want to get all crazy with some 
[graph kernels](https://github.com/BorgwardtLab/GraphKernels)?
Well, now you can.

### Installation

`SuchTree` depends on the following packages :

* `scipy`
* `numpy`
* `dendropy`
* `cython`
* `pandas`

To install the current release, you can install from PyPI :

```
pip install SuchTree
```

If you install using `pip`, binary packages
([`wheels`](https://realpython.com/python-wheels/)) are available for CPython 3.6, 3.7,
3.8, 3.9, 3.10 and 3.11 on Linux x86_64 and on MacOS with Intel and Apple
silicon. If your platform isn't in that list, but it is supported by
[`cibuildwheel`](https://github.com/pypa/cibuildwheel), please file an issue
to request your platform! I would be absolutely _delighted_ if someone was
actually running `SuchTree` on an exotic embedded system or a mainframe.

To install the most recent development version :

```
git clone https://github.com/ryneches/SuchTree.git
cd SuchTree
./setup.py install
```

To install via conda, first make sure you've got the
[bioconda](https://bioconda.github.io/) channel set up, if you haven't already :

```
conda config --add channels bioconda
conda config --add channels conda-forge
conda config --set channel_priority strict
```

Then, install in the usual way :

```
conda install suchtree
```

**Note that the conda package name is lower case!**


### Basic usage

`SuchTree` will accept either a URL or a file path :

```python
from SuchTree import SuchTree

T = SuchTree( 'test.tree' )
T = SuchTree( 'https://github.com/ryneches/SuchTree/blob/master/data/gopher-louse/gopher.tree' )
```

The available properties are :

* `length` : the number of nodes in the tree
* `depth` : the maximum depth of the tree
* `root` : the id of the root node
* `leafs` : a dictionary mapping leaf names to their ids
* `leafnodes` : a dictionary mapping leaf node ids to leaf names
* `RED` : a dictionary of RED (relative evolutionary divergence) scores for internal nodes, calculated on first access

The available methods of `SuchTree` are :

* `get_parent` : for a given node id or leaf name, return the parent id
* `get_support` : return the support value, if available
* `get_children` : for a given node id or leaf name, return the ids of
the child nodes (leaf nodes have no children, so their child node ids will
always be -1)
* `get_leafs` : return an array of ids of all leaf nodes that descend from a node
* `get_descendant_nodes` : generator for ids of all nodes that descend from a node, including leafs
* `get_bipartition` : return the two sets of leaf nodes partitioned by a node
* `bipartitions` : generator of all bipartitions
* `get_internal_nodes` : return array of internal nodes
* `get_nodes` : return an array of all nodes
* `in_order` : generator for an in-order traversal of the tree
* `pre_order` : generator for a pre-order traversal of the tree
* `get_distance_to_root` : for a given node id or leaf name, return
the integrated phylogenetic distance to the root node
* `mrca` : for a given pair of node ids or leaf names, return the id
of the nearest node that is parent to both
* `is_leaf` : returns True if the node is a leaf
* `is_internal_node` : returns True if the node is an internal node
* `is_ancestor` : returns 1 if *a* is an ancestor of *b*, -1 if *b* is an ancestor of *a*, or 0 otherwise
* `distance` : for a given pair of node ids or leaf names, return the
patristic distance between the pair
* `distances` : for an (n,2) array of pairs of node ids, return an (n)
array of patristic distances between the pairs
* `distances_by_name` for an (n,2) list of pairs of leaf names, return
an (n) list of patristic distances between each pair
* `get_quartet_topology` : for a given quartet, return the topology of that quartet
* `quartet_topologies` : compute the topologies of an array of quartets by id
* `quartet_topologies_by_name` : compute the topologies of quartets by their taxa names
* `dump_array` : print out the entire tree (for debugging only! May
produce pathologically gigantic output.)
* `adjacency` : build the graph adjacency matrix of the tree
* `laplacian` : build the Laplacian matrix of the tree
* `nodes_data` : generator for node data, compatible with `networkx`
* `edges_data` : generator for edge data, compatible with `networkx`
* `relationships` : builds a Pandas DataFrame describing relationships among taxa


### Thanks

Special thanks to [@camillescott](https://github.com/camillescott) and 
[@pmarkowsky](https://github.com/pmarkowsky) for their many helpful
suggestions (and for their patience).

