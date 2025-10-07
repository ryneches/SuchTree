---
title: SuchTree Documentation
subtitle: A Python library for doing fast, thread-safe computations with phylogenetic trees.
icon: material/code-block-braces
---

[![banner](docs/assets/banner.png)](https://suchtree.vort.org)

[![Actions Status](https://github.com/ryneches/SuchTree/workflows/Build%20wheels/badge.svg)](https://github.com/ryneches/SuchTree/actions) [![codecov](https://codecov.io/gh/ryneches/SuchTree/branch/master/graph/badge.svg)](https://codecov.io/gh/ryneches/SuchTree) [![License](https://img.shields.io/badge/license-BSD--3-blue.svg)](https://raw.githubusercontent.com/ryneches/SuchTree/master/LICENSE) [![JOSS](http://joss.theoj.org/papers/23bac1ae69cfaf201203dd52d7dd5610/status.svg)](http://joss.theoj.org/papers/23bac1ae69cfaf201203dd52d7dd5610) [![GitHub all releases](https://img.shields.io/github/downloads/ryneches/SuchTree/total?label=downloads&logo=github)](https://github.com/ryneches/SuchTree/graphs/traffic) [![PyPI - Downloads](https://img.shields.io/pypi/dd/SuchTree?logo=PyPI)](https://pypistats.org/packages/suchtree) [![Conda Downloads](https://img.shields.io/conda/d/bioconda/suchtree)](https://anaconda.org/bioconda/suchtree) ![GitHub commits since tagged version](https://img.shields.io/github/commits-since/ryneches/SuchTree/latest)
 [![Mastodon Follow](https://img.shields.io/mastodon/follow/109294614904147843?domain=ecoevo.social&style=flat&logo=mastodon)](https://ecoevo.social/@ryneches)

## Project Website

Documentation, example notebooks, literature survey data, benchmarks, the API
reference and our publications can be found at the SuchTree project website,
[suchtree.vort.org](https://suchtree.vort.org).

## High-performance sampling of very large trees

So, you have a phylogenetic tree, and you want to do some statistics with it.
There are lots of packages in Python that let you manipulate phylogenies, like
[`dendropy`](http://www.dendropy.org/), the tree model included in
[`scikit-bio`](http://scikit-bio.org/docs/latest/tree.html),
[`ete3`](http://etetoolkit.org/) and the awesome, shiny new
[`toytree`](https://github.com/eaton-lab/toytree). For trees of modest size and
statistical methods that don't require *too* many traversals, there a lot of
great options. If you're working with about a thousand taxa or less, you should
be able to use any of those packages for your tree.

However, if you are working with trees that include tens of thousands, or
perhaps millions of taxa, you will run into problems. `ete3`, `dendropy`,
`toytree`, and`scikit-bio`'s `TreeNode` are all designed to give you
lots of flexibility. You can re-root trees, use different traversal
schemes, attach metadata to nodes, attach and detach nodes, splice
sub-trees into or out of the main tree, plot trees for publication
figures and do lots of other useful things. That power and flexibility
comes with a price : speed.

For trees of moderate size, it is sometimes possible to solve the speed issue
by working with a matrix representation of the tree. Unfortunately, these
representations scale quadratically with the number of taxa in the tree.  For
example, the distance matrix for a tree of 100,000 taxa contains 10,000,000,000
elements, which will consume about 20GB of RAM. If your method performs
sampling on this matrix then almost every operation will be a cache miss.
Unless you are very clever about access patterns and matrix layout, the
performance will be limited by RAM latency, leaving the CPU mostly idle. SuchTree
is designed to solve this problem by representing trees as a highly compact
object that usually fits into the CPU's L3 cache even for very large trees,
and employs simple, assembly-language code paths for accessing data. Please
see the [Benchmarks](docs/benchmarks.md) for a more detailed look at performance.

## Sampling linked trees

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

## Installation

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
([`wheels`](https://realpython.com/python-wheels/)) are available for CPython
3.9, 3.10 and 3.11, 3.12, 3.13 on Linux x86_64 and on MacOS with Intel and
Apple silicon. If your platform isn't in that list, but it is supported by
[`cibuildwheel`](https://github.com/pypa/cibuildwheel), please file an issue to
request your platform! I would be absolutely _delighted_ to help you get
`SuchTree` deployed on an exotic embedded system or a mainframe.

To install the most recent development version :

```
git clone https://github.com/ryneches/SuchTree.git
cd SuchTree
pip install -r requirements.txt
pip install .
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


## Basic usage

`SuchTree` will accept URLs, file paths or valid NEWICK strings :

```python
from SuchTree import SuchTree

T = SuchTree( 'test.tree' )
T = SuchTree( 'https://github.com/ryneches/SuchTree/blob/master/data/gopher-louse/gopher.tree' )
T = SuchTree( '(A,B,(C,D));' )
```

If you are just starting out, begin with the [Working
Example](docs/examples/SuchTree_examples.md). If you are interested in working with
linked trees, you should start with the [Linked
Trees](docs/examples/SuchLinkedTree_examples.md).

For more, check out the [API Documentation](docs/api_docs.md) for how to use
SuchTree, or the [API Reference](docs/api.md). The API Reference is generated
automatically after each commit; it's guaranteed to be up-to-date, but not
necessarily fun to read. 

I highly recommend using SuchTree with [`toytree`](https://eaton-lab.org/toytree/)
for visualizing trees. Look for more convenient interoperation with `toytree` in
future releases of SuchTree!

## Citing SuchTree

Please cite our 2018 paper in the Journal of Open Source Software :

> Russell Y. Neches, and Camille Scott. "Suchtree: Fast, thread-safe computations with phylogenetic trees." *Journal of Open Source Software* 3, no. 26 (2018): 678.

DOI : [https://doi.org/10.21105/joss.00678](https://doi.org/10.21105/joss.00678)

## Thanks

Special thanks to [@camillescott](https://github.com/camillescott) and 
[@pmarkowsky](https://github.com/pmarkowsky) for their many helpful
suggestions (and for their patience).

