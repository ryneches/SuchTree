# SuchTree

A Python library for doing fast, thread-safe computations with
phylogenetic trees.

[![Build
Status](https://travis-ci.org/ryneches/SuchTree.svg?branch=master)](https://travis-ci.org/ryneches/SuchTree)

### So problem

You have a phylogenetic tree, and you want to do some statistics with
it. No problem! There are lots of packages in Python that let you
manipulate phylogenies, like [`dendropy`](http://www.dendropy.org/),
[`scikit-bio`](http://scikit-bio.org/docs/latest/tree.html) and
[`ETE`](http://etetoolkit.org/). Surely one of them will work. And
indeed they will, if your tree isn't *too* big and your statistical
method doesn't require *too* many traversals. If you're working with a
hundred or a thousand organisms, no problem. You should probably
forget about `SuchTree` and use a tree package that has lots of cool
features.

If, however, you are working with trees that include tens of
thousands, or maybe even millions of organisms, you are going to run
into problems. `ETE`, `dendropy` and `scikit-bio`'s `TreeNode` are all
implemented to give you lots of flexibility. You can re-root trees,
use different traversal schemes, attach metadata to nodes, attach and
detach nodes, splice sub-trees into or out of the main tree, and do
lots of other useful things. However, that power and flexibility comes
with a price; speed.

For trees of moderate size, it is possible to solve the speed issue by
doing your statistics on a matrix of patristic distances.
Unfortunately, distance matrixes scale quadratically with the number
of taxa in your tree. A distance matrix for a tree of 100,000 taxa
will consume about 20GB of RAM. If your statistical method performs
sampling, then almost every operation will be a cache miss. Even if
you have the RAM, it will be painfully slow.

### Much solution

`SuchTree` is motivated by the observation that, while a distance
matrix of 100,000 taxa is quite bulky, the tree it represents can be
made to fit into about 7.6MB of RAM if implemented simply using only
`C` primitives.  This is small enough to fit into L2 cache on many
modern microprocessors. This comes at the cost of traversing the tree
for every calculation (about 16 hops from leaf to root for a 100,000
taxa tree), but, as these operations all happen on-chip, the processor
can take full advantage of
[pipelining](https://en.wikipedia.org/wiki/Instruction_pipelining),
[speculative execution](https://en.wikipedia.org/wiki/Speculative_execution)
and other optimizations available in modern CPUs.

### Nice benchmark


### Wow

Special thanks to [@camillescott](https://github.com/camillescott) and 
[@pmarkowsky](https://github.com/pmarkowsky) for their many helpful
suggestions (and for their patience).
