---
title: API Reference
subtitle: A complete, always up-to-date reference of SuchTree's API.
icon: material/qrcode-scan
---

# API Reference

This reference is automatically generated from the [source code](https://github.com/ryneches/SuchTree)
and docstrings using [`mkdockstrings`](https://mkdocstrings.github.io/). This page will always 
be up to date to the latest git commit, but you may prefer [documentation written by an actual
human](api_docs.md) to be somewhat more palatable. 

Because `SuchTree` and `SuchLinkedTrees` are written in Cython and compiled into a shared library,
the `mkdocstrings` is not (yet) able to extract the function implementations for documentation purposes.
Sorry about that. You can always look in the source code.

## SuchTree.SuchTree

::: SuchTree.MuchTree.SuchTree
    options:
      heading_level: 3
      show_root_heading: false
      show_source: false
      show_signature_annotations: true
      group_by_category: true
      show_category_heading: true
      members_order: source
      filters:
        - "!^_" # careful : SuchTree prefixes cdef functions with _ by convention only
        - "!^__pyx_vtable__$"
        - "!^__doc__$"
        - "!^__new__$"
        - "!^__init_subclass__$"
        - "!^__subclasshook__$"

## SuchTree.SuchLinkedTrees

::: SuchTree.MuchTree.SuchLinkedTrees
    options:
      heading_level: 3
      show_root_heading: false
      show_source: false
      show_signature_annotations: true
      group_by_category: true
      show_category_heading: true
      members_order: source
      filters:
        - "!^_" # careful : SuchTree prefixes cdef functions with _ by convention only
        - "!^__pyx_vtable__$"
        - "!^__doc__$"
        - "!^__new__$"
        - "!^__init_subclass__$"
        - "!^__subclasshook__$"

::: SuchTree.exceptions
    options:
      heading_level: 2
      show_root_heading: true
      show_source: true
