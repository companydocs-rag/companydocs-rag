"""Retrieval (spec 006): hybrid search over the index.

``search.py`` holds :class:`.search.Retriever`, which spec 007 calls
in-process: it queries the dense and the sparse leg separately and fuses them
with reciprocal rank fusion (``fusion.py``, a pure function), and the
``retrieval`` settings. ``cli.py`` is the ``companydocs-search`` command, a
thin wrapper around it. This module imports nothing.
"""
