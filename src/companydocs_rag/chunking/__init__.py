"""Chunking (spec 005): the ``companydocs-chunk`` command and the pure
functions behind it.

``chunker.py`` cuts each normalized document into chunks: the sections of its
own article and clause numbering where an accepted numbering run is found,
and line-based pieces elsewhere, each chunk with its page provenance. The
text is never changed. ``cli.py`` is the command. This module imports
nothing.
"""
