"""The vector store (spec 006): the ``companydocs-index`` command and the index.

``store.py`` is the only module that opens Qdrant: local mode, one collection
with a named dense and a named sparse vector per chunk. ``manifest.py``
writes and reads ``manifest.json`` and compares it with the current settings
(D17). ``cli.py`` is the command, which rebuilds the whole index on every
run. Qdrant local mode locks its folder, so one process at a time uses an
index. This module imports nothing.
"""
