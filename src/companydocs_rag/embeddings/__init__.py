"""Embeddings (spec 006): the dense and the sparse side of the index.

``dense.py`` holds the embedding text (:func:`.dense.compose_embedding_text`),
the ``Embedder`` protocol, the sentence-transformers embedder, which imports
sentence-transformers only when it is constructed, and the ``embeddings``
settings. ``sparse.py`` turns a text into terms, term ids and BM25 values.
This module imports nothing.
"""
