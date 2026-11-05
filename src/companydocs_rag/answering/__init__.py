"""Answering (spec 007): answers from a local LLM over the retrieved chunks.

``answerer.py`` holds :class:`.answerer.Answerer`, the in-process API: it
selects chunks from :class:`companydocs_rag.retrieval.search.Retriever`'s
result for a prompt, asks a generator behind an OpenAI-compatible
chat-completions endpoint (Ollama on localhost by default), maps the cited
source numbers back to chunks with their page provenance, abstains when the
evidence is insufficient, and logs every answer. ``settings.py`` holds the
``answering`` settings and ``cli.py`` the ``companydocs-answer`` command. This
module imports nothing.
"""
