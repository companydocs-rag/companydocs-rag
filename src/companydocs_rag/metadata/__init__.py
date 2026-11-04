"""Metadata (spec 008): the date and term fields of the lease family.

``cli.py`` holds the ``companydocs-metadata`` command. Its ``extract``
subcommand assigns a document type to every chunked document (an override
file, then the client index), asks a local LLM behind an OpenAI-compatible
chat-completions endpoint (Ollama on localhost by default) for the date and
term fields of each document of the lease family, checks every quote against
its chunk, and writes one extraction file per chunk file and one JSONL log per
run. Its ``report`` subcommand computes, for a reference date the user gives,
the expiry date, the next renewal date and the latest notice date of each
document, and writes a follow-up overview in Dutch. ``settings.py`` holds the
``metadata`` settings. Only the extraction modules import the generator client
of :mod:`companydocs_rag.answering`; nothing that ``report`` uses does. This
module imports nothing.
"""
