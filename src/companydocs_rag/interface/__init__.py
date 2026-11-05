"""Demo interface (spec 009): ``companydocs-demo``, a local browser interface.

A single-user, single-threaded HTTP server on 127.0.0.1 over spec 007's
:class:`companydocs_rag.answering.answerer.Answerer`. A reader asks a question
in the browser, reads the answer with one card per cited source, and opens
each cited page of the PDF as an image. ``cli.py`` holds the command,
``settings.py`` the ``interface`` settings, ``texts.py`` the Dutch texts,
``fragments.py`` the HTML of an answer, ``pageview.py`` the hash to file map
and the page view, ``htmlpage.py`` the page (``page.html``) and
``server.py`` the server. This module imports nothing.
"""
