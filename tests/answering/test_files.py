"""The committed files and docs of spec 007 (D13, D14, R21, Q10, A14).

``git diff main -- README.md`` and ``git diff main -- docs/DEVELOPMENT.md`` are checked
outside the suite; these tests pin the content.
"""

from __future__ import annotations

import tomllib

import pytest

from companydocs_rag.ingestion.settings import find_project_root

ROOT = find_project_root()


@pytest.mark.parametrize(
    ("name", "base"),
    [
        ("companydocs-gemma4-e4b-8k.Modelfile", "gemma4:e4b"),
        ("companydocs-qwen3.5-4b-8k.Modelfile", "qwen3.5:4b"),
    ],
)
def test_each_modelfile_is_its_two_lines(name, base):
    raw = (ROOT / "config" / "ollama" / name).read_bytes()

    assert raw == f"FROM {base}\nPARAMETER num_ctx 8192\n".encode("ascii")


def test_the_default_model_has_a_modelfile():
    from companydocs_rag.answering.settings import MODEL

    assert (ROOT / "config" / "ollama" / f"{MODEL}.Modelfile").is_file()


def test_the_readme_has_the_section_on_the_ollama_models():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    section = text[text.index("## Ollama models") : text.index("## Tests")]

    for name in ("companydocs-gemma4-e4b-8k", "companydocs-qwen3.5-4b-8k"):
        assert f"ollama create {name} -f config/ollama/{name}.Modelfile" in section
    assert "A team member always pulls and creates the models by hand" in section
    assert "Falling back to Qwen means setting" in section


def test_the_development_guide_names_the_answering_stage():
    # The stage and its command are named; the wording around them is free.
    text = (ROOT / "docs" / "DEVELOPMENT.md").read_text(encoding="utf-8")

    assert "`answering`" in text
    assert "companydocs-answer" in text


def test_pyproject_declares_httpx_and_the_command():
    with (ROOT / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)["project"]

    assert "httpx>=0.28.1" in project["dependencies"]
    assert project["scripts"]["companydocs-answer"] == (
        "companydocs_rag.answering.cli:main"
    )
