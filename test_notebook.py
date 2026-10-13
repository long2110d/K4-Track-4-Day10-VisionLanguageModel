"""Offline structural checks and shared helpers for the lab notebook."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NB = ROOT / "lab_day10.ipynb"


def load_nb():
    return json.loads(NB.read_text(encoding="utf-8"))


def cells(kind):
    return [
        {**cell, "source": "".join(cell["source"])}
        for cell in load_nb()["cells"]
        if cell["cell_type"] == kind
    ]


def section_code(n):
    sources = []
    in_section = False
    for cell in load_nb()["cells"]:
        source = "".join(cell["source"])
        first_line = source.splitlines()[0] if source else ""
        if cell["cell_type"] == "markdown":
            if first_line.startswith(f"## {n}."):
                in_section = True
            elif in_section and first_line.startswith("## "):
                break
        elif in_section and cell["cell_type"] == "code":
            sources.append(source)
    return "\n\n".join(sources)


def strip_magics(src):
    lines = []
    for line in src.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith(("!", "%")):
            indentation = line[:len(line) - len(stripped)]
            ending = line[len(line.rstrip("\r\n")):]
            line = indentation + "pass" + ending
        lines.append(line)
    return "".join(lines)


def all_code():
    return "\n\n".join(strip_magics(cell["source"]) for cell in cells("code"))


def test_notebook_is_valid_json():
    nb = load_nb()
    assert nb["nbformat"] == 4
    for cell in nb["cells"]:
        assert "cell_type" in cell
        assert "source" in cell
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            assert cell["execution_count"] is None


def test_headings_in_skeleton_order():
    prefixes = ["# Lab Day 10"] + [f"## {n}." for n in range(9)]
    headings = []
    for cell in cells("markdown"):
        first_line = cell["source"].splitlines()[0] if cell["source"] else ""
        headings.extend(prefix for prefix in prefixes if first_line.startswith(prefix))
    assert headings == prefixes


def test_every_code_cell_compiles():
    for i, cell in enumerate(cells("code")):
        compile(strip_magics(cell["source"]), f"cell{i}", "exec")
