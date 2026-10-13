"""Offline structural checks and shared helpers for the lab notebook."""

import json
import subprocess
import sys
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


def test_generator_in_sync():
    result = subprocess.run(
        [sys.executable, "scripts/make_notebook.py", "--check"], cwd=ROOT,
        capture_output=True, text=True, encoding="utf-8",
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_first_code_cell_clones_and_chdirs():
    source = cells("code")[0]["source"]
    assert "https://github.com/long2110d/K4-Track-4-Day10-VisionLanguageModel.git" in source
    assert "'/content/lab10'" in source
    assert "['git', 'clone', REPO, BASE]" in source
    assert "os.chdir(BASE)" in source
    assert not any(line.lstrip().startswith(("!", "%")) for line in source.splitlines())
    notebook_code = "\n\n".join(cell["source"] for cell in cells("code"))
    for command in ("pip", "prepare_data"):
        assert notebook_code.index("os.chdir(") < notebook_code.index(command)


def test_no_absolute_paths_after_bootstrap():
    for cell in cells("code")[1:]:
        assert "/content" not in cell["source"]


def test_data_cell():
    source = section_code(1)
    for token in ("kagglehub", "prepare_data.py --kaggle", "prepare_data.py --source",
                  "data/splits/meta.json", "data/splits/train.jsonl"):
        assert token in source


def test_no_secrets_printed():
    for cell in cells("code"):
        for line in cell["source"].splitlines():
            assert not ("print(" in line and "KAGGLE_KEY" in line)


def test_generator_check_detects_drift_without_writing(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    generator = scripts / "make_notebook.py"
    generator.write_text((ROOT / "scripts/make_notebook.py").read_text(encoding="utf-8"),
                         encoding="utf-8")
    notebook = tmp_path / "lab_day10.ipynb"
    notebook.write_text("{}", encoding="utf-8")
    result = subprocess.run([sys.executable, str(generator), "--check"],
                            capture_output=True, encoding="utf-8")
    assert result.returncode == 1
    assert "git diff" in result.stdout
    assert notebook.read_text(encoding="utf-8") == "{}"
