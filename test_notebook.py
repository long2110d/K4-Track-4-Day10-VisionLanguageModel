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


def test_label_functions_match_lab10():
    import ast
    import lab10

    tree = ast.parse(strip_magics(section_code(2)))
    names = {'yolo_to_box', 'objects_to_answer', 'norm1000_to_pixel'}
    nodes = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
             or isinstance(node, ast.FunctionDef) and node.name in names
             or isinstance(node, ast.Assign) and any(
                 isinstance(target, ast.Name) and target.id == 'PROMPT' for target in node.targets)]
    ns = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<labels>', 'exec'), ns)
    assert ns['PROMPT'] == lab10.PROMPT
    fixtures = [
        [{'label': 'crack', 'yolo': [0.5, 0.5, 0.2, 0.1]}],
        [{'label': 'pothole', 'yolo': [0.01, 0.99, 0.1, 0.1]}],
        [],
        [{'label': 'manhole', 'yolo': [0.5, 0.5, 1, 1]}],
        [{'label': 'crack', 'yolo': [0.8, 0.6, 0.1, 0.1]},
         {'label': 'manhole', 'yolo': [0.2, 0.6, 0.1, 0.1]},
         {'label': 'pothole', 'yolo': [0.5, 0.2, 0.1, 0.1]}],
        [{'label': 'crack', 'yolo': [0.3005, 0.4005, 0.05, 0.02]},
         {'label': 'manhole', 'yolo': [0.7, 0.8, 0.1, 0.08]}],
    ]
    for objects in fixtures:
        assert ns['objects_to_answer'](objects) == lab10.objects_to_answer(objects)
        for obj in objects:
            box = ns['yolo_to_box'](obj['label'], obj['yolo'])
            assert box == lab10.yolo_to_box(obj['label'], obj['yolo'])
            for width, height in ((640, 360), (1280, 720), (1920, 1080)):
                assert ns['norm1000_to_pixel'](box['bbox_2d'], width, height) == (
                    lab10.norm1000_to_pixel(box['bbox_2d'], width, height))


def test_load_model_uses_pinned_api():
    import ast
    import inspect
    import lab10

    source = strip_magics(section_code(3))
    function = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == 'load_model')
    assert [arg.arg for arg in function.args.args] == list(inspect.signature(lab10.load_model).parameters)
    body = ast.get_source_segment(source, function)
    for token in ('Qwen3VLForConditionalGeneration', 'BitsAndBytesConfig', '"nf4"',
                  'dtype=torch.float16'):
        assert token in body
    assert 'torch_dtype' not in body
    for code in (body, inspect.getsource(lab10.load_model)):
        keywords = [node for node in ast.walk(ast.parse(code))
                    if isinstance(node, ast.keyword) and node.arg == 'llm_int8_skip_modules']
        assert len(keywords) == 1
        assert isinstance(keywords[0].value, ast.List)
        assert ast.literal_eval(keywords[0].value) == ['visual', 'lm_head']


def test_e0_is_resumable():
    import ast
    import inspect
    import lab10

    source = strip_magics(section_code(3))
    for token in ('report/evidence/e0.json', 'work/e0.jsonl', 'scripts/score.py'):
        assert token in section_code(3)
    function = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == 'predict')
    assert [arg.arg for arg in function.args.args] == list(inspect.signature(lab10.predict).parameters)
    assert 'run_e0(' not in source
    # The evidence guard must enclose model loading.
    tree = ast.parse(source)
    guard = next(node for node in tree.body if isinstance(node, ast.If)
                 and 'report/evidence/e0.json' in ast.get_source_segment(source, node.test))
    assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
               and node.func.id == 'load_model' for node in ast.walk(ast.Module(body=guard.orelse, type_ignores=[])))
    prediction = ast.get_source_segment(source, function)
    assert 'if e["image_id"] not in done' in prediction
    assert 'do_sample=False' in prediction
    assert 'out[:, inputs["input_ids"].shape[1]:]' in prediction
