"""Offline structural checks and shared helpers for the lab notebook."""

import json
import ast
import builtins
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


def test_no_placeholders():
    for cell in load_nb()['cells']:
        assert 'YOUR CODE HERE' not in ''.join(cell['source'])


def test_lab10_imports_exist():
    import lab10

    tree = ast.parse(all_code())
    core = {'yolo_to_box', 'objects_to_answer', 'load_model', 'predict',
            'encode_example', 'make_collate', 'apply_setup', 'train', 'save_checkpoint'}
    definitions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert core <= definitions
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == 'lab10':
            for alias in node.names:
                assert alias.name != '*'
                assert hasattr(lab10, alias.name), alias.name
                assert alias.name not in core
                assert (alias.asname or alias.name) not in core
        elif (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
              and node.value.id == 'lab10'):
            assert hasattr(lab10, node.attr), node.attr


def test_notebook_called_functions_are_known():
    tree = ast.parse(all_code())
    known = set(dir(builtins)) | {'display', 'get_ipython'}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            known.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            known.add(node.id)
        elif isinstance(node, ast.arg):
            known.add(node.arg)
        elif isinstance(node, ast.Import):
            known.update(alias.asname or alias.name.split('.')[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            known.update(alias.asname or alias.name for alias in node.names)
    calls = {node.func.id for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert not calls - known, f'Unknown notebook calls: {sorted(calls - known)}'


def test_e4_rank_ablation():
    source = section_code(8)
    for token in ('e4_rank_', '(4, 64)', 'check_submission.py'):
        assert token in source
    assert '2 * r' in source or '2*r' in source
    tree = ast.parse(strip_magics(source))
    loop = next(node for node in tree.body if isinstance(node, ast.For))
    assert ast.literal_eval(loop.iter) == (4, 64)
    calls = [node.value for node in loop.body if isinstance(node, ast.Expr)]
    assert [call.func.id for call in calls] == ['train_setup', 'evaluate_setup']
    assert [ast.unparse(arg) for arg in calls[0].args] == ['name', 'BEST']
    assert {kw.arg: ast.unparse(kw.value) for kw in calls[0].keywords} == {
        'r': 'r', 'alpha': '2 * r'}
    final_calls = [node.value.func.id for node in tree.body
                   if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)]
    assert final_calls == ['make_figures', 'plot_losses', 'save_to_drive', 'bundle_outputs']


def assert_signatures_match_lab10(names):
    import ast
    import inspect
    import lab10

    definitions = {node.name: node for node in ast.parse(all_code()).body
                   if isinstance(node, ast.FunctionDef)}
    for name in names:
        args = definitions[name].args
        positional = args.posonlyargs + args.args
        defaults = [inspect.Parameter.empty] * (len(positional) - len(args.defaults))
        defaults += [ast.literal_eval(value) for value in args.defaults]
        parameters = [inspect.Parameter(
            arg.arg, inspect.Parameter.POSITIONAL_ONLY if i < len(args.posonlyargs)
            else inspect.Parameter.POSITIONAL_OR_KEYWORD, default=default)
            for i, (arg, default) in enumerate(zip(positional, defaults))]
        if args.vararg:
            parameters.append(inspect.Parameter(args.vararg.arg, inspect.Parameter.VAR_POSITIONAL))
        parameters.extend(inspect.Parameter(
            arg.arg, inspect.Parameter.KEYWORD_ONLY,
            default=inspect.Parameter.empty if default is None else ast.literal_eval(default))
            for arg, default in zip(args.kwonlyargs, args.kw_defaults))
        if args.kwarg:
            parameters.append(inspect.Parameter(args.kwarg.arg, inspect.Parameter.VAR_KEYWORD))
        assert inspect.Signature(parameters) == inspect.signature(getattr(lab10, name))


def test_core_defs_match_lab10_signatures():
    assert_signatures_match_lab10(('encode_example', 'make_collate', 'check_example', 'apply_setup'))


def test_train_defs_match_lab10_signatures():
    assert_signatures_match_lab10(('train', 'save_checkpoint', 'predict'))


def test_training_guide_details():
    import ast

    source = section_code(6)
    for token in ('GradScaler', 'get_cosine_schedule_with_warmup', 'use_reentrant',
                  'clip_grad_norm_', 'projector.pt', 'run.json', 'OutOfMemoryError', 'configs/'):
        assert token in source
    calls = [node for node in ast.parse(source).body if isinstance(node, ast.Expr)
             and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
             and node.value.func.id == 'train_setup']
    assert [ast.literal_eval(node.value.args[0]) for node in calls] == ['e1', 'e2', 'e3']


def test_scoring_uses_meta_and_resume():
    source = section_code(7)
    for token in ('scripts/score.py', '--meta', 'trainable_params', 'peak_vram_gb',
                  'train_minutes', 'report/evidence/', 'evaluate_setup(k)', 'check=True'):
        assert token in source


def test_no_one_line_lab10_runners():
    source = NB.read_text(encoding='utf-8')
    assert 'run_experiment(' not in source
    assert 'run_e0(' not in source


def test_evidence_skips_training_and_scoring(tmp_path, monkeypatch):
    import ast

    monkeypatch.chdir(tmp_path)
    evidence = Path('report/evidence')
    evidence.mkdir(parents=True)
    (evidence / 'e1.json').write_text('{}', encoding='utf-8')
    # Execute only the definitions: the guards must return before heavy imports or loading.
    namespace = {'Path': Path, 'save_to_drive': lambda: None}
    for section, name in ((6, 'train_setup'), (7, 'evaluate_setup')):
        node = next(node for node in ast.parse(section_code(section)).body
                    if isinstance(node, ast.FunctionDef) and node.name == name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<resume>', 'exec'), namespace)
    for _ in range(2):
        namespace['train_setup']('e1', 'E1')
        namespace['evaluate_setup']('e1')


def test_target_constants_match_lab10():
    import ast
    import re
    import lab10

    names = {'LLM_TARGETS', 'VIT_TARGETS', 'PROJECTOR_KEYS'}
    nodes = [node for node in ast.parse(section_code(5)).body
             if isinstance(node, ast.Assign) and any(
                 isinstance(target, ast.Name) and target.id in names for target in node.targets)]
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<targets>', 'exec'), namespace)
    for name in names:
        assert namespace[name] == getattr(lab10, name)
    for module in ('model.language_model.layers.3.mlp.down_proj',
                   'model.language_model.layers.0.self_attn.q_proj'):
        assert re.fullmatch(namespace['LLM_TARGETS'], module)
    assert not re.fullmatch(namespace['LLM_TARGETS'], 'model.visual.blocks.0.attn.qkv')
    assert re.fullmatch(namespace['VIT_TARGETS'], 'model.visual.blocks.5.mlp.linear_fc1')


def test_mask_loss_checks_present():
    source = section_code(4)
    for token in ('-100', '220', '<|im_end|>'):
        assert token in source


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
