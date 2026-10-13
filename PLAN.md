# PLAN: make `lab_day10.ipynb` a complete, self-contained Run All notebook for Colab T4

## Project overview

**Goal.** `lab_day10.ipynb` must run top to bottom with **Run All** on a Google Colab T4 runtime, from the browser or from the Colab extension in VS Code. The notebook must contain the **full lab code inside its cells**, in the cell order and under the section headings of the original skeleton (commit `c2f3997`). The grader must be able to read the logic in the notebook, so a cell must not just make one-line calls into `lab10.py`. `lab10.py` stays as a library, and the notebook must stay consistent with it. Nothing in this plan needs a GPU. Every test runs offline.

**Stack.** Python 3.10+. Pinned GPU stack is in `requirements-colab.txt`: `transformers==5.19.0`, `peft==0.21.2`, `trl==1.9.2`, `accelerate==1.15.0`, `bitsandbytes==0.50.2`, plus Colab's own torch. Tests use `pytest` with the standard library only, no `nbformat`. Do NOT import torch, transformers or peft in tests.

**Repository layout (relevant parts).**
- `lab_day10.ipynb`: the notebook (nbformat 4). It is **generated**; see below.
- `scripts/make_notebook.py`: builds `lab_day10.ipynb` from a `CELLS` list (`md(...)` and `code(...)` helpers). Right now it emits the old empty skeleton, while the committed notebook is a hand-edited version that calls `lab10.py` helpers on one line each (`run_e0()`, `run_experiment(...)`).
- `lab10.py`: library with the reference implementation. Contains `PROMPT`, `LLM_TARGETS`, `VIT_TARGETS`, `PROJECTOR_KEYS`, `MODEL_ID`, `yolo_to_box`, `objects_to_answer`, `norm1000_to_pixel`, `open_image`, `draw_boxes`, `show_grid`, `split_stats`, `user_message`, `load_model`, `load_projector`, `projector_state`, `predict`, `encode_example`, `make_collate`, `check_example`, `apply_setup`, `count_blocks`, `train`, `save_checkpoint`, `score`, `per_image_f1`, `free`, `run_experiment`, `run_e0`, `data_stats`, `token_stats`, `coord_check`, `best_setup`, `make_figures`, `plot_losses`, `bundle_outputs`. It imports heavy libraries only inside functions, so `import lab10` works offline.
- `scripts/prepare_data.py` (CLI: `--source DIR | --zip PATH | --kaggle`), `scripts/score.py` (CLI: `--pred --name --out [--split] [--coord] [--meta k=v]...`; also exposes `parse_output`, `evaluate`, `_gt_boxes`), `scripts/check_submission.py`, `scripts/labcommon.py` (`CLASSES`, `read_jsonl`).
- `configs/e0.yaml … e5.yaml`: hyper-parameters. E1–E3 have keys `lora: {r, alpha, dropout, targets}`, `lr`, `batch_size`, `grad_accum`, `max_steps`, `seed`, `train_projector`.
- `tests/`: instructor tests. `tests/test_configs_and_docs.py::test_notebook_sections_and_no_secrets` requires markdown cells whose first source line starts with `## 0.` … `## 8.`, and no hard-coded Kaggle key.
- `test_lab10.py` (repo root): offline tests for label helpers.
- `GUIDE.md` (step-by-step pseudo code, Bước 0–9), `README.md`, `RUBRIC.md`, `REPORT.md`.
- Outputs: `data/` (git-ignored), `work/` (git-ignored: predictions `work/<name>.jsonl`, adapters `work/adapter_<name>/`), `report/evidence/<name>.json` (scored evidence), `report/extra/`, `report/figures/`.

**Hard constraints (all tasks).**
- Do NOT modify `scripts/prepare_data.py`, `scripts/score.py`, `scripts/check_submission.py` or anything under `tests/`. New tests go in a **new root-level file `test_notebook.py`**.
- The notebook is edited **only through `scripts/make_notebook.py`**. After every change, run `python scripts/make_notebook.py` to regenerate `lab_day10.ipynb` and commit both files. Write cell sources as raw triple-quoted strings (`r'''…'''`) so regexes and backslashes survive. Keep `json.dumps(nb, ensure_ascii=False, indent=1)` output and the existing notebook metadata (`accelerator: GPU`, `colab`, `kernelspec python3`).
- Keep the original skeleton order, with markdown headings `# Lab Day 10…`, `## 0. Môi trường (Colab T4)`, `## 1. Dữ liệu: tải và chia split`, `## 2. Nhãn: YOLO → JSON thang 0–1000`, `## 3. E0: zero-shot`, `## 4. Dựng dữ liệu huấn luyện: chat template và mask loss`, `## 5. Setup LoRA / đóng băng cho E1, E2, E3`, `## 6. Vòng huấn luyện`, `## 7. Suy luận và chấm E1–E3`, `## 8. E4 (ablation tự chọn) và E5 (tuỳ chọn, GRPO)`. Each heading must be the first line of its markdown cell. Extra explanatory text below a heading, and extra code cells inside a section, are allowed.
- All paths inside the notebook are **relative to the repo root** (`data/...`, `work/...`, `report/...`, `configs/...`, `scripts/...`). The only absolute paths allowed are in the first code cell: `/content`, `/content/lab10` and `/content/drive/...`.
- Notebook prose and comments are in Vietnamese, matching the existing style. Code identifiers match `lab10.py`.
- The notebook must **not** use `from lab10 import *`. It imports only non-core helpers by name, for example `from lab10 import draw_boxes, show_grid, bundle_outputs`. It **defines itself** (with `def`) the core functions listed in each task. Their names and parameter lists must match `lab10.py`.
- Every experiment is resumable. If `report/evidence/<name>.json` already exists, the cell prints a skip message and does no model loading.
- Code cells may use IPython `!` or `%` lines, including indented ones inside `if` blocks. The tests replace each such line with `pass` at the same indentation and then `compile()` the cell, so every cell must compile after that transformation.

**Testing conventions.** `pytest.ini` has `pythonpath = .`. Tests read files with `encoding="utf-8"` (the dev machine is Windows). Keep test output compact and avoid prints.

TEST_CMD: python -m pytest -q

---

## Task 1: Test configuration and notebook test harness

Make `python -m pytest -q` also collect the root-level tests, and add the shared notebook test helpers plus structural tests that pass on the current notebook.

**Files**
- Modify `pytest.ini`: `testpaths = tests test_lab10.py test_notebook.py`. Keep `pythonpath = .`.
- Create `test_notebook.py` with:
  - `ROOT = Path(__file__).resolve().parent`, `NB = ROOT / "lab_day10.ipynb"`.
  - Helpers:
    - `load_nb()` returns the parsed JSON.
    - `cells(kind)` returns cells of that type, each with its source joined into one string.
    - `section_code(n)` returns the joined sources of the code cells between the markdown heading `## n.` and the next `## ` heading.
    - `strip_magics(src)` replaces each line whose stripped form starts with `!` or `%` with `<same indentation>pass`.
    - `all_code()` returns all code cells joined, after `strip_magics`.
  - Tests:
    1. `test_notebook_is_valid_json`: `nbformat == 4`; every cell has `cell_type` and `source`; every code cell has `outputs == []` and `execution_count is None` (the notebook is committed clean).
    2. `test_headings_in_skeleton_order`: the ten headings listed in the overview appear as the first lines of markdown cells, in that order. Match on the prefixes `# Lab Day 10`, `## 0.` … `## 8.`.
    3. `test_every_code_cell_compiles`: `compile(strip_magics(src), f"cell{i}", "exec")` succeeds for every code cell.

**Acceptance criteria**
- `python -m pytest -q` runs `tests/`, `test_lab10.py` and `test_notebook.py`, and all pass on the current notebook.
- No other files changed.

## Task 2: Generator owns the notebook; section 0 bootstrap and section 1 data cells

Rewrite `scripts/make_notebook.py` so that it emits the full notebook, then regenerate the notebook. This task implements sections 0 and 1. Sections 2–8 keep the current cell contents for now: copy them verbatim from the current `lab_day10.ipynb` into `CELLS`. Later tasks replace them.

**Files**
- Modify `scripts/make_notebook.py`. Add `--check`: it exits 1 and prints a diff hint when the regenerated JSON differs from the file on disk.
- Regenerate `lab_day10.ipynb`.
- Modify `test_notebook.py` (add tests).

**Cell content**
- Title markdown cell. Say that the notebook contains the full code, that Run All works on Colab T4, and that each experiment is skipped when its evidence exists.
- `## 0.` markdown, then the **first code cell (bootstrap)**. It must run before anything else and use only the standard library:
  - Set `REPO = 'https://github.com/long2110d/K4-Track-4-Day10-VisionLanguageModel.git'`, `USE_DRIVE = False` and `USE_KAGGLE_CLI = False`.
  - Set `BASE = '/content/lab10' if os.path.isdir('/content') else os.getcwd()`.
  - If `BASE/lab10.py` is missing, run `subprocess.run(['git','clone',REPO,BASE], check=True)`. Otherwise run `git -C BASE pull --ff-only` with `check=False`.
  - Call `os.chdir(BASE)`, add `BASE` to `sys.path` if not already there, and print `os.getcwd()`.
  - If `USE_DRIVE`, wrap this in try/except so that it prints a hint in the VS Code extension, where `drive.mount` is unavailable:
    - Mount Drive and make `DRIVE_DIR = '/content/drive/MyDrive/lab10'`.
    - Symlink `work` to `DRIVE_DIR/work` if `work` does not exist.
    - Copy any `DRIVE_DIR/report/{evidence,extra}` files missing locally into `report/` (restore for resume).
  - Define `def save_to_drive():`. It copies `report/evidence` and `report/extra` to `DRIVE_DIR/report/` when `USE_DRIVE`, and does nothing otherwise. Later cells call it after each experiment.
- Second code cell:
  - `!nvidia-smi -L`
  - `!pip -q install -r requirements-colab.txt kagglehub matplotlib`
  - `!python -m pytest -q`
- Third code cell (optional Kaggle secrets). Wrap it in try/except: read `KAGGLE_USERNAME` and `KAGGLE_KEY` from `google.colab.userdata` into `os.environ` only if present, and never print them. If they are missing, print that kagglehub works without a key.
- `## 1.` markdown, then the data code cell:
  - If `data/splits/meta.json` exists, print a skip message.
  - Else if `USE_KAGGLE_CLI`, run `!python scripts/prepare_data.py --kaggle`.
  - Otherwise, call `import kagglehub; src = kagglehub.dataset_download('lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes')`, then run `!python scripts/prepare_data.py --source "{src}"`.
  - Then `assert os.path.exists('data/splits/train.jsonl')`.
- Next code cell, written **inline** (GUIDE Bước 1):
  - Read `data/splits/{train,val,test}.jsonl` with `json.loads` per line.
  - Print image and box counts per split and class.
  - Check that no `session` appears in two splits.
  - Draw the boxes of 6 training images with PIL `ImageDraw`. Convert YOLO `[xc,yc,w,h]` to pixels inline.
  - Save the grid to `report/figures/train_samples.jpg` and display it.
  - Keep `train_rows`, `val_rows` and `test_rows` as globals.

**Tests to add in `test_notebook.py`**
- `test_generator_in_sync`: `subprocess.run([sys.executable, "scripts/make_notebook.py", "--check"], cwd=ROOT)` returns 0.
- `test_first_code_cell_clones_and_chdirs`: the first code cell contains the repo URL, `'/content/lab10'`, `git` with `clone`, and `os.chdir`. Its text contains no `!` or `%` magic lines. `os.chdir` appears before the first non-import statement that touches files, so it is enough to check that `os.chdir(` occurs before `pip` or `prepare_data` anywhere in the notebook.
- `test_no_absolute_paths_after_bootstrap`: no code cell after the first contains `/content`.
- `test_data_cell`: section 1 code contains `kagglehub`, `prepare_data.py --kaggle`, `prepare_data.py --source`, `data/splits/meta.json` and `data/splits/train.jsonl`.
- `test_no_secrets_printed`: no code cell contains `print(` on the same line as `KAGGLE_KEY`.

**Acceptance criteria**
- `python scripts/make_notebook.py --check` exits 0.
- All tests pass, including `tests/test_configs_and_docs.py`.

## Task 3: Section 2 labels and section 3 E0 zero-shot written inline

Replace the section 2 and section 3 cells in `scripts/make_notebook.py` with full inline code, regenerate the notebook, and add tests.

**Section 2 cell (GUIDE Bước 2).** Defines inline, identical in behaviour to `lab10.py`:
- `PROMPT`: the exact same string as `lab10.PROMPT`.
- `def yolo_to_box(label, yolo)`: round each value and clamp it to [0, 1000].
- `def objects_to_answer(objects)`: sort by `(y1, x1)` and serialise with `json.dumps(..., separators=(",", ":"))`.
- `def norm1000_to_pixel(bbox, width, height)`.

Then the cell runs the round-trip check:
- Pick a train row with at least 2 objects.
- Print the JSON answer.
- Convert back to pixels and compare with pixel boxes computed directly from YOLO. Assert a max abs error ≤ 1 px.
- Draw the result using `draw_boxes` imported from `lab10`. This is allowed because it is a plotting helper.

The cell must start with the imports it needs (`import json`) and must not depend on any notebook global other than `train_rows`. The test executes the `def`s and `PROMPT` in isolation, so put the round-trip check after them under a clear comment.

**Section 3 cells (GUIDE Bước 3).**
- A cell that defines inline `def user_message(image)`, `def open_image(entry)` (path `data/` + `entry['image_path']`) and `def load_model(adapter_dir=None, model_id=MODEL_ID)`. `MODEL_ID = 'Qwen/Qwen3-VL-4B-Instruct'`. `load_model` uses:
  - `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True, llm_int8_skip_modules=["visual", "lm_head"])`.
  - `Qwen3VLForConditionalGeneration.from_pretrained(model_id, quantization_config=quant, dtype=torch.float16, device_map={"": 0})`. Use `dtype=`, not `torch_dtype=`: transformers 5.x renamed it.
  - `AutoProcessor`, then `padding_side='left'`.
  - When `adapter_dir` is given, load it with `PeftModel.from_pretrained` and reload `projector.pt` if present. Define `load_projector(model, path)` and `projector_state(model)` inline, same as `lab10`.
- In the same cell or the next one, `def predict(model, processor, entries, out_path, batch_size=4, max_new_tokens=256)`. It does batched greedy generation via `processor.apply_chat_template(..., tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt", padding=True)`. It decodes only the new tokens and appends `{"image_id","raw"}` lines. Image ids already present in `out_path` are skipped (resume).
- The E0 run cell:
  - If `report/evidence/e0.json` exists, skip.
  - Otherwise: load the model; save `count_blocks(model)` (imported from `lab10`) to `report/extra/model_blocks.json`; run `predict(...)` on `test_rows` into `work/e0.jsonl`; `del model`, `gc.collect()`, `torch.cuda.empty_cache()`.
  - Then run `!python scripts/score.py --pred work/e0.jsonl --name E0 --out report/evidence/e0.json`. This keeps the original skeleton's score cell. Guard it with `if not os.path.exists('report/evidence/e0.json'):`.
  - Print 5 raw outputs and the `parse_valid_rate`.
  - Call `save_to_drive()`.

**Tests to add**
- `test_label_functions_match_lab10`:
  - Take section 2 code and run `strip_magics`.
  - Use `ast` to keep only the `Import`/`ImportFrom` nodes, the `FunctionDef`s `yolo_to_box`, `objects_to_answer` and `norm1000_to_pixel`, and the `PROMPT` assignment.
  - `exec` them in a fresh dict.
  - Assert `ns["PROMPT"] == lab10.PROMPT`.
  - On ~6 fixtures, assert that the outputs equal those of `lab10`. Fixtures: the two cases in `test_lab10.py`, an empty list, boxes that touch the edges, and unsorted objects.
- `test_load_model_uses_pinned_api`:
  - Parse section 3 code with `ast`.
  - The `load_model` `FunctionDef` exists and its argument names equal those of `lab10.load_model` (`inspect.signature`).
  - Its source contains `Qwen3VLForConditionalGeneration`, `BitsAndBytesConfig`, `"nf4"` and `dtype=torch.float16`, and does not contain `torch_dtype`.
  - The `llm_int8_skip_modules` keyword is a list literal equal to `["visual", "lm_head"]`.
  - Apply the same `llm_int8_skip_modules` check to `lab10.load_model` via `inspect.getsource`.
- `test_e0_is_resumable`: section 3 code contains `report/evidence/e0.json`, `work/e0.jsonl` and `scripts/score.py`, and defines `predict` with the same parameter names as `lab10.predict`.

**Acceptance criteria**
- Generator in sync, all tests pass.
- Sections 2 and 3 contain no call to `lab10.run_e0`.

## Task 4: Sections 4–5, training examples with loss mask, and the LoRA/freeze setup

Replace the section 4 and section 5 cells in the generator, regenerate, and add tests.

**Section 4 (GUIDE Bước 4).**
- Define inline `def encode_example(processor, image, answer)`, matching `lab10.encode_example`:
  - Build the prompt and full encodings with `apply_chat_template`.
  - Assert that the prompt is a prefix of the full encoding.
  - Truncate after the last `<|im_end|>`. Truncation keys: `input_ids`, `attention_mask`, `mm_token_type_ids`, `token_type_ids`.
  - Set `labels[:, :n] = -100`.
- Define `def make_collate(processor)`, returning `collate(entries)`. It pads `input_ids` with the pad id, `labels` with -100 and other sequence keys with 0, and concatenates `pixel_values`/`image_grid_thw` without padding.
- Define `def check_example(processor, entry)`.
- Check cell:
  - If no `processor` global exists, create one with `AutoProcessor.from_pretrained(MODEL_ID)`. This is cheap; no model load is needed.
  - Run `check_example` on `train_rows[0]` and `assert n_img == 220`.
  - Assert that the decoded answer ends with `<|im_end|>` and that, once that suffix is stripped, it equals `objects_to_answer(...)`.
  - Save `token_stats` (imported from `lab10`, or computed inline) to `report/extra/token_stats.json`.

**Section 5 (GUIDE Bước 5).**
- Define inline the `LLM_TARGETS`, `VIT_TARGETS` and `PROJECTOR_KEYS` constants with the exact same values as `lab10`.
- Define `def apply_setup(model, setup, r=16, alpha=32, dropout=0.05)`, matching `lab10.apply_setup`:
  - Freeze everything, then call `enable_input_require_grads()`.
  - Use the regex target `LLM_TARGETS` for E1/E2 and `({LLM})|({VIT})` for E3.
  - Print the number of `re.fullmatch` module matches with the expected 252/348.
  - Call `get_peft_model` with `LoraConfig(r, lora_alpha, lora_dropout, target_modules=<regex str>, task_type="CAUSAL_LM")`.
  - For E2/E3, cast the projector params to fp32 and set `requires_grad=True`.
  - Cast all trainable params to fp32 (GradScaler).
  - Print the trainable count, with the projector share.
  - Return `(model, n_matched, trainable)`.
- Add a short markdown note that sets the expected counts against GUIDE: 252 / 348 modules, and the E2 − E1 difference ≈ 109M parameters.

**Tests to add**
- `test_core_defs_match_lab10_signatures`: for `encode_example`, `make_collate`, `check_example` and `apply_setup`, the notebook has a top-level `def` (found via `ast` over `all_code()`) whose positional and keyword argument names and default literals equal `inspect.signature(getattr(lab10, name))`.
- `test_target_constants_match_lab10`: exec only the `LLM_TARGETS`, `VIT_TARGETS` and `PROJECTOR_KEYS` assignments from section 5 and compare them with `lab10`. Using `re`, check that `LLM_TARGETS` fully matches `model.language_model.layers.3.mlp.down_proj` and `model.language_model.layers.0.self_attn.q_proj`, and does not match `model.visual.blocks.0.attn.qkv`. `VIT_TARGETS` must fully match `model.visual.blocks.5.mlp.linear_fc1`.
- `test_mask_loss_checks_present`: section 4 code contains `-100`, `220` and `<|im_end|>`.

**Acceptance criteria**
- Generator in sync, all tests pass.
- Sections 4–5 contain no call to `lab10.` functions except `token_stats`.

## Task 5: Sections 6–7, training loop with resume, then inference and scoring of E1–E3

Replace the section 6 and section 7 cells in the generator, regenerate, and add tests.

**Section 6 (GUIDE Bước 6).**
- Define inline `def train(model, processor, out_dir, setup, lr=2e-4, batch_size=2, grad_accum=4, max_steps=300, seed=42, train_fraction=1.0, save_every=100, log_every=10)`, matching `lab10.train`:
  - AdamW with `weight_decay=0`; `get_cosine_schedule_with_warmup` with 5% warmup; `torch.amp.GradScaler("cuda")`.
  - Gradient checkpointing with `use_reentrant=False`; `use_cache=False`.
  - Gradient accumulation; fp16 autocast; raise when the loss is not finite; `unscale_`, clip 1.0, `step`, `update`.
  - Call `save_checkpoint` every `save_every` steps and at the end.
  - Write `run.json` (`trainable_params`, `peak_vram_gb`, `train_minutes`, plus hyper-parameters) and `loss.json`.
- Define `def save_checkpoint(model, out_dir, setup)`: call `save_pretrained`, and for E2/E3 also write `torch.save(projector_state(model), out_dir/'projector.pt')`.
- Define `def train_setup(name, setup, r=None, alpha=None)`:
  - Skip if `report/evidence/{name}.json` or `work/adapter_{name}/run.json` exists.
  - Read `configs/{setup.lower()}.yaml`. `r` and `alpha` default to `cfg['lora']`.
  - Load the model, call `apply_setup`, then `train(...)` with the config values.
  - On `torch.cuda.OutOfMemoryError`, free memory and retry once with `batch_size=1, grad_accum=batch_size*grad_accum` (GUIDE "Khi gặp sự cố").
  - Add `lora_modules`, `r` and `alpha` to `run.json`.
  - Copy `run.json` and `loss.json` to `report/extra/runs/{name}.json` and `report/extra/loss/{name}.json`.
  - Free GPU memory and call `save_to_drive()`.
- Three code cells: `train_setup('e1', 'E1')`, `train_setup('e2', 'E2')`, `train_setup('e3', 'E3')`.

**Section 7 (GUIDE Bước 7).**
- Define inline `def evaluate_setup(name, coord='norm1000')`:
  - Skip if `report/evidence/{name}.json` exists.
  - Otherwise call `load_model(f'work/adapter_{name}')`, which reloads `projector.pt`.
  - Run `predict` on `test_rows` into `work/{name}.jsonl`, then free memory.
  - Run `scripts/score.py` with `subprocess.run([sys.executable, 'scripts/score.py', '--pred', ..., '--name', name.upper(), '--out', f'report/evidence/{name}.json', '--coord', coord, '--meta', f'trainable_params=…', '--meta', f'peak_vram_gb=…', '--meta', f'train_minutes=…'], check=True)`, taking the `--meta` values from `work/adapter_{name}/run.json`.
  - Call `save_to_drive()`.
- A cell that loops `for k in ('e1','e2','e3'): evaluate_setup(k)`. It then prints a table for E0–E3 from the evidence files: micro F1, CI95, parse_valid_rate and meta.
- Choose `BEST` as the argmax of F1 over e1–e3 (inline, or `best_setup` from `lab10`).
- Call `coord_check(BEST.lower())` from `lab10` (GUIDE Câu 3).

**Tests to add**
- `test_train_defs_match_lab10_signatures`: `train`, `save_checkpoint` and `predict` have the same signatures as in `lab10`, using the Task 4 helper.
- `test_training_guide_details`: section 6 code contains `GradScaler`, `get_cosine_schedule_with_warmup`, `use_reentrant`, `clip_grad_norm_`, `projector.pt`, `run.json`, `OutOfMemoryError` and `configs/`. It calls `train_setup` for `e1`, `e2` and `e3` in that order.
- `test_scoring_uses_meta_and_resume`: section 7 code contains `scripts/score.py`, `--meta`, `trainable_params`, `peak_vram_gb`, `train_minutes` and `report/evidence/`, and calls `evaluate_setup`.
- `test_no_one_line_lab10_runners`: the notebook text contains neither `run_experiment(` nor `run_e0(`.

**Acceptance criteria**
- Generator in sync, all tests pass.
- Running sections 6–7 twice does not retrain or rescore when the evidence exists. This is checked statically by the tests above.

## Task 6: Section 8 E4 ablation, final cells, and the global consistency test

Replace the section 8 cells in the generator, regenerate, and add the final tests.

**Section 8 (GUIDE Bước 8–9).**
- Markdown: E4 changes the **LoRA rank** of the best setup, with r = 4 and r = 64 and alpha = 2r, and nothing else (`configs/e4.yaml` option `rank`). E5 is not done.
- Code cell:
  ```
  for r in (4, 64):
      name = f'e4_rank_{r}'
      train_setup(name, BEST, r=r, alpha=2*r)
      evaluate_setup(name)
  ```
  Evidence goes to `report/evidence/e4_rank_{r}.json`, per GUIDE `e4_<biến>_<giá trị>.json`. Each step is skipped when its evidence exists.
- Final code cell:
  - `make_figures(BEST.lower())` and `plot_losses()`, imported from `lab10`.
  - `!python scripts/check_submission.py`.
  - `save_to_drive()`.
  - `bundle_outputs()` from `lab10`.
  - A comment that `python extract_outputs.py` on the local machine unpacks the bundle.

**Tests to add**
- `test_no_placeholders`: no cell contains `YOUR CODE HERE`.
- `test_lab10_imports_exist`:
  - Parse `all_code()` with `ast`.
  - Every `ImportFrom` with `module == "lab10"` lists only names that exist as attributes of `lab10`, and none is `*`.
  - Every `Attribute` access `lab10.X` refers to an existing attribute.
  - The core functions `yolo_to_box`, `objects_to_answer`, `load_model`, `predict`, `encode_example`, `make_collate`, `apply_setup`, `train` and `save_checkpoint` are **defined** in the notebook and **not** imported from `lab10`.
- `test_notebook_called_functions_are_known`: every bare-name call `f(...)` in `all_code()` resolves to one of:
  - a name defined in the notebook (`def`, assignment, import, `for` target),
  - a name imported from `lab10` that exists there,
  - a Python builtin,
  - `display` or `get_ipython`.
- `test_e4_rank_ablation`: section 8 code contains `e4_rank_`, `(4, 64)`, `2 * r` or `2*r`, and `check_submission.py`.

**Acceptance criteria**
- `python scripts/make_notebook.py --check` exits 0.
- `python -m pytest -q` passes: `tests/`, `test_lab10.py`, `test_notebook.py`.
- The notebook has no `YOUR CODE HERE`.
- Every section 1–8 code cell contains real logic: no section's code consists only of single calls to `lab10` functions.
