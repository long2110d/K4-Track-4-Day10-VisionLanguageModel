"""Sinh notebook đầy đủ. Chạy: python scripts/make_notebook.py [--check]."""
import argparse
import json
from pathlib import Path


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.strip("\n").splitlines(True)}


CELLS = [
    md(r'''# Lab Day 10: Finetune Qwen3-VL-4B (LoRA) để định vị hư hại mặt đường
Notebook chứa toàn bộ mã của bài lab; chọn GPU **Colab T4** rồi **Run All** trong trình duyệt hoặc extension Colab của VS Code.
Mỗi thí nghiệm được bỏ qua khi đã có `report/evidence/<tên>.json` để chạy tiếp.
Lưu notebook rồi chạy `python extract_outputs.py` ở máy local để lấy gói kết quả.'''),
    md(r'''## 0. Môi trường (Colab T4)'''),
    code(r'''import os
import sys
import subprocess
import shutil
from pathlib import Path

REPO = 'https://github.com/long2110d/K4-Track-4-Day10-VisionLanguageModel.git'
USE_DRIVE = False
USE_KAGGLE_CLI = False
BASE = '/content/lab10' if os.path.isdir('/content') else os.getcwd()
if not os.path.exists(os.path.join(BASE, 'lab10.py')):
    subprocess.run(['git', 'clone', REPO, BASE], check=True)
else:
    subprocess.run(['git', '-C', BASE, 'pull', '--ff-only'], check=False)
os.chdir(BASE)
if BASE not in sys.path:
    sys.path.insert(0, BASE)
print(os.getcwd())

DRIVE_DIR = '/content/drive/MyDrive/lab10'
drive_ready = False
if USE_DRIVE:
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        os.makedirs(os.path.join(DRIVE_DIR, 'work'), exist_ok=True)
        if not os.path.exists('work'):
            os.symlink(os.path.join(DRIVE_DIR, 'work'), 'work')
        # Khôi phục bằng chứng và thống kê còn thiếu để chạy tiếp.
        for folder in ('evidence', 'extra'):
            source = Path(DRIVE_DIR) / 'report' / folder
            if source.exists():
                for file in source.rglob('*'):
                    if file.is_file():
                        target = Path('report') / folder / file.relative_to(source)
                        if not target.exists():
                            target.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(file, target)
        drive_ready = True
    except Exception as exc:
        print('Không thể gắn Drive; extension VS Code có thể không hỗ trợ drive.mount. '
              'Đặt USE_DRIVE = False để lưu cục bộ.', type(exc).__name__)


def save_to_drive():
    if not USE_DRIVE or not drive_ready:
        return
    for folder in ('evidence', 'extra'):
        source = Path('report') / folder
        if source.exists():
            shutil.copytree(source, Path(DRIVE_DIR) / 'report' / folder, dirs_exist_ok=True)'''),
    code(r'''!nvidia-smi -L
!pip -q install -r requirements-colab.txt kagglehub matplotlib
!python -m pytest -q'''),
    code(r'''# Khoá Kaggle là tuỳ chọn; không in giá trị bí mật.
try:
    from google.colab import userdata
    for key in ('KAGGLE_USERNAME', 'KAGGLE_KEY'):
        try:
            value = userdata.get(key)
        except Exception:
            value = None
        if value:
            os.environ[key] = value
    if not all(os.environ.get(key) for key in ('KAGGLE_USERNAME', 'KAGGLE_KEY')):
        print('kagglehub tải dataset công khai không cần khoá.')
except Exception:
    print('Không có Colab Secrets; kagglehub tải dataset công khai không cần khoá.')'''),
    md(r'''## 1. Dữ liệu: tải và chia split
Dataset công khai tải bằng `kagglehub` không cần khoá. Đặt `USE_KAGGLE_CLI = True` nếu muốn dùng Kaggle CLI.'''),
    code(r'''if os.path.exists('data/splits/meta.json'):
    print('Bỏ qua chuẩn bị dữ liệu: đã có data/splits/meta.json.')
elif USE_KAGGLE_CLI:
    !python scripts/prepare_data.py --kaggle
else:
    import kagglehub
    src = kagglehub.dataset_download('lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes')
    !python scripts/prepare_data.py --source "{src}"
assert os.path.exists('data/splits/train.jsonl')'''),
    code(r'''import json
from PIL import Image, ImageDraw
from IPython.display import display
from scripts.labcommon import CLASSES
# Giữ các tên mà những ô phần 2–8 hiện tại sử dụng; các nhiệm vụ sau thay mã lõi.
from lab10 import (
    objects_to_answer, norm1000_to_pixel, PROMPT, draw_boxes, run_e0,
    bundle_outputs, LLM_TARGETS, VIT_TARGETS, PROJECTOR_KEYS, run_experiment,
    best_setup, coord_check, make_figures, plot_losses,
)

split_rows = {}
for split in ('train', 'val', 'test'):
    with open(f'data/splits/{split}.jsonl', encoding='utf-8') as file:
        split_rows[split] = [json.loads(line) for line in file if line.strip()]
train_rows, val_rows, test_rows = (split_rows[s] for s in ('train', 'val', 'test'))
sessions = {}
for split, rows in split_rows.items():
    sessions[split] = {row['session'] for row in rows}
    print(split, 'số ảnh:', len(rows), 'số box:', sum(len(row['objects']) for row in rows))
    for label in CLASSES:
        image_count = sum(any(obj['label'] == label for obj in row['objects']) for row in rows)
        box_count = sum(obj['label'] == label for row in rows for obj in row['objects'])
        print(' ', label, 'số ảnh:', image_count, 'số box:', box_count)
assert not (sessions['train'] & sessions['val'] or sessions['train'] & sessions['test']
            or sessions['val'] & sessions['test']), 'Có phiên quay xuất hiện ở hai split!'
print('Không có phiên quay trùng giữa các split.')

# Vẽ trực tiếp nhãn YOLO trên sáu ảnh huấn luyện, rồi ghép lưới 3 × 2.
assert len(train_rows) >= 6, 'Cần ít nhất sáu ảnh huấn luyện để xem nhãn.'
grid = Image.new('RGB', (3 * 640, 2 * 360), 'white')
for i, row in enumerate(train_rows[:6]):
    with Image.open(Path('data') / row['image_path']) as source:
        image = source.convert('RGB')
    width, height = image.size
    draw = ImageDraw.Draw(image)
    for obj in row['objects']:
        xc, yc, w, h = obj['yolo']
        x1, y1 = max(0, (xc - w / 2) * width), max(0, (yc - h / 2) * height)
        x2, y2 = min(width, (xc + w / 2) * width), min(height, (yc + h / 2) * height)
        draw.rectangle((x1, y1, x2, y2), outline=(0, 220, 0), width=2)
        draw.text((x1 + 2, y1 + 2), obj['label'], fill=(0, 220, 0))
    grid.paste(image.resize((640, 360)), ((i % 3) * 640, (i // 3) * 360))
Path('report/figures').mkdir(parents=True, exist_ok=True)
grid.save('report/figures/train_samples.jpg', quality=90)
display(grid)'''),
    md(r'''## 2. Nhãn: YOLO → JSON thang 0–1000'''),
    code(r'''from scripts.score import _gt_boxes
e = next(r for r in train_rows if len(r['objects']) >= 2)
ans = objects_to_answer(e['objects'])
print(ans)
# quy ngược về pixel và đối chiếu với nhãn YOLO gốc (sai số chỉ do làm tròn)
gt = sorted((g['bbox'] for g in _gt_boxes(e)), key=lambda b: (round(b[1]), round(b[0])))
for b, g in zip(json.loads(ans), gt):
    print(b['label'], [round(v, 1) for v in norm1000_to_pixel(b['bbox_2d'], 640, 360)],
          [round(v, 1) for v in norm1000_to_pixel(g, 640, 360)])
print(PROMPT)
draw_boxes(e, raw=ans, title='xanh = GT, do = JSON da quy doi')'''),
    md(r'''## 3. E0: zero-shot'''),
    code(r'''# nạp model 4-bit (ViT + lm_head fp16), đếm tham số từng khối, kiểm tra mask loss, suy luận test, chấm
run_e0()
print(json.load(open('report/extra/model_blocks.json')))
for r in json.load(open('report/extra/e0_samples.json'))['first']:
    print(repr(r['raw'][:300]))
bundle_outputs()'''),
    md(r'''## 4. Dựng dữ liệu huấn luyện: chat template và mask loss
`encode_example` (trong `lab10.py`) đã chạy ở ô E0 (`check_example`): token ảnh phải là **220**, phần `labels != -100` giải mã ra đúng JSON + `<|im_end|>`.'''),
    code(r'''tok = json.load(open('report/extra/token_stats.json'))
print(tok)
assert tok['image'] == 220, tok'''),
    md(r'''## 5. Setup LoRA / đóng băng cho E1, E2, E3
`apply_setup`: E1 LoRA trên 252 module LLM; E2 + projector fp32 train đầy đủ; E3 + LoRA trên 96 module ViT (tổng 348).'''),
    code(r'''print('LLM :', LLM_TARGETS)
print('ViT :', VIT_TARGETS)
print('projector:', PROJECTOR_KEYS)'''),
    md(r'''## 6. Vòng huấn luyện (mỗi setup ≈ 20–40 phút + ≈ 10 phút suy luận)'''),
    code(r'''run_experiment('e1', 'E1')
bundle_outputs()'''),
    code(r'''run_experiment('e2', 'E2')
bundle_outputs()'''),
    code(r'''run_experiment('e3', 'E3')
bundle_outputs()'''),
    md(r'''## 7. Suy luận và chấm E1–E3'''),
    code(r'''for k in ('e0', 'e1', 'e2', 'e3'):
    d = json.load(open(f'report/evidence/{k}.json'))
    print(k.upper(), 'F1=%.3f' % d['micro']['f1'], 'CI95=[%.3f, %.3f]' % tuple(d['f1_ci95']),
          'valid=%.3f' % d['parse_valid_rate'], 'meta=', d['meta'])
BEST = best_setup()
coord_check(BEST.lower())     # Câu 3: chấm sai thang toạ độ'''),
    md(r'''## 8. E4 (ablation tự chọn) và E5 (tuỳ chọn, GRPO)
E4: đổi **rank LoRA** trên setup tốt nhất của E1–E3, r = 4 và r = 64 (alpha = 2r). E5: không làm.'''),
    code(r'''for r in (4, 64):
    run_experiment(f'e4_rank_{r}', BEST, r=r, alpha=2 * r)
    bundle_outputs()'''),
    code(r'''make_figures(BEST.lower())
plot_losses()
!python scripts/check_submission.py
bundle_outputs()   # gói cuối cùng: evidence + figures + extra'''),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Kiểm tra notebook đồng bộ với mã sinh")
    args = parser.parse_args()
    nb = {"cells": CELLS,
          "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                       "kernelspec": {"name": "python3", "display_name": "Python 3"}},
          "nbformat": 4, "nbformat_minor": 5}
    path = Path(__file__).resolve().parents[1] / "lab_day10.ipynb"
    generated = json.dumps(nb, ensure_ascii=False, indent=1)
    if args.check:
        if not path.exists() or path.read_text(encoding="utf-8") != generated:
            print("Notebook is out of sync: run python scripts/make_notebook.py then git diff -- lab_day10.ipynb")
            return 1
        return 0
    path.write_text(generated, encoding="utf-8")
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
