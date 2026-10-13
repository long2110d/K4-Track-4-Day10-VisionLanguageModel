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
    draw_boxes,
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
    code(r'''import json

PROMPT = (
    "You are inspecting a dashcam road image for damage. "
    "Detect every pothole, crack and manhole. "
    "Answer with ONLY a JSON list, no other text, in the form "
    '[{"label": "pothole", "bbox_2d": [x1, y1, x2, y2]}]. '
    "label must be one of: pothole, crack, manhole. "
    "bbox_2d are integers on a 0-1000 scale relative to the image width and height "
    "(0,0 is the top-left corner, 1000,1000 the bottom-right). "
    "If there is no damage, answer []."
)


def yolo_to_box(label, yolo):
    xc, yc, w, h = yolo
    box = [(xc - w / 2) * 1000, (yc - h / 2) * 1000, (xc + w / 2) * 1000, (yc + h / 2) * 1000]
    return {"label": label, "bbox_2d": [min(1000, max(0, round(v))) for v in box]}


def objects_to_answer(objects):
    boxes = [yolo_to_box(o["label"], o["yolo"]) for o in objects]
    boxes.sort(key=lambda b: (b["bbox_2d"][1], b["bbox_2d"][0]))
    return json.dumps(boxes, separators=(",", ":"))


def norm1000_to_pixel(bbox, width, height):
    x1, y1, x2, y2 = bbox
    return [x1 / 1000 * width, y1 / 1000 * height, x2 / 1000 * width, y2 / 1000 * height]

# Kiểm tra vòng chuyển đổi sau các định nghĩa độc lập.
def check_label_roundtrip(rows):
    from PIL import Image
    from IPython.display import display
    from lab10 import draw_boxes

    entry = next(row for row in rows if len(row['objects']) >= 2)
    answer = objects_to_answer(entry['objects'])
    print(answer)
    with Image.open('data/' + entry['image_path']) as image:
        width, height = image.size
    objects = sorted(entry['objects'], key=lambda obj: (
        yolo_to_box(obj['label'], obj['yolo'])['bbox_2d'][1],
        yolo_to_box(obj['label'], obj['yolo'])['bbox_2d'][0]))
    errors = []
    for box, obj in zip(json.loads(answer), objects):
        xc, yc, w, h = obj['yolo']
        direct = [max(0, min(width, (xc - w / 2) * width)),
                  max(0, min(height, (yc - h / 2) * height)),
                  max(0, min(width, (xc + w / 2) * width)),
                  max(0, min(height, (yc + h / 2) * height))]
        pixels = norm1000_to_pixel(box['bbox_2d'], width, height)
        errors.extend(abs(a - b) for a, b in zip(pixels, direct))
    assert max(errors) <= 1, f'Sai số quy đổi vượt 1 pixel: {max(errors)}'
    print('Sai số lớn nhất (pixel):', max(errors))
    display(draw_boxes(entry, raw=answer, title='xanh = GT, do = JSON da quy doi'))

check_label_roundtrip(train_rows)'''),
    md(r'''## 3. E0: zero-shot'''),
    code(r'''import json
import os
import time
from pathlib import Path
from scripts.labcommon import read_jsonl

MODEL_ID = 'Qwen/Qwen3-VL-4B-Instruct'
PROJECTOR_KEYS = ('visual.merger', 'visual.deepstack_merger_list')

def user_message(image):
    return {"role": "user", "content": [{"type": "image", "image": image},
                                        {"type": "text", "text": PROMPT}]}


def open_image(entry):
    from PIL import Image
    return Image.open(Path("data") / entry["image_path"]).convert("RGB")


def load_model(adapter_dir=None, model_id=MODEL_ID):
    """Model 4-bit NF4 (ViT + lm_head giữ fp16). Có adapter_dir thì nạp LoRA và projector.pt nếu có."""
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True, llm_int8_skip_modules=["visual", "lm_head"])
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, quantization_config=quant, dtype=torch.float16, device_map={"": 0})
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"
    if adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir)
        proj = Path(adapter_dir) / "projector.pt"
        if proj.exists():
            load_projector(model, proj)
    model.eval()
    return model, processor


def load_projector(model, path):
    import torch
    state = torch.load(path, map_location="cpu")
    params = dict(model.named_parameters())
    missing = [k for k in state if k not in params]
    assert not missing, f"projector.pt có tham số không khớp model: {missing[:3]}"
    with torch.no_grad():
        for k, v in state.items():
            params[k].copy_(v.to(params[k].dtype))
    print(f"đã nạp lại {len(state)} tensor projector từ {path}")


def projector_state(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters()
            if p.requires_grad and any(k in n for k in PROJECTOR_KEYS)}


def predict(model, processor, entries, out_path, batch_size=4, max_new_tokens=256):
    """Suy luận greedy, ghi {"image_id","raw"} mỗi dòng. Bỏ qua ảnh đã có trong out_path (chạy tiếp được)."""
    import torch
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {r["image_id"] for r in read_jsonl(out_path)} if out_path.exists() else set()
    todo = [e for e in entries if e["image_id"] not in done]
    processor.tokenizer.padding_side = "left"
    model.eval()
    model.config.use_cache = True
    t0 = time.time()
    with open(out_path, "a", encoding="utf-8") as f:
        for i in range(0, len(todo), batch_size):
            chunk = todo[i:i + batch_size]
            inputs = processor.apply_chat_template(
                [[user_message(open_image(e))] for e in chunk], tokenize=True,
                add_generation_prompt=True, return_dict=True, return_tensors="pt", padding=True,
            ).to(model.device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            raws = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                          skip_special_tokens=True)
            for e, raw in zip(chunk, raws):
                f.write(json.dumps({"image_id": e["image_id"], "raw": raw}, ensure_ascii=False) + "\n")
            f.flush()
            if (i // batch_size) % 10 == 0:
                print(f"  {i + len(chunk)}/{len(todo)} ảnh, {time.time() - t0:.0f}s")
    print(f"xong {out_path} ({len(done) + len(todo)} dòng)")'''),
    code(r'''import gc
from lab10 import count_blocks

if os.path.exists('report/evidence/e0.json'):
    print('E0: đã có report/evidence/e0.json, bỏ qua.')
else:
    import torch
    model, processor = load_model()
    Path('report/extra').mkdir(parents=True, exist_ok=True)
    Path('report/extra/model_blocks.json').write_text(
        json.dumps(count_blocks(model), ensure_ascii=False, indent=2), encoding='utf-8')
    predict(model, processor, test_rows, 'work/e0.jsonl')
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()'''),
    code(r'''# Chấm riêng để có thể chạy tiếp nếu suy luận đã hoàn tất.
if not os.path.exists('report/evidence/e0.json'):
    !python scripts/score.py --pred work/e0.jsonl --name E0 --out report/evidence/e0.json
for row in read_jsonl('work/e0.jsonl')[:5]:
    print(row['raw'])
with open('report/evidence/e0.json', encoding='utf-8') as file:
    evidence = json.load(file)
print('parse_valid_rate:', evidence['parse_valid_rate'])
save_to_drive()'''),
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
