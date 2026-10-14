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
    bundle_outputs, LLM_TARGETS, VIT_TARGETS, PROJECTOR_KEYS,
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
Chỉ tính loss trên JSON đáp án và `<|im_end|>`; bỏ qua prompt và token padding bằng nhãn -100.'''),
    code(r'''def encode_example(processor, image, answer):
    import torch
    user = user_message(image)
    assistant = {"role": "assistant", "content": [{"type": "text", "text": answer}]}
    kw = dict(tokenize=True, return_dict=True, return_tensors="pt")
    prompt_enc = processor.apply_chat_template([user], add_generation_prompt=True, **kw)
    full_enc = processor.apply_chat_template([user, assistant], add_generation_prompt=False, **kw)
    n = prompt_enc["input_ids"].shape[1]
    assert torch.equal(full_enc["input_ids"][0, :n], prompt_enc["input_ids"][0]), \
        "prompt không phải tiền tố của chuỗi đầy đủ"
    # cắt phần sau <|im_end|> cuối (dấu xuống dòng) để đáp án kết thúc đúng bằng <|im_end|>
    end_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    ids = full_enc["input_ids"][0].tolist()
    last = len(ids) - 1 - ids[::-1].index(end_id)
    enc = {k: (v[:, :last + 1] if k in ("input_ids", "attention_mask", "mm_token_type_ids",
                                        "token_type_ids") else v)
           for k, v in full_enc.items()}
    labels = enc["input_ids"].clone()
    labels[:, :n] = -100
    enc["labels"] = labels
    return enc


def make_collate(processor):
    import torch
    pad_id = processor.tokenizer.pad_token_id

    def collate(entries):
        encs = [encode_example(processor, open_image(e), objects_to_answer(e["objects"]))
                for e in entries]
        L = max(x["input_ids"].shape[1] for x in encs)
        batch = {}
        for k in encs[0]:
            if k in ("pixel_values", "image_grid_thw"):
                batch[k] = torch.cat([x[k] for x in encs], dim=0)   # không pad
                continue
            fill = {"input_ids": pad_id, "labels": -100}.get(k, 0)
            batch[k] = torch.cat([torch.nn.functional.pad(x[k], (0, L - x[k].shape[1]), value=fill)
                                  for x in encs], dim=0)
        return batch
    return collate


def check_example(processor, entry):
    """In số token ảnh và phần đáp án (labels != -100) đã giải mã."""
    enc = encode_example(processor, open_image(entry), objects_to_answer(entry["objects"]))
    img_tok = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    n_img = int((enc["input_ids"] == img_tok).sum())
    ans = processor.tokenizer.decode(enc["input_ids"][enc["labels"] != -100])
    print("token ảnh:", n_img, "| tổng token:", enc["input_ids"].shape[1])
    print("phần được tính loss:", ans)
    return n_img, ans'''),
    code(r'''from lab10 import token_stats

# Chỉ nạp processor nếu E0 đã được bỏ qua hoặc đã giải phóng bộ nhớ.
if 'processor' not in globals():
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(MODEL_ID)

n_img, ans = check_example(processor, train_rows[0])
assert n_img == 220, n_img
assert ans.endswith('<|im_end|>'), ans
assert ans[:-len('<|im_end|>')] == objects_to_answer(train_rows[0]['objects']), ans
tok = token_stats(processor)
Path('report/extra').mkdir(parents=True, exist_ok=True)
Path('report/extra/token_stats.json').write_text(
    json.dumps(tok, ensure_ascii=False, indent=2), encoding='utf-8')
print(tok)'''),
    md(r'''## 5. Setup LoRA / đóng băng cho E1, E2, E3
Theo GUIDE: E1/E2 có 252 module LLM; E3 thêm 96 module ViT, tổng 348.
E2 − E1 ≈ 109M tham số do huấn luyện thêm projector ở fp32.'''),
    code(r'''PROJECTOR_KEYS = ("visual.merger", "visual.deepstack_merger_list")
LLM_TARGETS = r".*language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
VIT_TARGETS = r".*visual\.blocks\.\d+\.(attn\.(qkv|proj)|mlp\.linear_fc[12])"


def apply_setup(model, setup, r=16, alpha=32, dropout=0.05):
    """setup: "E1" (LoRA LLM), "E2" (+ projector), "E3" (+ LoRA ViT)."""
    import re

    import torch
    from peft import LoraConfig, get_peft_model

    for p in model.parameters():
        p.requires_grad = False
    model.enable_input_require_grads()
    target = LLM_TARGETS if setup == "E1" or setup == "E2" else f"({LLM_TARGETS})|({VIT_TARGETS})"
    matched = [n for n, _ in model.named_modules() if re.fullmatch(target, n)]
    print(f"{setup}: {len(matched)} module LoRA (mong đợi {252 if setup != 'E3' else 348})")
    model = get_peft_model(model, LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout,
                                             target_modules=target, task_type="CAUSAL_LM"))
    if setup in ("E2", "E3"):
        for n, p in model.named_parameters():
            if any(k in n for k in PROJECTOR_KEYS):
                p.data = p.data.float()
                p.requires_grad = True
    for p in model.parameters():          # GradScaler cần gradient fp32
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    proj = sum(p.numel() for n, p in model.named_parameters()
               if p.requires_grad and any(k in n for k in PROJECTOR_KEYS))
    print(f"tham số huấn luyện: {trainable:,} (trong đó projector {proj:,})")
    return model, len(matched), trainable'''),
    md(r'''## 6. Vòng huấn luyện'''),
    code(r'''def train(model, processor, out_dir, setup, lr=2e-4, batch_size=2, grad_accum=4, max_steps=300,
          seed=42, train_fraction=1.0, save_every=100, log_every=10):
    import torch
    import random
    from transformers import get_cosine_schedule_with_warmup

    random.seed(seed)
    torch.manual_seed(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = train_rows[:]
    if train_fraction < 1.0:
        entries = random.Random(seed).sample(entries, max(1, int(len(entries) * train_fraction)))
    collate = make_collate(processor)
    processor.tokenizer.padding_side = "right"

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()
    # GradScaler chỉ unscale được gradient fp32: ép mọi tham số đang học về fp32 SAU khi bật
    # gradient checkpointing và TRƯỚC khi tạo optimizer (E2/E3: projector nạp ở fp16)
    fp16 = [n for n, p in model.named_parameters() if p.requires_grad and p.dtype != torch.float32]
    if fp16:
        print(f"ép {len(fp16)} tensor đang học về fp32, ví dụ: {fp16[:3]}")
    for p in model.parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, int(0.05 * max_steps), max_steps)
    scaler = torch.amp.GradScaler("cuda")
    torch.cuda.reset_peak_memory_stats()

    rng = random.Random(seed)
    order, pos = [], 0
    log = []
    t0 = time.time()
    for step in range(1, max_steps + 1):
        total = 0.0
        for _ in range(grad_accum):
            if pos + batch_size > len(order):
                order, pos = entries[:], 0
                rng.shuffle(order)
            batch = collate(order[pos:pos + batch_size])
            pos += batch_size
            batch = {k: v.to(model.device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**batch).loss / grad_accum
            if not torch.isfinite(loss):
                raise RuntimeError(f"loss không hữu hạn ở bước {step}: {loss.item()} (tràn fp16?)")
            scaler.scale(loss).backward()
            total += loss.item()
        try:
            scaler.unscale_(opt)
        except ValueError:                 # còn gradient fp16: đưa cả tham số lẫn gradient về fp32
            for p in params:
                if p.grad is not None and p.grad.dtype != torch.float32:
                    g = p.grad.float()
                    p.grad = None
                    p.data = p.data.float()
                    p.grad = g
            scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()
        log.append({"step": step, "loss": total, "lr": sched.get_last_lr()[0]})
        if step % log_every == 0:
            print(f"bước {step}/{max_steps} loss={total:.4f} lr={sched.get_last_lr()[0]:.2e} "
                  f"{(time.time() - t0) / 60:.1f} phút")
        if step % save_every == 0 or step == max_steps:
            save_checkpoint(model, out_dir, setup)
    run = {
        "setup": setup,
        "trainable_params": sum(p.numel() for p in params),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2),
        "train_minutes": round((time.time() - t0) / 60, 1),
        "max_steps": max_steps, "lr": lr, "batch_size": batch_size, "grad_accum": grad_accum,
        "train_images": len(entries), "seed": seed, "train_fraction": train_fraction,
        "save_every": save_every, "log_every": log_every,
    }
    (out_dir / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    (out_dir / "loss.json").write_text(json.dumps(log), encoding="utf-8")
    model.gradient_checkpointing_disable()
    model.eval()
    print(run)
    return run


def save_checkpoint(model, out_dir, setup):
    import torch
    model.save_pretrained(out_dir)                     # peft: CHỈ lưu adapter LoRA
    if setup in ("E2", "E3"):                         # projector train đầy đủ phải lưu riêng
        torch.save(projector_state(model), Path(out_dir) / "projector.pt")
    print(f"  đã lưu checkpoint vào {out_dir}")


def train_setup(name, setup, r=None, alpha=None):
    evidence = Path(f'report/evidence/{name}.json')
    out_dir = Path(f'work/adapter_{name}')
    if evidence.exists() or (out_dir / 'run.json').exists():
        print(f'{name}: đã có bằng chứng hoặc huấn luyện xong, bỏ qua.')
        if (out_dir / 'run.json').exists():
            for source, folder in (('run.json', 'runs'), ('loss.json', 'loss')):
                target = Path('report/extra') / folder / f'{name}.json'
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(out_dir / source, target)
        save_to_drive()
        return
    import torch
    import yaml
    from lab10 import free

    cfg = yaml.safe_load(Path(f'configs/{setup.lower()}.yaml').read_text(encoding='utf-8'))
    r = cfg['lora']['r'] if r is None else r
    alpha = cfg['lora']['alpha'] if alpha is None else alpha
    batch_size, grad_accum = cfg['batch_size'], cfg['grad_accum']
    model = processor = None
    try:
        for attempt in range(2):
            oom = False
            try:
                model, processor = load_model()
                model, lora_modules, _ = apply_setup(
                    model, setup, r=r, alpha=alpha, dropout=cfg['lora']['dropout'])
                run = train(model, processor, out_dir, setup, lr=cfg['lr'],
                            batch_size=batch_size, grad_accum=grad_accum,
                            max_steps=cfg['max_steps'], seed=cfg['seed'])
            except torch.cuda.OutOfMemoryError:
                if attempt == 1:
                    raise
                oom = True
            # Ra khỏi except để giải phóng traceback giữ tensor của lần lỗi.
            if oom:
                model = processor = None
                free()
                batch_size, grad_accum = 1, cfg['batch_size'] * cfg['grad_accum']
                print(f'{name}: hết VRAM, thử lại với batch 1, tích luỹ {grad_accum}.')
                continue
            run.update({'lora_modules': lora_modules, 'r': r, 'alpha': alpha})
            (out_dir / 'run.json').write_text(json.dumps(run, indent=2), encoding='utf-8')
            for source, folder in (('run.json', 'runs'), ('loss.json', 'loss')):
                target = Path('report/extra') / folder / f'{name}.json'
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(out_dir / source, target)
            break
    finally:
        model = processor = None
        free()
    save_to_drive()'''),
    code(r'''train_setup('e1', 'E1')'''),
    code(r'''train_setup('e2', 'E2')'''),
    code(r'''train_setup('e3', 'E3')'''),
    md(r'''## 7. Suy luận và chấm E1–E3'''),
    code(r'''def evaluate_setup(name, coord='norm1000'):
    evidence = Path(f'report/evidence/{name}.json')
    if evidence.exists():
        print(f'{name}: đã có {evidence}, bỏ qua suy luận và chấm.')
        return
    from lab10 import free

    run = json.loads(Path(f'work/adapter_{name}/run.json').read_text(encoding='utf-8'))
    model = processor = None
    try:
        model, processor = load_model(f'work/adapter_{name}')
        predict(model, processor, test_rows, f'work/{name}.jsonl')
    finally:
        model = processor = None
        free()
    subprocess.run([
        sys.executable, 'scripts/score.py', '--pred', f'work/{name}.jsonl',
        '--name', name.upper(), '--out', f'report/evidence/{name}.json', '--coord', coord,
        '--meta', f"trainable_params={run['trainable_params']}",
        '--meta', f"peak_vram_gb={run['peak_vram_gb']}",
        '--meta', f"train_minutes={run['train_minutes']}",
    ], check=True)
    save_to_drive()'''),
    code(r'''for k in ('e1', 'e2', 'e3'):
    evaluate_setup(k)

print('Thí nghiệm | micro F1 | CI95 | parse_valid_rate | meta')
results = {}
for k in ('e0', 'e1', 'e2', 'e3'):
    d = json.loads(Path(f'report/evidence/{k}.json').read_text(encoding='utf-8'))
    results[k] = d
    print(k.upper(), 'F1=%.3f' % d['micro']['f1'], 'CI95=[%.3f, %.3f]' % tuple(d['f1_ci95']),
          'valid=%.3f' % d['parse_valid_rate'], 'meta=', d['meta'])
BEST = max(('e1', 'e2', 'e3'), key=lambda k: results[k]['micro']['f1']).upper()
coord_check(BEST.lower())     # Câu 3: chấm sai thang toạ độ
save_to_drive()'''),
    md(r'''## 8. E4 (ablation tự chọn) và E5 (tuỳ chọn, GRPO)
E4: đổi **rank LoRA** trên setup tốt nhất của E1–E3, r = 4 và r = 64 (alpha = 2r), theo lựa chọn `rank` trong `configs/e4.yaml`; giữ nguyên mọi thiết lập khác. E5: không làm.'''),
    code(r'''for r in (4, 64):
    name = f'e4_rank_{r}'
    train_setup(name, BEST, r=r, alpha=2 * r)
    evaluate_setup(name)'''),
    code(r'''make_figures(BEST.lower())
plot_losses()
!python scripts/check_submission.py
save_to_drive()
bundle_outputs()   # gói cuối cùng: evidence + figures + extra
# Chạy python extract_outputs.py trên máy local để giải nén gói kết quả.'''),
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
