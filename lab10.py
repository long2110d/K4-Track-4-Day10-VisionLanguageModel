"""Mã của mình cho Lab Day 10: nhãn JSON, nạp model 4-bit, mask loss, LoRA/đóng băng, train, suy luận.

Các import nặng (torch, transformers, peft) để trong hàm, nên phần nhãn/vẽ chạy được không cần GPU.
"""
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

from scripts.labcommon import CLASSES, read_jsonl

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
PROJECTOR_KEYS = ("visual.merger", "visual.deepstack_merger_list")

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

# 36 lớp LLM × 7 = 252 module; 24 khối ViT × 4 = 96 module (chỉ E3)
LLM_TARGETS = r".*language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
VIT_TARGETS = r".*visual\.blocks\.\d+\.(attn\.(qkv|proj)|mlp\.linear_fc[12])"


# ---------------------------------------------------------------- Bước 1–2: dữ liệu và nhãn
def load_split(split):
    return read_jsonl(DATA / "splits" / f"{split}.jsonl")


def split_stats():
    """In số ảnh/box từng split và kiểm tra không phiên quay nào nằm ở hai split."""
    sessions = {}
    for s in ("train", "val", "test"):
        rows = load_split(s)
        boxes = {c: sum(o["label"] == c for r in rows for o in r["objects"]) for c in CLASSES}
        print(f"{s:5s} {len(rows):5d} ảnh  {boxes}")
        sessions[s] = {r["session"] for r in rows}
    overlap = (sessions["train"] & sessions["val"]) | (sessions["train"] & sessions["test"]) \
        | (sessions["val"] & sessions["test"])
    print("phiên trùng giữa các split:", sorted(overlap) or "không có")
    return sessions


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


def open_image(entry):
    from PIL import Image
    return Image.open(DATA / entry["image_path"]).convert("RGB")


def draw_boxes(entry, raw=None, title=None):
    """Ảnh có box đúng (xanh lá) và box dự đoán từ output thô `raw` (đỏ)."""
    from PIL import ImageDraw

    from scripts.score import parse_output
    img = open_image(entry)
    d = ImageDraw.Draw(img)
    w, h = img.size
    for o in json.loads(objects_to_answer(entry["objects"])):
        box = norm1000_to_pixel(o["bbox_2d"], w, h)
        d.rectangle(box, outline=(0, 220, 0), width=2)
        d.text((box[0] + 2, box[1] + 1), o["label"], fill=(0, 220, 0))
    for p in (parse_output(raw) or []) if raw is not None else []:
        box = norm1000_to_pixel(p["bbox"], w, h)
        d.rectangle(box, outline=(255, 40, 40), width=2)
        d.text((box[0] + 2, box[3] - 11), p["label"], fill=(255, 40, 40))
    if title:
        d.rectangle([0, 0, w, 14], fill=(0, 0, 0))
        d.text((3, 1), title, fill=(255, 255, 255))
    return img


def show_grid(images, cols=3, path=None):
    from PIL import Image
    w, h = images[0].size
    rows = math.ceil(len(images) / cols)
    grid = Image.new("RGB", (cols * w, rows * h), (255, 255, 255))
    for i, im in enumerate(images):
        grid.paste(im, ((i % cols) * w, (i // cols) * h))
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        grid.save(path, quality=90)
    return grid


def user_message(image):
    return {"role": "user", "content": [{"type": "image", "image": image},
                                        {"type": "text", "text": PROMPT}]}


# ---------------------------------------------------------------- Bước 3: nạp model, suy luận
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
    print(f"xong {out_path} ({len(done) + len(todo)} dòng)")


# ---------------------------------------------------------------- Bước 4: ví dụ huấn luyện và mask loss
def encode_example(processor, image, answer):
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
    return n_img, ans


# ---------------------------------------------------------------- Bước 5: LoRA và đóng băng
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
    return model, len(matched), trainable


def count_blocks(model):
    """Số tham số từng khối: ViT (không tính projector), projector, LLM + embedding, lm_head."""
    out = {"vit": 0, "projector": 0, "llm": 0, "lm_head": 0}
    for n, p in model.named_parameters():
        k = p.numel() * (2 if p.__class__.__name__ == "Params4bit" else 1)  # 4-bit đóng gói 2 số/byte
        if any(x in n for x in PROJECTOR_KEYS):
            out["projector"] += k
        elif "visual" in n:
            out["vit"] += k
        elif "lm_head" in n:
            out["lm_head"] += k
        else:
            out["llm"] += k
    return out


# ---------------------------------------------------------------- Bước 6: vòng huấn luyện
def train(model, processor, out_dir, setup, lr=2e-4, batch_size=2, grad_accum=4, max_steps=300,
          seed=42, train_fraction=1.0, save_every=100, log_every=10):
    import torch
    from transformers import get_cosine_schedule_with_warmup

    random.seed(seed)
    torch.manual_seed(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = load_split("train")
    if train_fraction < 1.0:
        entries = random.Random(seed).sample(entries, max(1, int(len(entries) * train_fraction)))
    collate = make_collate(processor)
    processor.tokenizer.padding_side = "right"

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, int(0.05 * max_steps), max_steps)
    scaler = torch.amp.GradScaler("cuda")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()
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
        "train_images": len(entries),
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


# ---------------------------------------------------------------- Bước 7: chấm
def score(pred, name, out, run_json=None, split="test", coord="norm1000"):
    cmd = [sys.executable, str(ROOT / "scripts" / "score.py"), "--pred", str(pred), "--name", name,
           "--out", str(out), "--split", split, "--coord", coord]
    if run_json:
        run = json.loads(Path(run_json).read_text(encoding="utf-8"))
        for k in ("trainable_params", "peak_vram_gb", "train_minutes"):
            cmd += ["--meta", f"{k}={run[k]}"]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    print(r.stdout, r.stderr)
    r.check_returncode()


def per_image_f1(entries, preds):
    """F1 từng ảnh (dùng evaluate của trọng tài) để chọn ảnh minh hoạ đúng/sai."""
    from scripts.score import evaluate
    return {e["image_id"]: evaluate([e], {e["image_id"]: preds.get(e["image_id"], "")}, n_boot=1)
            for e in entries}
