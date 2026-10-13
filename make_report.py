"""Điền REPORT.md từ report/evidence/*.json và report/extra/*.json (số liệu do notebook sinh ra).

Chạy: python make_report.py   (sau extract_outputs.py). Đọc lại và sửa lời văn cho đúng ý mình.
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
EV = ROOT / "report" / "evidence"
EX = ROOT / "report" / "extra"
NAME = "Đỗ Đình Long"
CLASSES = ("pothole", "crack", "manhole")
SETUPS = {"e0": "zero-shot", "e1": "LoRA LLM", "e2": "+ projector", "e3": "+ LoRA ViT"}


def load(p, default=None):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def f3(x):
    return f"{x:.3f}"


def ci(d):
    return f"[{f3(d['f1_ci95'][0])}, {f3(d['f1_ci95'][1])}]"


def f1(d):
    return d["micro"]["f1"]


def overlap(a, b):
    return a["f1_ci95"][0] <= b["f1_ci95"][1] and b["f1_ci95"][0] <= a["f1_ci95"][1]


def millions(n):
    return f"{n / 1e6:.1f}M"


def compare(a, b, na, nb):
    diff = f1(b) - f1(a)
    verdict = ("hai CI95 **chồng lên nhau**, nên chưa thể khẳng định chênh lệch này là thật"
               if overlap(a, b) else "hai CI95 **không chồng nhau**, nên chênh lệch này khá chắc chắn")
    return f"{nb} so với {na}: ΔF1 = {diff:+.3f} ({f3(f1(a))} → {f3(f1(b))}); {verdict}"


def row(key, d, label):
    m = d.get("meta", {})
    pc = d["per_class"]
    tp = m.get("trainable_params", 0)
    return (f"| {key.upper()} | {label} | {tp:,} | {m.get('peak_vram_gb', '–')} | {m.get('train_minutes', '–')} | "
            f"{f3(f1(d))} | {ci(d)} | {d['fp_per_image']:.2f} | "
            + " | ".join(f3(pc[c]["f1"]) for c in CLASSES) + " |")


def main():
    ev = {k: load(EV / f"{k}.json") for k in SETUPS}
    missing = [k for k, v in ev.items() if v is None]
    if missing:
        raise SystemExit(f"thiếu evidence: {missing}. Chạy notebook rồi extract_outputs.py trước.")
    e4 = {p.stem: load(p) for p in sorted(EV.glob("e4*.json"))}
    runs = {p.stem: load(p) for p in sorted((EX / "runs").glob("*.json"))}
    stats = load(EX / "data_stats.json", {})
    blocks = load(EX / "model_blocks.json", {})
    tok = load(EX / "token_stats.json", {})
    e0s = load(EX / "e0_samples.json", {})
    coord = load(EX / "coord_check.json", {})
    version = ev["e0"]["dataset_version"]
    e0, e1, e2, e3 = (ev[k] for k in ("e0", "e1", "e2", "e3"))
    best = max(("e1", "e2", "e3"), key=lambda k: f1(ev[k]))
    vram = {k: ev[k]["meta"]["peak_vram_gb"] for k in ("e1", "e2", "e3")}
    mins = {k: ev[k]["meta"]["train_minutes"] for k in ("e1", "e2", "e3")}
    tparams = {k: ev[k]["meta"]["trainable_params"] for k in ("e1", "e2", "e3")}
    bs = {k: (runs.get(k, {}).get("batch_size"), runs.get(k, {}).get("grad_accum")) for k in ("e1", "e2", "e3")}

    L = [f"Họ tên: {NAME}", f"`dataset_version`: {version}", "",
         "## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)", "",
         "| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, lab in SETUPS.items():
        L.append(row(k, ev[k], lab))
    for k, d in e4.items():
        L.append(row("E4", d, f"{best.upper()}, {k.replace('e4_', '').replace('_', ' = ')}"))
    L += ["", "Ảnh minh hoạ: `report/figures/` (box đúng màu xanh lá, dự đoán màu đỏ). "
          "Đường loss: `report/figures/loss.png`.", "",
          "## 10 câu hỏi (trả lời bằng **số của bạn**, không chép lại tài liệu)", ""]

    # Câu 1
    proj_e2 = tparams["e2"] - tparams["e1"]
    vit_lora = tparams["e3"] - tparams["e2"]
    L += ["### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?",
          (f"Đếm trên model đã nạp: ViT ≈ {millions(blocks.get('vit', 0))}, projector (`visual.merger` + "
           f"`visual.deepstack_merger_list`, DeepStack lấy đặc trưng ở lớp 5/11/17 của ViT) ≈ {millions(blocks.get('projector', 0))}, "
           f"LLM + embedding ≈ {millions(blocks.get('llm', 0))}, "
           + (f"`lm_head` ≈ {millions(blocks['lm_head'])}. " if blocks.get("lm_head") else "`lm_head` dùng chung trọng số với embedding (tied). ")
           + 
           "Ảnh đi qua ViT → projector (gộp 2×2 patch và chiếu sang chiều ẩn của LLM) → thành token ảnh trong chuỗi của LLM."),
          "",
          f"- **E1:** chỉ LoRA r=16 trên 252 module của LLM (q/k/v/o + gate/up/down × 36 lớp): {tparams['e1']:,} tham số; ViT và projector đóng băng.",
          f"- **E2:** E1 + mở đầy đủ projector: {tparams['e2']:,} tham số (nhiều hơn E1 {millions(proj_e2)}, đúng bằng kích thước projector).",
          f"- **E3:** E2 + LoRA trên 96 module của ViT (qkv/proj/fc1/fc2 × 24 khối, tổng 348 module): {tparams['e3']:,} tham số (+{millions(vit_lora)} so với E2).",
          ""]

    # Câu 2
    first = (e0s.get("invalid") or e0s.get("first") or [{"raw": ""}])[0]["raw"]
    raws = (e0s.get("first") or [])
    fence = sum("```" in r["raw"] for r in raws)
    other_fp = e0["per_class"]["other"]["fp"]
    L += ["### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?",
          (f"`parse_valid_rate` của E0 = **{f3(e0['parse_valid_rate'])}** ({e0s.get('n_invalid', '?')}/{e0['n_images']} output không parse được). "
           f"F1 = {f3(f1(e0))} (P = {f3(e0['micro']['precision'])}, R = {f3(e0['micro']['recall'])}), "
           f"FP/ảnh = {e0['fp_per_image']:.2f}, mean IoU của các cặp khớp = {f3(e0['mean_iou'])}."),
          "",
          (f"1. **Box lệch / bịa box:** precision chỉ {f3(e0['micro']['precision'])} và FP/ảnh = {e0['fp_per_image']:.2f}: "
           "phần lớn box của model gốc không khớp GT ở IoU ≥ 0,5 (box quá to bao cả mảng đường, lệch vị trí, hoặc bịa ra chỗ không có hư hại); "
           "box GT lại rất nhỏ nên chỉ lệch vài chục pixel là trượt."),
          (f"2. **Sai định dạng / sai lớp:** {fence}/{len(raws)} output đầu tiên bọc trong ```json; "
           f"{other_fp} box có nhãn ngoài 3 lớp (bị tính là FP lớp `other`). "
           f"Model gốc chưa biết quy ước \"crack\" của bộ dữ liệu (recall crack = {f3(e0['per_class']['crack']['recall'])})."),
          "", "Ví dụ output thô:", "", "````text", first[:400], "````",
          "Ảnh minh hoạ: `report/figures/e0_examples.jpg`.", ""]

    # Câu 3
    if coord:
        c3 = (f"Kiểm chứng trên dự đoán của {coord['name'].upper()}: chấm đúng thang 0–1000 F1 = **{f3(coord['correct'])}**; "
              f"cùng output đó nhưng chấm như pixel (`--coord pixel`) F1 = **{f3(coord['norm1000_scored_as_pixel'])}**; "
              f"đổi box sang pixel rồi chấm như 0–1000 F1 = **{f3(coord['pixel_scored_as_norm1000'])}**; "
              f"đổi sang pixel và chấm đúng `--coord pixel` thì F1 lại = {f3(coord['pixel_scored_as_pixel'])} "
              "(chỉ khác do làm tròn).")
    else:
        c3 = "(chưa có report/extra/coord_check.json)"
    L += ["### Câu 3 — Thang 0–1000: nếu chấm output theo pixel (hoặc ngược lại) thì F1 ra sao và vì sao? Bạn đã kiểm chứng điều này thế nào?",
          c3, "",
          ("Lý do: ảnh 640×360 không vuông. Diễn giải số 0–1000 như pixel thì trục x bị co ×0,64 và trục y bị giãn ×2,78 (1000/360) "
           "rồi bị kẹp ngoài khung, nên gần như không box nào còn IoU ≥ 0,5 với GT. JSON vẫn hợp lệ (parse_valid_rate không đổi) "
           "nên lỗi này \"im lặng\". Qwen3-VL được pretrain grounding trên thang tương đối 0–1000, độc lập với độ phân giải "
           "sau khi processor resize ảnh, nên dạy đúng thang này thì model học nhanh nhất. "
           "Mình còn kiểm tra ở bước 2: đổi nhãn JSON ngược về pixel và vẽ lại, box trùng box YOLO gốc (sai số ≤ 0,64 px do làm tròn); "
           "có test tự động trong `test_lab10.py`."), ""]

    # Câu 4
    L += ["### Câu 4 — Mask loss: vì sao chỉ tính loss trên phần trả lời? Điều gì xảy ra nếu tính trên cả prompt và token ảnh?",
          (f"Đo trên {tok.get('n', 40)} ảnh train: mỗi ví dụ có trung bình {tok.get('total', '?')} token, trong đó "
           f"{tok.get('image', '?')} token ảnh (đúng 220 cho ảnh 640×360) và chỉ {tok.get('answer', '?')} token đáp án "
           "(JSON + `<|im_end|>`) được tính loss; còn lại đặt `labels = -100`."),
          "",
          ("Mục tiêu là học p(đáp án | ảnh, prompt). Prompt giống hệt nhau ở mọi ví dụ, nên nếu tính loss trên đó thì model "
           "phải học thuộc lòng câu lệnh, gradient bị chi phối bởi phần dễ (loss nhanh chóng về ~0 trên prompt) và tín hiệu "
           "cho phần JSON bị pha loãng theo tỉ lệ token. Token ảnh `<|image_pad|>` là placeholder được thay bằng embedding của "
           "ViT, không phải token mà LLM sinh ra; ép LLM \"dự đoán\" chúng là mục tiêu vô nghĩa và làm méo phân phối đầu ra. "
           "Hậu quả thường thấy: loss trông thấp nhưng F1 không tăng, model có thể sinh lại prompt hoặc lặp vô hạn. "
           "Mình kiểm tra bằng cách giải mã riêng các vị trí `labels != -100`: kết quả đúng bằng JSON đáp án + `<|im_end|>`."), ""]

    # Câu 5
    L += ["### Câu 5 — Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit, và bạn đã xử lý thế nào? Vì sao projector phải ở float32?",
          ("Khi lượng tử hoá NF4, trọng số Linear được lưu thành `Params4bit` (uint8 đóng gói + hằng số scale theo block). "
           "Đây là tensor nguyên, không có gradient và không cập nhật được bằng AdamW; bitsandbytes chỉ cho phép train adapter "
           "LoRA *bên cạnh* lớp 4-bit. Vì vậy mình đặt `llm_int8_skip_modules=[\"visual\", \"lm_head\"]` để cả ViT (gồm merger và "
           "deepstack merger) giữ fp16 thật, rồi ở E2/E3 chuyển các tham số `visual.merger*`/`deepstack_merger_list*` sang float32 "
           "và bật `requires_grad`."),
          "",
          (f"Float32 vì train fp16 + GradScaler: cập nhật lr·grad (lr = 2e-4) trên trọng số fp16 thường nhỏ hơn bước lượng tử "
           "của fp16 (≈ 1e-3 tương đối) nên bị làm tròn mất, và trạng thái Adam (m, v) ở fp16 dễ tràn/underflow. "
           f"`GradScaler.unscale_` cũng từ chối gradient fp16. Giữ master weight fp32 cho {millions(proj_e2)} tham số projector "
           f"tốn thêm bộ nhớ: VRAM đỉnh E2 = {vram['e2']} GB so với E1 = {vram['e1']} GB. Projector train đầy đủ nên không nằm trong "
           "adapter của peft; mình lưu riêng `projector.pt` và nạp lại lúc suy luận."), ""]

    # Câu 6
    cnt = stats.get("counts", {})
    lines6 = [f"| {s} | {cnt[s]['images']} | " + " | ".join(str(cnt[s]['boxes'][c]) for c in CLASSES)
              + f" | {stats.get('n_sessions', {}).get(s, '?')} |" for s in ("train", "val", "test") if s in cnt]
    L += ["### Câu 6 — Split theo phiên quay: vì sao chia ngẫu nhiên sẽ cho F1 cao giả? Số ảnh và số box từng lớp ở mỗi split là bao nhiêu?",
          ("Ảnh là frame cắt từ video dashcam, hai frame liền nhau gần như trùng (cùng ổ gà, cùng góc nhìn, cùng ánh sáng). "
           "Chia ngẫu nhiên thì hầu hết ảnh test có \"anh em sinh đôi\" trong train: model chỉ cần nhớ thay vì tổng quát hoá, "
           "nên F1 trên test bị thổi phồng và không phản ánh khả năng trên đoạn đường mới. `prepare_data.py` gom ảnh theo phiên quay "
           "(ngày + giờ trong tên file) và gán nguyên phiên vào một split; mình đã kiểm tra không phiên nào xuất hiện ở hai split."),
          "", "| Split | Ảnh | pothole | crack | manhole | Số phiên |", "|---|---|---|---|---|---|", *lines6, "",
          ("Hệ quả: phân bố lớp lệch giữa các split (val rất ít pothole), nên phải đọc F1 từng lớp và CI95."), ""]

    # Câu 7
    L += ["### Câu 7 — So sánh E1/E2/E3: mở projector và LoRA trên ViT có đáng không (F1, CI95, VRAM, thời gian)?",
          f"- {compare(e1, e2, 'E1', 'E2')}. VRAM {vram['e1']} → {vram['e2']} GB, thời gian {mins['e1']} → {mins['e2']} phút.",
          f"- {compare(e2, e3, 'E2', 'E3')}. VRAM {vram['e2']} → {vram['e3']} GB, thời gian {mins['e2']} → {mins['e3']} phút.",
          f"- {compare(e1, e3, 'E1', 'E3')}.",
          f"- Batch thực tế (batch × accum): E1 {bs['e1'][0]}×{bs['e1'][1]}, E2 {bs['e2'][0]}×{bs['e2'][1]}, E3 {bs['e3'][0]}×{bs['e3'][1]}.",
          "",
          (f"Setup F1 cao nhất là **{best.upper()}** ({f3(f1(ev[best]))}). "
           + ("Vì các CI95 chồng nhau, xét theo chi phí thì E1 là lựa chọn \"đáng tiền\" nhất: ít tham số, ít VRAM và nhanh nhất; "
              "mở thêm projector/ViT chưa chứng minh được lợi ích trên tập test này."
              if overlap(e1, ev[best]) or best == "e1" else
              f"Chênh lệch giữa {best.upper()} và E1 vượt CI95, nên chi phí thêm (VRAM, thời gian) là đáng: "
              "box nhỏ (vết nứt mảnh) cần đặc trưng thị giác được tinh chỉnh chứ không chỉ cách LLM đọc đặc trưng.")), ""]

    # Câu 8
    widths = {k: ev[k]["f1_ci95"][1] - ev[k]["f1_ci95"][0] for k in SETUPS}
    L += ["### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?",
          ("CI95 bootstrap theo ảnh (1000 lần lấy mẫu lại 311 ảnh) có độ rộng: "
           + ", ".join(f"{k.upper()} {widths[k]:.3f}" for k in SETUPS) + "."),
          "",
          (f"Nghĩa là với F1 quanh {f3(f1(ev[best]))}, sai số do chọn mẫu khoảng ±{max(widths.values()) / 2:.3f}. "
           f"E1 so với E0 chênh {f1(e1) - f1(e0):+.3f}, {'vượt xa' if not overlap(e0, e1) else 'chưa vượt'} CI nên kết luận \"finetune có ích\" là chắc chắn. "
           + ("Các chênh lệch giữa E1/E2/E3 đều nằm trong CI (các khoảng chồng nhau), nên chỉ nói được \"không thấy khác biệt có ý nghĩa\", "
              "không xếp hạng được. " if all(overlap(ev[a], ev[b]) for a, b in (("e1", "e2"), ("e2", "e3"), ("e1", "e3")))
              else "Có cặp setup trong E1/E2/E3 có CI không chồng nhau (xem Câu 7), đó là kết luận đứng vững; các cặp còn lại thì chưa. ")
           + "Thêm nữa, ảnh trong cùng phiên tương quan với nhau, nên CI bootstrap theo ảnh còn *hẹp hơn thực tế*; "
           f"test chỉ đến từ {stats.get('n_sessions', {}).get('test', 'vài')} phiên quay, F1 từng lớp "
           f"(ví dụ pothole chỉ {cnt.get('test', {}).get('boxes', {}).get('pothole', '?')} box trong test) còn dao động mạnh hơn. "
           "Muốn so sánh chắc hơn cần nhiều seed hoặc tập test lớn hơn / nhiều phiên hơn."), ""]

    # Câu 9
    pc = ev[best]["per_class"]
    worst = min(CLASSES, key=lambda c: pc[c]["f1"])
    area = stats.get("median_area_pct", {})
    train_cnt = cnt.get("train", {}).get("boxes", {})
    L += ["### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.",
          f"F1 từng lớp của {best.upper()}:", "",
          "| Lớp | P | R | F1 | Box train | Diện tích trung vị (% khung) |", "|---|---|---|---|---|---|",
          *[f"| {c} | {f3(pc[c]['precision'])} | {f3(pc[c]['recall'])} | {f3(pc[c]['f1'])} | {train_cnt.get(c, '?')} | {area.get(c, '?')} |"
            for c in CLASSES], "",
          (f"Lớp kém nhất là **{worst}**. "
           + {"crack": ("Vết nứt mảnh, dài, ranh giới mơ hồ: một vết nứt có thể được gán nhãn thành một box dài hoặc nhiều box ngắn, "
                        "nên dù model nhìn thấy đúng vết nứt thì IoU vẫn dễ < 0,5. Box lại nhỏ (diện tích trung vị "
                        f"{area.get('crack', '?')}% khung), sau khi gộp 2×2 patch mỗi token ảnh phủ 32×32 px nên chi tiết mảnh bị mất."),
              "pothole": ("Ổ gà ít mẫu hơn crack, hình dạng/màu đa dạng (có nước, có bóng), dễ nhầm với vá đường hoặc bóng cây; "
                          "split theo phiên khiến test có điều kiện ánh sáng khác train."),
              "manhole": ("Nắp cống ít mẫu nhất trong train và hay bị nhầm với vá đường tròn/vết bẩn; ở xa thì box rất nhỏ.")}[worst]),
          "", f"Ảnh: `report/figures/{best}_good_bad.jpg` (3 ảnh tốt nhất, 3 ảnh tệ nhất), `report/figures/errors_{{pothole,crack,manhole}}.jpg`.", ""]

    # Câu 10
    if e4:
        parts = [f"{k.replace('e4_', '')}: F1 = {f3(f1(d))} {ci(d)}, {d['meta'].get('trainable_params', 0):,} tham số, "
                 f"{d['meta'].get('peak_vram_gb')} GB, {d['meta'].get('train_minutes')} phút" for k, d in e4.items()]
        ds = list(e4.values())
        same = overlap(ds[0], ds[-1]) if len(ds) > 1 else True
        c10 = ("Biến đổi: **rank LoRA** (alpha = 2r) trên setup tốt nhất " + best.upper()
               + ", giữ nguyên mọi thứ khác (lr, 300 bước, batch, seed, dữ liệu). " + "; ".join(parts)
               + f". Tham chiếu r=16: F1 = {f3(f1(ev[best]))} {ci(ev[best])}. "
               + ("Các CI95 chồng nhau, nên **không** kết luận được rank nào tốt hơn: với 300 bước và ~1.400 ảnh, "
                  "ngay cả r=4 đã đủ dung lượng; tăng rank chủ yếu tăng tham số mà không tăng F1 rõ ràng."
                  if same else
                  "CI95 của hai giá trị không chồng nhau, nên khác biệt về rank là đáng tin trên tập test này "
                  "(dù vẫn chỉ là một seed)."))
    else:
        c10 = "(chưa có e4*.json)"
    L += ["### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?", c10, ""]

    L += ["## Nhìn lại", "",
          ("- Công cụ chấm dùng IoU ≥ 0,5 theo từng box, nên với vết nứt dài, một dự đoán bao trọn cả vết nứt nhưng GT chia làm "
           "nhiều box nhỏ vẫn bị tính sai hoàn toàn. Thước đo này phạt nặng sự mơ hồ của nhãn hơn là lỗi của model."),
          ("- CI bootstrap theo ảnh bỏ qua tương quan giữa các frame trong cùng phiên; bootstrap theo phiên sẽ cho CI trung thực hơn."),
          ("- Mỗi setup mới chạy một seed; chênh lệch nhỏ giữa E1/E2/E3 có thể chỉ là nhiễu do seed và thứ tự dữ liệu."), ""]
    (ROOT / "REPORT.md").write_text("\n".join(L), encoding="utf-8")
    print("đã ghi REPORT.md")


if __name__ == "__main__":
    main()
