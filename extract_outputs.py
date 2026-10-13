"""Giải report/ (evidence, figures, extra) từ output của lab_day10.ipynb đã chạy trên Colab.

Chạy ở máy local sau khi lưu notebook:  python extract_outputs.py [lab_day10.ipynb]
"""
import base64
import io
import json
import sys
import zipfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
BEGIN, END = "=====LAB10-BUNDLE-BEGIN=====", "=====LAB10-BUNDLE-END====="


def find_bundle(nb):
    found = None
    for cell in nb["cells"]:
        for out in cell.get("outputs", []):
            text = out.get("text", "")
            text = "".join(text) if isinstance(text, list) else text
            if BEGIN in text and END in text:
                found = text.split(BEGIN, 1)[1].split(END, 1)[0]
    return found


def main():
    nb_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "lab_day10.ipynb"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    data = find_bundle(nb)
    if data is None:
        sys.exit("không thấy bundle: hãy chạy ô cuối (bundle_outputs) trên Colab rồi LƯU notebook (Ctrl+S)")
    with zipfile.ZipFile(io.BytesIO(base64.b64decode("".join(data.split())))) as z:
        for name in z.namelist():
            if not name.startswith("report/") or ".." in name:
                continue
            dest = ROOT / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(z.read(name))
            print("ghi", name)


if __name__ == "__main__":
    main()
