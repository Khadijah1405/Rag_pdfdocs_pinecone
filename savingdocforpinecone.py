import os
from pathlib import Path
from pdf2image import convert_from_path
import pytesseract

# === Configuration ===
PDF_DIR = Path("rag pdf testing")
OUTPUT_DIR = PDF_DIR / "text_cache"
LANG = "deu"
TESSERACT_PATH = r"C:\Users\Khadijah-ali.shah\AppData\Local\Programs\Tesseract-OCR"
POPPLER_BIN = r"C:\poppler\bin"

# === Setup ===
os.environ["PATH"] += os.pathsep + TESSERACT_PATH
os.environ["PATH"] += os.pathsep + POPPLER_BIN
os.environ["TESSDATA_PREFIX"] = TESSERACT_PATH
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === Process PDFs ===
for pdf_path in PDF_DIR.glob("*.pdf"):
    output_file = OUTPUT_DIR / (pdf_path.stem + ".txt")
    if output_file.exists():
        print(f"✅ Already processed: {output_file.name}")
        continue

    print(f"\n📄 OCR: {pdf_path.name}")
    images = convert_from_path(str(pdf_path))
    text = ""

    for i, img in enumerate(images):
        page_text = pytesseract.image_to_string(img, lang=LANG)
        if page_text.strip():
            print(f"  ✅ Page {i+1}: {len(page_text.strip())} chars")
            text += f"\n\n--- Page {i + 1} ---\n\n{page_text}"
        else:
            print(f"  ⛔ Page {i+1}: no text found")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text.strip())

    print(f"💾 Saved to: {output_file.name}")
