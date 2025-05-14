# Rag_pdfdocs_pinecone
RAG_pdf_testdocs_pinecone is a lightweight document search and Q&amp;A system that combines OCR (Optical Character Recognition), LangChain, OpenAI embeddings, and Pinecone vector storage.  It is designed to help teams search and query unstructured PDF documents especially image-based files that lack selectable text by extracting  content via OCR.

This project implements a Retrieval-Augmented Generation (RAG) pipeline that processes PDF documents, extracts OCR text, chunks and embeds it using OpenAI embeddings, and stores it in Pinecone for semantic retrieval. Users can then ask questions and receive context-aware answers powered by GPT-4.

---

## 📁 Project Structure

- `savingdocforpinecone.py`:  
  Extracts text from image-based PDFs using OCR (Tesseract + Poppler) and saves it to `.txt` files.

- `rag_with_ocr_pinecone.py`:  
  Loads the pre-OCR `.txt` files, chunks and embeds them, and uploads the vectors to Pinecone (with batching and metadata).

- `askquesfrompinecone.py`:  
  Loads the Pinecone vector store and allows users to ask questions via a RetrievalQA chain using GPT-4.

---

## 🔧 Requirements

- Python 3.8+
- Tesseract OCR (Windows path setup required)
- Poppler for Windows (for PDF rendering)

---

# 🚀 Setup Instructions

### 1. Clone the repo and navigate to the folder:

```bash
git clone https://github.com/your-org/Rag_pdf_testdocs_pinecone.git
cd Rag_pdf_testdocs_pinecone
```

### 2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### 3. Install dependencies:

```bash
pip install -r requirements.txt
```

### 4. Create a .env file and add the following:

```env
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=ragpdftestupdated
PINECONE_REGION=us-east-1
PINECONE_CLOUD=aws
```

### 5. Run the scripts in order:

```bash
python savingdocforpinecone.py        # OCR extract PDFs to .txt
python rag_with_ocr_pinecone.py       # Upload text to Pinecone
python askquesfrompinecone.py         # Ask questions
```

