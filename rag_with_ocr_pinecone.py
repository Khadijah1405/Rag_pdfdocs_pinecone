# ✅ RAG with Pinecone v3 SDK and safe batched upload
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_pinecone import Pinecone as PineconeLangChain
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

# === Load API Keys ===
load_dotenv()
print("✅ DEBUG: Loaded index name:", os.getenv("PINECONE_INDEX_NAME"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
NAMESPACE = "rag-docs"

# === Load OCR documents ===
DOC_DIR = Path("rag pdf testing/text_cache")
docs = []
for file in DOC_DIR.glob("*.txt"):
    with open(file, "r", encoding="utf-8") as f:
        text = f.read()
        docs.append(Document(page_content=text, metadata={"source": file.name}))

if not docs:
    print("❌ No documents found in text_cache/. Run OCR script first.")
    exit(1)

# === Chunking ===
print(f"🧩 Chunking {len(docs)} documents...")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# === Embedding ===
print(f"📐 Embedding {len(chunks)} chunks...")
embedding = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# === Pinecone Setup ===
print("🌐 Connecting to Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)

if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    print(f"🆕 Creating index '{PINECONE_INDEX_NAME}'...")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION)
    )

index = pc.Index(PINECONE_INDEX_NAME)

# === Upload in Batches (Avoid Pinecone 4MB limit) ===
print(f"📦 Uploading chunks to namespace '{NAMESPACE}'...")
vectorstore = PineconeLangChain(
    index=index,
    embedding=embedding,
    text_key="text",
    namespace=NAMESPACE
)

batch_size = 50  # Safe batch size to avoid 4MB limit
for i in tqdm(range(0, len(chunks), batch_size), desc="🔁 Uploading in batches"):
    batch = chunks[i:i + batch_size]
    vectorstore.add_documents(batch)

# === Ask Questions ===
def ask_questions(vs):
    retriever = vs.as_retriever(search_kwargs={"k": 5})
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

    print("\n🤖 Ask anything about the uploaded documents. Type 'exit' to quit.")
    while True:
        q = input("\nQ: ")
        if q.strip().lower() == "exit":
            break
        print("\n🔍 Answer:\n", qa.run(q))

ask_questions(vectorstore)
