import os
from dotenv import load_dotenv
from langchain_pinecone import Pinecone as PineconeLangChain
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from pinecone import Pinecone

# === Load environment variables ===
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "ragpdftestupdated"
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
NAMESPACE = "rag-docs"

# === Reconnect to Pinecone index ===
print("🌐 Connecting to Pinecone index:", PINECONE_INDEX_NAME)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# === Load vector store (no document upload here) ===
embedding = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
vectorstore = PineconeLangChain(
    index=index,
    embedding=embedding,
    text_key="text",
    namespace=NAMESPACE
)

# === Ask Questions in CLI ===
def ask_questions(vs):
    retriever = vs.as_retriever(search_kwargs={"k": 5})
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

    print("\n🤖 Ask anything about the uploaded documents. Type 'exit' to quit.\n")
    while True:
        q = input("Q: ")
        if q.strip().lower() == "exit":
            break
        answer = qa.run(q)
        print("\n🔍 Answer:\n", answer, "\n")

ask_questions(vectorstore)
