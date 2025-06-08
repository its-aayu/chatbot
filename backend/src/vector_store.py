import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from PyPDF2 import PdfReader
from dotenv import load_dotenv

load_dotenv()

def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text

def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    return splitter.create_documents([text])

def load_or_create_vector_store():
    persist_dir = "embeddings"

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if os.path.isdir(persist_dir) and os.listdir(persist_dir):
        print("📂 Loading existing vector store...")
        vectordb = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    else:
        print("📄 Creating new vector store from PDF...")
        pdf_path = os.path.join("data", "indian_constitution.pdf")
        text = load_pdf(pdf_path)
        docs = split_text(text)

        # print(f"✅ Created {len(docs)} text chunks.")
        # for i, doc in enumerate(docs[:5]):
        #     print(f"\n🔹 Chunk {i+1}:\n{doc.page_content[:300]}")

        vectordb = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=persist_dir
        )
        vectordb.persist()

    return vectordb

if __name__ == "__main__":
    vector_store = load_or_create_vector_store()
    print("✅ Vector store ready")
    print(f"📊 Number of documents in vector store: {len(vector_store.get())}")
