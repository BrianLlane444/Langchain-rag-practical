from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pdfplumber
import os

DOCUMENTS_PATH = "documents/sources/"
CHROMA_PATH = "embeddings/chromadb/"

# 1. Load all PDFs
def load_all_pdfs(folder_path):
    texts = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            path = os.path.join(folder_path, filename)
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    texts.append(page.extract_text())
    return "\n".join([t for t in texts if t])

# 2. Split into chunks
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_text(text)

# 3. Embed and store in ChromaDB
def embed_documents(chunks):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_texts(chunks, embedding=embedding_model, persist_directory=CHROMA_PATH)
    vectordb.persist()
    print(f"Stored {len(chunks)} chunks in ChromaDB")

if __name__ == "__main__":
    print("Loading PDFs...")
    full_text = load_all_pdfs(DOCUMENTS_PATH)

    print("Splitting text into chunks...")
    chunks = split_text(full_text)

    print("Embedding and storing in Chroma...")
    embed_documents(chunks)
