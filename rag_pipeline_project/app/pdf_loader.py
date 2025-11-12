from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from pathlib import Path
from typing import List

def load_pdfs_from_folder(folder_path: str) -> List[Document]:
    """
    Load all PDFs in the folder and return a list of their full texts.
    """
    folder = Path(folder_path).resolve()
    print(f"[DEBUG] Looking for PDFs in: {folder}")

    all_docs = []
    pdf_files = list(folder.glob("*.pdf"))
    print(f"[DEBUG] Found {len(pdf_files)} PDFs: {[f.name for f in pdf_files]}")

    for pdf_file in pdf_files:
        loader = PyPDFLoader(str(pdf_file))
        documents = loader.load()
        print(f"Loaded {len(documents)} pages from {pdf_file.name}")
        all_docs.extend(documents)

    return all_docs
