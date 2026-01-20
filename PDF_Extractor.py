import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# Step 1: Load the PDF
def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return documents

# Step 2: Split text into chunks
def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    splits = text_splitter.split_documents(documents)
    return splits

# Step 3: Create embeddings and store in ChromaDB
def store_in_chroma(splits, persist_directory="./chroma_db", embedding_model="all-MiniLM-L6-v2"):
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=f"sentence-transformers/{embedding_model}"
    )
    
    # Create Chroma vectorstore
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    # Persist the database
    vectorstore.persist()
    print(f"Documents stored in ChromaDB at: {persist_directory}")
    return vectorstore

# Step 4: Main function
def pdf_to_chroma(pdf_path, persist_directory="./chroma_db"):
    print("Loading PDF...")
    documents = load_pdf(pdf_path)
    print(f"Loaded {len(documents)} pages")
    
    print("Splitting into chunks...")
    splits = split_documents(documents)
    print(f"Created {len(splits)} text chunks")

    print("Storing in ChromaDB...")
    vectorstore = store_in_chroma(splits, persist_directory)
    
    return vectorstore

# Usage
if __name__ == "__main__":
    pdf_path = "D:\\Rashika\\Git\\RTB\\rtb\\Attention_all_you_need.pdf"  # Replace with your PDF path
    vectorstore = pdf_to_chroma(pdf_path)