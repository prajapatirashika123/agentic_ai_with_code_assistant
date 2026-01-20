from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Load existing vectorstore
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)


def ask_question(question):
    results = vectorstore.similarity_search(question, k=3)
    answer = "\n\n".join([doc.page_content for doc in results])
    return answer

ask_question("Explain the attention mechanism in transformers.")  
