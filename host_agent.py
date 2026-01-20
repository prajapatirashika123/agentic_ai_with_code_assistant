import os
from crewai import Agent, Task, Crew
from crewai_tools import TavilySearchTool
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Set API keys
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

# Load vectorstore from persisted Chroma DB
try:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
except Exception:
    vectorstore = None  # Will handle in VectorSearchTool

# Custom vector search tool (adapt to your store)
from crewai.tools import BaseTool

class VectorSearchTool(BaseTool):
    name: str = "Vector Store Search"
    description: str = "Searches the processed documents vector store for relevant info."
    
    def _run(self, query: str) -> str:
        if vectorstore is None:
            return "Vector store not initialized. Please process documents first."
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(query)
        return "".join([doc.page_content for doc in docs])

# Tools
vector_tool = VectorSearchTool()
web_tool = TavilySearchTool()

# Agents
doc_agent = Agent(
    role="Document Retrieval Specialist",
    goal="Extract precise info from processed documents via vector search",
    backstory="Expert in RAG over enterprise docs like fraud detection PDFs.",
    tools=[vector_tool],
    verbose=True,
    llm="groq/llama-3.1-8b-instant"  # Groq Llama 3.1 8B
)

web_agent = Agent(
    role="Web Research Specialist",
    goal="Supplement with current web data when docs lack recency",
    backstory="Skilled at targeted searches for AI/ML updates.",
    tools=[web_tool],
    verbose=True,
    llm="groq/llama-3.1-8b-instant"  # Groq Llama 3.1 8B
)

# Tasks (hierarchical process)
vector_task = Task(
    description="Query vector store first for document-specific answers: {query}",
    expected_output="Retrieved document excerpts with sources.",
    agent=doc_agent
)

web_task = Task(
    description="If vector results incomplete, web search for updates on {query}. Combine with vector findings.",
    expected_output="Augmented report with web insights.",
    agent=web_agent,
    context=[vector_task]  # Sequential dependency
)

# Crew
crew = Crew(
    agents=[doc_agent, web_agent],
    tasks=[vector_task, web_task],
    verbose=True
)

# Run
result = crew.kickoff(inputs={"query": "Latest fraud detection techniques in my docs?"})
print(result)