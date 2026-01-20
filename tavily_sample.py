# To install: pip install tavily-python
import os
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()

client = TavilyClient(os.getenv("TAVILY_API_KEY"))
response = client.search(
    query="What is machine learning?",
    search_depth="advanced"
)
print(response)