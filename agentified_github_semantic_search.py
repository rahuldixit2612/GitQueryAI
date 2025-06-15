# Agentified GitHub Semantic Search

import os
import faiss
import numpy as np
import regex as re
import nest_asyncio

from sentence_transformers import SentenceTransformer
from llama_index.readers.github import GithubRepositoryReader, GithubClient

from langchain.tools import Tool
from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType
from langchain_groq import ChatGroq


# Setup environment and API keys
os.environ["GROQ_API_KEY"] = "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"

# LLM Setup
llm = ChatGroq(model="Gemma2-9b-It", temperature=0)

# Apply async fix for Jupyter environments
nest_asyncio.apply()


# GitHub Repository Setup


github_token = "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"
owner = "rahuldixit2612"
repo = "github_poc"
branch = "main"

github_client = GithubClient(github_token=github_token, verbose=True)
documents = GithubRepositoryReader(
    github_client=github_client,
    owner=owner,
    repo=repo,
    use_parser=False,
    verbose=False,
    filter_file_extensions=([".py"], GithubRepositoryReader.FilterType.INCLUDE),
).load_data(branch=branch)


# Preprocessing
full_text = documents[0].text_resource.text
split_parts = [part.strip() for part in re.split(r'(?=\/\*\*)', full_text) if part.strip()]
# Embedding
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
embeddings = model.encode(split_parts, show_progress_bar=True)
embedding_array = np.array(embeddings).astype('float32')
if embedding_array.ndim == 1:
    embedding_array = embedding_array.reshape(1, -1)




dimension = embedding_array.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embedding_array)



def search_code_snippets(query: str) -> str:
    query_embedding = model.encode([query]).astype('float32')
    distances, indices = index.search(query_embedding, k=1)
    matched_text = split_parts[indices[0][0]]
    return f"Query: {query}\nMatch:\n{matched_text}\nDistance: {distances[0][0]:.4f}"

code_search_tool = Tool(
    name="CodeSearchTool",
    func=search_code_snippets,
    description="Use this tool to search for code snippets related to specific queries from a GitHub repository."
)



agent = initialize_agent(
    tools=[code_search_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

queries = [
    "What method is used to close the current browser window?",
    "What method is used to navigate to a URL?",
    "How is implicit wait handled in this WebDriver setup?",
    "How do you initialize the WebDriver with the default browser?",
    "What browsers are supported by the WebDriverBase class?",
    "What method returns the current page title?",
    "What happens if you call getDriver() before initializing the driver?",
    "How is the WebDriver instance closed and cleaned up completely?",
    "Which method sets Chrome-specific preferences and arguments?",
    "What method returns the current page URL?"
]

for q in queries:
    print("\n" + "="*60)
    print(agent.run(q))
