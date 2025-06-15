# Imports and Setup

import nest_asyncio
import regex as re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from llama_index.readers.github import GithubRepositoryReader, GithubClient

nest_asyncio.apply()

# GitHub Repository Reader Setup

github_token = "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"
owner = "rahuldixit2612"
repo = "github_poc"
branch = "main"

# Initialize GitHub client
github_client = GithubClient(github_token=github_token, verbose=True)

# Load Python files from the GitHub repository
documents = GithubRepositoryReader(
    github_client=github_client,
    owner=owner,
    repo=repo,
    use_parser=False,
    verbose=False,
    filter_file_extensions=([".py"], GithubRepositoryReader.FilterType.INCLUDE),
).load_data(branch=branch)

print(documents)


# Text Preprocessing

full_text = documents[0].text_resource.text
split_parts = [part.strip() for part in re.split(r'(?=\/\*\*)', full_text) if part.strip()]

# Embedding Generation

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
embeddings = model.encode(split_parts, show_progress_bar=True)

embedding_array = np.array(embeddings).astype('float32')
if embedding_array.ndim == 1:
    embedding_array = embedding_array.reshape(1, -1)


# FAISS Index Creation

dimension = embedding_array.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embedding_array)

# Multi-query Search Setup

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

# Encode all queries
query_embeddings = model.encode(queries, show_progress_bar=True).astype('float32')

if query_embeddings.ndim == 1:
    query_embeddings = query_embeddings.reshape(1, -1)

# Perform Search for Each Query

k = 1  # Top K neighbors to retrieve per query

for idx, query_embedding in enumerate(query_embeddings):
    query_text = queries[idx]
    query_embedding = query_embedding.reshape(1, -1)
    distances, indices = index.search(query_embedding, k)

    print("\n" + "="*60)
    print(f" Query {idx+1}: {query_text}")
    print("="*60)

    for i in range(k):
        neighbor_index = indices[0][i]
        distance = distances[0][i]
        print(f" Match Snippet:\n{split_parts[neighbor_index]}")
        print(f" Distance: {distance:.4f}")
