# GitHub Repo Analyser

**GitHub Repo Analyser** is an AI-powered tool that enables semantic search and analysis of GitHub repositories using natural language queries.  
It leverages **LangChain**, **Groq LLM**, **Sentence Transformers**, and **FAISS** for fast and intelligent code snippet retrieval.

## Features
- Load and parse GitHub repositories programmatically
- Split code into semantic chunks for better search results
- Generate embeddings for code snippets using sentence-transformers
- Search code using natural language queries
- Agentified workflow using LangChain + Groq LLM

## Installation

```bash
git clone https://github.com/<your-username>/GitHub-Repo-Analyser.git
cd GitHub-Repo-Analyser
python -m venv env
source env/bin/activate  # Linux/Mac
env\Scripts\activate     # Windows
pip install -r requirements.txt
