# AskHR — Enterprise Knowledge Assistant (RAG)

AskHR is a small, educational Retrieval-Augmented Generation (RAG) demo. It answers
employee questions (onboarding, benefits, conduct, time off) using the **public
Osio Labs Employee Handbook**, and shows which handbook files the answer came from.

> This is an independent learning project. It is **not** affiliated with Osio Labs,
> and it is **not** official or current HR guidance. It runs locally only.

**Status:** work in progress.

## Planned tech stack

- **LangChain** for loading, chunking, prompts and the answer chain
- **Pinecone** hosted embeddings (`llama-text-embed-v2`) and a dense vector index
- **Ollama** running `qwen3:4b-instruct` locally
- **Streamlit** for the chat interface
- **pytest**, **Ruff** and **GitHub Actions** for tests, linting and CI

## Data source

[Osio Labs Employee Handbook](https://github.com/OsioLabs/emphandbook) (CC0-1.0).
The handbook files are downloaded locally into `data/handbook/` and are not committed.
