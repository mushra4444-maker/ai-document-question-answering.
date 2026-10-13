# ai-document-question-answering.
# AI Document Question & Answer System

## Overview
An AI-powered document question-answering application that allows users to upload PDF or TXT documents and ask questions about their content.

## Features
- Upload PDF and TXT documents.
- Extract and split document text into chunks.
- Generate embeddings and store them in ChromaDB.
- Retrieve relevant document sections using RAG.
- Answer questions using a local LLM.
- Maintain chat history using SQLite.

## Technologies
- Python
- Streamlit
- LangChain
- Ollama — Llama 3.2
- ChromaDB
- Nomic Embeddings
- SQLite
- PyPDF

## Installation

1. Install Ollama and download the required models:
   ```bash
   ollama pull llama3.2
   ollama pull nomic-embed-text
   ```

2. Install Python dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```

3. Start the application:
   ```bash
   python -m streamlit run app.py
   ```

4. Open `http://localhost:8501` in your browser.

## How It Works
The application extracts text from uploaded documents, divides the text into chunks, creates embeddings, retrieves relevant sections, and uses the local LLM to generate answers grounded in the retrieved content.

## Future Improvements
- Support more document formats.
- Add source citations and page references.
- Improve semantic search and retrieval.
- Export chat history and answers.
