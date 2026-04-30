# Tesla 10-K RAG Demo

This repository contains a Retrieval-Augmented Generation (RAG) system specialized for Tesla's 2025 10-K filing. It includes a backend API, a CLI for direct interaction, and a frontend interface for a full research experience.

## Features

- **High-Precision Retrieval**: Uses Google GenAI embeddings to find relevant sections from the Tesla 10-K.
- **Financial Charting**: Built-in tool support for rendering financial charts (revenue, profit, etc.) directly in the response.
- **CLI & Web Interface**: Query the analyst via command line or a modern web dashboard.
- **Professional Citations**: Responses include citations to specific SEC Items for data verification.

## Project Structure

- `rag_pipeline.py`: Core RAG logic and implementation of the Tesla Analyst.
- `server.py`: FastAPI server to serve the RAG pipeline.
- `query_tesla.py`: CLI tool for interactive or single-shot queries.
- `scripts/`:
  - `chunk_data.py`: Pre-processes the raw 10-K text into manageable chunks.
  - `vector_store.py`: Generates and manages the local vector database.
- `frontend/`: Web dashboard for interacting with the RAG system.
- `data/tesla/`: Directory for storing raw text, chunks, and embeddings.

## Setup

### Prerequisites

- Python 3.9+
- Google Cloud Project with Vertex AI enabled.
- Application Default Credentials (ADC) configured locally.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/seantupps/rag-demo.git
   cd rag-demo
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables:
   Create a `.env` file in the root directory:
   ```env
   GOOGLE_CLOUD_PROJECT=your-project-id
   GOOGLE_CLOUD_LOCATION=us-central1
   ```

## Usage

### 1. Data Preparation

If the vector store hasn't been generated, run the following:

```bash
# Chunk the raw 10-K data
python scripts/chunk_data.py

# Generate embeddings and build the vector store
python scripts/vector_store.py
```

### 2. Running the RAG Pipeline

#### CLI Mode
Interact with the analyst via terminal:
```bash
python query_tesla.py "What's Tesla's biggest risk factor in 2025?"
```

#### Web Server Mode
Start the backend server:
```bash
python server.py
```
The server will run on `http://localhost:8000`. You can then open `frontend/index.html` in your browser to use the dashboard.

## Verification

To verify the installation and connectivity, you can run the test script:
```bash
python scripts/test_gemini.py
```
