# PDF Q&A RAG with Local LLM

A document analysis chat application that uses RAG (Retrieval Augmented Generation) with a local Mistral-7B model to answer questions about PDF documents.

## Features
- Local LLM processing using Mistral-7B model
- PDF document ingestion and analysis
- WhatsApp-style chat interface built with Streamlit
- Document chunking and semantic search using ChromaDB vector store
- Conversation memory to maintain context
- Source citations for answers

## System Requirements
- Python 3.8+
- At least 16GB RAM (32GB recommended)
- GPU recommended but not required
- Storage space for model and vector store

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd pdf-qa-rag

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install streamlit langchain chromadb sentence-transformers llama-cpp-python pypdf
```

## Project Structure
```
pdf-qa-rag/
├── pdfs/              # Place your PDF documents here
├── model/             # Store the Mistral model here
├── vectorstore/       # Vector store will be created here
├── app.py             # Core RAG implementation
├── streamlit_app.py   # Web interface
└── LICENSE
```

## Model Setup

1. Create required directories:
```bash
mkdir pdfs model vectorstore
```

2. Download the Mistral-7B-Instruct model (quantized version):
```bash
# Download the GGUF model file
curl -L https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf -o ./model/mistral-7b-instruct-v0.1.Q4_K_M.gguf
```

## Usage

1. Place your PDF documents in the `pdfs/` directory

2. Start the Streamlit application:
```bash
streamlit run streamlit_app.py
```

3. Access the web interface at `http://localhost:8501`

## Technical Details

### Core Components

1. **Document Processing** (`app.py`)
   - PDF loading with PyPDFLoader
   - Text chunking using RecursiveCharacterTextSplitter
   - Document embedding using all-MiniLM-L6-v2
   - Vector storage with ChromaDB

2. **Language Model** (`app.py`)
   - Mistral-7B-Instruct model via llama.cpp
   - Conversation memory management
   - RAG implementation using LangChain

3. **User Interface** (`streamlit_app.py`)
   - WhatsApp-style chat interface
   - Real-time question answering
   - Source citations
   - Message timestamps

### Configuration Parameters

```python
# LLM Configuration
LLM_CONFIG = {
    "temperature": 0.75,
    "max_tokens": 4096,
    "n_ctx": 8192,
    "n_batch": 512,
    "top_p": 1
}

# Document Chunking
CHUNK_CONFIG = {
    "chunk_size": 500,
    "chunk_overlap": 50
}
```

## Customization

### Adjusting Model Parameters

To modify the LLM behavior, edit these parameters in `app.py`:
```python
self.llm = LlamaCpp(
    model_path="./model/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    temperature=0.75,  # Increase for more creative responses
    max_tokens=4096,   # Adjust based on your needs
    n_ctx=8192,        # Context window size
    n_batch=512,       # Batch size for processing
    top_p=1,
)
```

### Modifying Document Processing

To change document chunking behavior, adjust these settings in `app.py`:
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,    # Adjust chunk size
    chunk_overlap=50,  # Modify overlap
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)
```

## Troubleshooting

### Common Issues and Solutions

1. **Memory Errors**
   - Reduce `chunk_size` in document splitter
   - Lower `n_ctx` in LLM configuration
   - Use a more quantized model version

2. **Slow Response Times**
   - Enable GPU acceleration
   - Adjust `n_batch` size
   - Reduce context window size

3. **PDF Loading Issues**
   - Ensure PDFs are not corrupted
   - Check file permissions
   - Verify PDF is not password protected

4. **Vector Store Problems**
   - Delete `vectorstore/` directory to rebuild
   - Check disk space
   - Verify embeddings model is downloaded

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.