 # Document QA System with Local LLM

## Model Setup
This project requires [meta-llama/Llama-3.2-1B] which can be downloaded from [https://huggingface.co/meta-llama/Llama-3.2-1B].
After downloading, place the model files in the `models/MYllama/` directory.


A locally-running document question-answering system that processes documents, generates embeddings, and answers user queries using local language models.

## Features
- Document processing and chunking for efficient analysis
-ChromaDB persistence for storage for production use
-Local LLM integration for private, offline question answering
- Customizable chunk size and overlap parameters


## Technical Implementation
- **Document Processing**: Extracts text from PDFs/documents and splits into optimal chunks
- **Embedding Generation**: Creates vector representations of text chunks
- **Vector Storage**: Efficiently stores and retrieves relevant document sections
- **Question Answering**: Uses local LLM to generate accurate answers from retrieved context

### Prerequisites
- Python 3.8+
- Required packages: `pip install -r requirements.txt`

### Setup
1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Download model weights (instructions below)
4. Run the application: `python main.py`


