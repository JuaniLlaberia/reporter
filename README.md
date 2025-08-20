# 📄 Document Ingestion Service

This repository contains the **document ingestion pipeline**, responsible for processing documents, extracting structured content and images, generating embeddings, and storing everything in **ChromaDB**.

<br>

## 🚀 Features
- Document parsing with [docling](https://github.com/docling).
- Extraction of text, tables, and images.
- Chunking & embeddings generation.
- Automatic ingestion into the ChromaDB instance.

<br>

## 📦 Installation & Usage
Note: Ensure [ChromaDB](https://github.com/JuaniLlaberia/chroma_db) is running locally.  

1. Clone the repo:
   ```bash
   git clone https://github.com/your-org/document-ingestion.git
   cd document-ingestion
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run ChromaDB locally (Flask endpoint):

   ```bash
   python .
   ```

<br>

# 🔗 Related Repositories
- [chromadb-service](https://github.com/JuaniLlaberia/chroma_db) → Vector database backend.
- [reporter](https://github.com/JuaniLlaberia/reporter) → Uses ChromaDB data for multi-agent report generation.

<br>

# 📌 Coming Soon
- Cloud-ready deployment of ChromaDB.
- User Interface for using the application (Chat-like UI)
