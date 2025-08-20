# 📰 Reporter – Multi-Agent Report Generator

This repository contains the **Reporter**, a multi-agent system that generates structured reports using ingested documents from **ChromaDB**.

The system orchestrates multiple agents working together:
- **Orchestrator** – coordinates the workflow.  
- **Retriever** – fetches relevant context from ChromaDB.  
- **Planner** – structures the report outline.  
- **Reporter & Writer** – drafts sections of the report.  
- **Concluder** – validates and finalizes the output.
  
<br>

## 🚀 Features
- Multi-agent collaboration for high-quality report generation.  
- Context retrieval from **ChromaDB**.  
- Validation & structured report outputs. 

<br>

## 📦 Installation & Usage
Note: Ensure [ChromaDB](https://github.com/JuaniLlaberia/chroma_db) is running locally.  

1. Clone the repo:
   ```bash
   git clone https://github.com/your-org/reporter.git
   cd reporter
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
- [document-ingestion](https://github.com/JuaniLlaberia/document-ingestion) → Extracts and ingests documents into ChromaDB.

<br>

# 📌 Coming Soon
- Cloud-ready deployment of ChromaDB.
- User Interface for using the application (Chat-like UI)
