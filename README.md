# 🔍 Advanced Retrieval-Augmented Generation (Advanced\_RAG)

This project explores advanced features for **Retrieval-Augmented Generation (RAG)** by combining image-aware data extraction, vector search, and answer refinement logic.

We designed a multi-step system that:

- Extracts information from documents and **uses GPT-Vision** to describe embedded images.
- Retrieves context from a **vector database (Chroma)** using semantic search.
- Evaluates the quality of the answer generated.
- If the result is insufficient, it automatically **rephrases or redirects the question**, optionally invoking a different LLM for a better response.

---

## ⚙️ Tech Stack

- **Chroma** (vector database, using cosine similarity)
- **FastAPI** (backend)
- **Streamlit** (frontend)
- **OpenAI GPT** (including GPT-Vision)
- **Large Embeddings model**
- **Langchain**
- **Langgraph**

We previously tested **PGVector**, but observed poor semantic search results. Switching to Chroma with **cosine similarity** provided significantly improved relevance.

---

## 🧪 Running the Project

To start the full application:

```bash
cd app
docker-compose up
```

Default credentials:

- **User**: `root`
- **Password**: `root`

> Ensure Docker and Docker Compose are installed on your machine.

---

## 📈 Next Steps

- improve the methos tool in retriaver to use the class dastabase. 
- Replace Chroma with **Milvus** for more scalable vector storage.
- Experiment with **BM25 indexing**, which is currently unsupported by Chroma.
- Extend answer evaluation strategies (e.g., confidence scoring, multi-agent debate).

---

## 🧠 Goal Summary

Build a flexible and intelligent RAG pipeline capable of:

- Handling unstructured documents with visual elements.
- Retrieving and grounding responses in the most relevant knowledge.
- Adapting dynamically when responses are insufficient.

---

## 📬 Contributing

Contributions are welcome! Open issues, submit pull requests, or suggest ideas.

---

## 📄 License

MIT License (or update to reflect the correct one for your project).

