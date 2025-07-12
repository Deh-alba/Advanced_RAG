from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain.schema import Document

import os
from dotenv import load_dotenv
from typing import List, Optional

from langchain_chroma import Chroma
import logging


class DBVector:


    def __init__(self):
        """
        Initializes the database connection and vector store.
        - Loads environment variables from the specified .env file.
        - Retrieves and sets the OpenAI API key.
        - Initializes OpenAI embeddings using the "text-embedding-3-small" model.
        - Sets up a Chroma vector store with cosine similarity, using the specified collection name and persistence directory.
        Raises:
            ValueError: If the OPENAI_API_KEY is not found in the environment variables.
        """
        # Load environment variables
        env_path = "tools_agents/.env"
        load_dotenv(dotenv_path=env_path)

        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")
        os.environ["OPENAI_API_KEY"] = openai_key

        # Initialize OpenAI embeddings
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

  
        collection_name = "my_docs"

        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory="./chroma_langchain_db_cosine",
            collection_metadata={"hnsw:space": "cosine"}
        )



    def insert_documents(self, documents: List[Document]):
        """Insert LangChain Document objects into the vector store."""
        self.vectorstore.add_documents(documents)

    def query_text(self, query: str, top_k: int = 5):
        """Search for similar documents given a text query."""
        logging.info(f"Querying vector store with query: {query} and top_k: {top_k}")
        data = self.vectorstore.similarity_search_with_relevance_scores(query, k=top_k)
        logging.info(f"Retrieved documents: {data}")
        return data
