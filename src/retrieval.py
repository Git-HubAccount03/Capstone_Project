"""
===============================================================================
File Name : retrieval.py

Purpose:
--------
Loads the Chroma Vector Database and retrieves the most relevant document chunks.
===============================================================================
"""

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config import VECTOR_DB, EMBEDDING_MODEL


class DocumentRetriever:

    def __init__(self):

        print("=" * 60)
        print("Loading Embedding Model...")
        print("=" * 60)

        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )

        print("Embedding Model Loaded Successfully.\n")

        print("=" * 60)
        print("Loading Chroma Vector Database...")
        print("=" * 60)

        self.vector_db = Chroma(
            persist_directory=str(VECTOR_DB),
            embedding_function=self.embeddings
        )

        print("Vector Database Loaded Successfully.\n")

        # Create Retriever
        self.retriever = self.vector_db.as_retriever(
            search_kwargs={"k": 3}
        )

    def retrieve_documents(self, question):

        docs = self.retriever.invoke(question)

        return docs

    def get_context(self, question):

        docs = self.retrieve_documents(question)

        context = "\n\n".join(
            doc.page_content for doc in docs
        )

        return context


if __name__ == "__main__":

    retriever = DocumentRetriever()

    while True:

        question = input("\nAsk Question (type exit to quit): ")

        if question.lower() == "exit":
            break

        docs = retriever.retrieve_documents(question)

        print("\n")

        print("=" * 70)
        print("Retrieved Documents")
        print("=" * 70)

        for i, doc in enumerate(docs, start=1):

            print(f"\nChunk {i}")

            print("-" * 60)

            print(doc.page_content)