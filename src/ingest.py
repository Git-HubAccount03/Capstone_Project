"""
===============================================================================
File Name : ingest.py

Purpose:
--------
Reads all PDF files from the documents folder,
splits them into chunks,
creates embeddings,
and stores them inside Chroma Vector Database.
===============================================================================
"""

import os
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from config import (
    DOCUMENT_FOLDER,
    VECTOR_DB,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)


def load_documents():
    """
    Load all PDF files from documents folder.
    """

    documents = []

    print("\nLoading PDF files...\n")

    for file in os.listdir(DOCUMENT_FOLDER):

        if file.lower().endswith(".pdf"):

            pdf_path = os.path.join(DOCUMENT_FOLDER, file)

            print(f"Reading : {file}")

            loader = PyPDFLoader(pdf_path)

            docs = loader.load()

            documents.extend(docs)

    print(f"\nTotal Pages Loaded : {len(documents)}")

    return documents


def split_documents(documents):
    """
    Split documents into smaller chunks.
    """

    print("\nSplitting documents into chunks...\n")

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=CHUNK_SIZE,

        chunk_overlap=CHUNK_OVERLAP,

        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = splitter.split_documents(documents)

    print(f"Total Chunks Created : {len(chunks)}")

    return chunks


def create_vector_database(chunks):
    """
    Create Chroma Vector Database.
    """

    print("\nLoading Embedding Model...\n")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    print("Embedding Model Loaded Successfully.")

    if os.path.exists(VECTOR_DB):

        print("\nExisting Vector Database Found.")

        print("Deleting old Vector Database...")

        shutil.rmtree(VECTOR_DB)

    print("\nCreating Vector Database...\n")

    Chroma.from_documents(

        documents=chunks,

        embedding=embeddings,

        persist_directory=str(VECTOR_DB)

    )

    print("\nVector Database Created Successfully!")


def main():

    print("=" * 70)
    print("Document Ingestion Started")
    print("=" * 70)

    documents = load_documents()

    if len(documents) == 0:

        print("\nNo PDF files found inside documents folder.")

        return

    chunks = split_documents(documents)

    create_vector_database(chunks)

    print("\n")
    print("=" * 70)
    print("Document Ingestion Completed Successfully")
    print("=" * 70)


if __name__ == "__main__":

    main()