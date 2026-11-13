import os
import hashlib
from pathlib import Path

from dotenv import load_dotenv
from pinecone import Pinecone,ServerlessSpec
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeEmbeddings

load_dotenv()


PDF_FILE = "Medical_book.pdf"

# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

INDEX_NAME = os.environ["PINECONE_INDEX"]


# INDEX_NAME = pc.Index("rag-medical-demo")

# EMBED_MODEL = "llama-text-embed-v2"

embeddings = PineconeEmbeddings(
    model="llama-text-embed-v2"
)


existing_indexes = pc.list_indexes().names()

if INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=INDEX_NAME,
        dimension=1024,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

docs = PyPDFLoader(PDF_FILE).load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

splits = splitter.split_documents(docs)

PineconeVectorStore.from_documents(
    documents=splits,
    embedding=embeddings,
    index_name=INDEX_NAME
)

print("Upload completed")