#####################################################
# LIFT DOCUMENT INTAKE PROCESSOR [STORAGE]
#####################################################
# Jonathan Wang (jwang15, jonathan.wang@discover.com)
# Greenhouse GenAI Modeling Team

# ABOUT: 
# This project automates the processing of legal documents 
# requesting information about our customers.
# These are handled by LIFT
# (Legal, Investigation, and Fraud Team)

# This is the setup for the Storage in the RAG pipeline.
#####################################################
## TODOS:
# Migrate to Qdrant / Pinecone / Weviate / PGVector / ...

#####################################################
## IMPORTS:
from typing import Optional, IO, List, Tuple

import streamlit as st

import qdrant_client
from llama_index.core import StorageContext
from llama_index.core.storage.docstore.types import BaseDocumentStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex

from pdf_reader import UnstructuredPDFReader

#####################################################
# Get Vector Store
@st.cache_resource
def get_vector_store() -> QdrantVectorStore:
    qdr_client = qdrant_client.QdrantClient(
        location=":memory:"
    )
    qdr_aclient = qdrant_client.AsyncQdrantClient(
        location=":memory:"
    )
    return QdrantVectorStore(client=qdr_client, aclient=qdr_aclient, collection_name='pdf', prefer_grpc=True)


# Get Document Store from List of Documents
# @st.cache_resource  # can't hash a list.
def get_docstore(documents: List) -> BaseDocumentStore:
    """Get the document store from a list of documents."""
    docstore = SimpleDocumentStore()
    docstore.add_documents(documents)
    return docstore


# Get storage context and 
@st.cache_resource  # can't cache the pdf_reader or vector_store
def pdf_to_storage(
    pdf_file_path: Optional[str],
    pdf_file: Optional[IO[bytes]],
    _pdf_reader: UnstructuredPDFReader,
    _vector_store: Optional[QdrantVectorStore]=None,
) -> Tuple[StorageContext, VectorStoreIndex]:
    """Read in PDF and save to storage."""

    # Read the PDF with the PDF reader
    pdf_chunks = _pdf_reader.load_data(pdf_file_path=pdf_file_path, pdf_file=pdf_file)

    # Create Document Store
    docstore = get_docstore(documents=pdf_chunks)
    
    # Create Vector Store if not provided
    if (_vector_store is None):
        _vector_store = get_vector_store()

    # Save into Storage
    storage_context = StorageContext.from_defaults(docstore=docstore, vector_store=_vector_store)
    vector_store_index = VectorStoreIndex(
        pdf_chunks, storage_context=storage_context
    )

    return (storage_context, vector_store_index)