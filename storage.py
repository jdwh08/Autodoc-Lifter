#####################################################
### DOCUMENT PROCESSOR [STORAGE]
#####################################################
# Jonathan Wang

# ABOUT: 
# This project creates an app to chat with PDFs.

# This is the setup for the Storage in the RAG pipeline.
#####################################################
## TODOS:
# Handle creating multiple vector stores, one for each document which has been processed (?)

#####################################################
## IMPORTS:
import gc
from torch.cuda import empty_cache

from typing import Optional, IO, List, Tuple

import streamlit as st

import qdrant_client
from llama_index.core import StorageContext
from llama_index.core.storage.docstore.types import BaseDocumentStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex

from llama_index.core.settings import Settings
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.node_parser import NodeParser

# Reader and processing
from pdf_reader import UnstructuredPDFReader
from pdf_reader_utils import clean_abbreviations, dedupe_title_chunks, combine_listitem_chunks, remove_header_footer_repeated, chunk_by_header
from metadata_adder import UnstructuredPDFPostProcessor

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
# @st.cache_resource  # can't cache the pdf_reader or vector_store
# def pdf_to_storage(
#     pdf_file_path: Optional[str],
#     pdf_file: Optional[IO[bytes]],
#     _pdf_reader: UnstructuredPDFReader,
#     _embed_model: BaseEmbedding,
#     _node_parser: Optional[NodeParser] = None,
#     _pdf_postprocessor: Optional[UnstructuredPDFPostProcessor] = None,
#     _vector_store: Optional[QdrantVectorStore]=None,
# ) -> Tuple[StorageContext, VectorStoreIndex]:
#     """Read in PDF and save to storage."""

#     # Read the PDF with the PDF reader
#     pdf_chunks = _pdf_reader.load_data(pdf_file_path=pdf_file_path, pdf_file=pdf_file)

#     # Clean the PDF chunks
#     # Insert any cleaners here.
    
#     # TODO: Cleaners to remove repeated header/footer text, overlapping elements, ...
#     pdf_chunks = clean_abbreviations(pdf_chunks)
#     pdf_chunks = dedupe_title_chunks(pdf_chunks)
#     pdf_chunks = combine_listitem_chunks(pdf_chunks)
#     pdf_chunks = remove_header_footer_repeated(pdf_chunks)
#     empty_cache()
#     gc.collect()

#     # Postprocess the PDF nodes.
#     if (_node_parser is None):
#         _node_parser = Settings.node_parser

#     # Combine by semantic headers
#     pdf_chunks = chunk_by_header(pdf_chunks, 1000)
#     pdf_chunks = _node_parser.get_nodes_from_documents(pdf_chunks)

#     if (_pdf_postprocessor is not None):
#         pdf_chunks = _pdf_postprocessor(pdf_chunks)

#     # Add embeddings
#     pdf_chunks = _embed_model(pdf_chunks)

#     # Create Document Store
#     docstore = get_docstore(documents=pdf_chunks)

#     # Create Vector Store if not provided
#     if (_vector_store is None):
#         _vector_store = get_vector_store()

#     ## TODO: Handle images in StorageContext.

#     # Save into Storage
#     storage_context = StorageContext.from_defaults(
#         docstore=docstore, 
#         vector_store=_vector_store
#     )
#     vector_store_index = VectorStoreIndex(
#         pdf_chunks, storage_context=storage_context
#     )

#     return (storage_context, vector_store_index)