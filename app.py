#####################################################
### DOCUMENT PROCESSOR [APP]
#####################################################
### Jonathan Wang

# ABOUT: 
# This creates an app to chat with PDFs.

# This is the APP
# which runs the backend and codes the frontend UI.
#####################################################
### TODO Board:
# Citations like CitationQueryEngine
# https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/query_engine/citation_query_engine.py

#####################################################
### IMPORTS
import config

import os
import sys
import logging

from streamlit_pdf_viewer import pdf_viewer
from streamlit import session_state as ss
import streamlit as st
from io import BytesIO  # type hinting for streamlit documents

import asyncio
import nest_asyncio

# Set seeds for reproducibility
import random
from torch.cuda import empty_cache, manual_seed

from typing import Optional, Any, Dict, List, Set, Tuple, Callable, Dict, cast

import pandas as pd

from joblib import Parallel, delayed

from llama_index.core import Settings

from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import QueryBundle
from llama_index.core.response.notebook_utils import display_source_node
from llama_index.core.postprocessor import SimilarityPostprocessor, SentenceEmbeddingOptimizer
from llama_index.core import Response
from llama_index.core.response_synthesizers import ResponseMode

# These are mine, not llama-index's
from pdf_reader import UnstructuredPDFReader
from storage import get_vector_store, pdf_to_storage
from parsers import get_parser
from retriever import get_retriever
from prompts import get_qa_prompt, get_refine_prompt
from models import get_embedder, get_reranker, get_llm
from engine import get_engine
from obs_logging import get_callback_manager, get_obs

################################################################################
### SETTINGS

# Logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# CUDA GPU memory avoid fragmentation.
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'  # CUDA GPU memory avoid fragmentation.
os.environ['MAX_SPLIT_SIZE_MB'] = '128'
os.environ['SCARF_NO_ANALYTICS'] = 'true'  # get rid of data collection from Unstructured-IO. be sure this is set, otherwise unstructured will take ages pinging the telemetry.

# Asyncio: fix some issues with nesting https://github.com/run-llama/llama_index/issues/9978
nest_asyncio.apply()

# Set seeds
if (random.getstate() is None or random.getstate() != 31415926):
    random.seed(31415926)
    manual_seed(31415926)

# SESSION STATE INITIALIZATION
if 'pdf_ref' not in ss:
    ss.pdf_ref = None
    ss.new_pdf_ref = False
if 'embed_model' not in ss:
    ss.embed_model = None
if 'reranker_model' not in ss:
    ss.reranker_model = None
if 'llm' not in ss:
    ss.llm = None
if 'callback_manager' not in ss:
    ss.callback_manager = None
if 'node_parser' not in ss:
    ss.node_parser = None
if 'vector_store' not in ss:
    ss.vector_store = None
if 'storage_ctx' not in ss:
    ss.storage_ctx = None
if 'vector_store_index' not in ss:
    ss.vector_store_index = None
if 'retriever' not in ss:
    ss.retriever = None
if 'node_postprocessors' not in ss:
    ss.node_postprocessors = None
if 'response_synthesizer' not in ss:
    ss.response_synthesizer = None
if 'engine' not in ss:
    ss.engine = None
if 'agent' not in ss:
    ss.agent = None
if 'observability' not in ss:
    ss.observability = None
if 'pdf_reader' not in ss:
    ss.pdf_reader = None

################################################################################
### SCRIPT

### Get Models and Settings
# Get LLM
if (ss.llm is None):
    llm = get_llm()
    ss.llm = llm
    Settings.llm = llm

# Get Embedding Model
if (ss.embed_model is None):
    embed_model = get_embedder()
    ss.embed_model = embed_model
    Settings.embed_model = embed_model

# Get Reranker
if (ss.reranker_model is None):
    ss.reranker_model = get_reranker()

# Get Callback Manager
if (ss.callback_manager is None):
    callback_manager = get_callback_manager()
    ss.callback_manager = callback_manager
    Settings.callback_manager = callback_manager

# Get Node Parser
if (ss.node_parser is None):
    node_parser = get_parser(embed_model=Settings.embed_model)
    ss.node_parser = node_parser
    Settings.node_parser = node_parser

### PDF Upload UI
st.title("PDF document reader.")

# Upload file
input_file = st.file_uploader(
    "Upload file",
    type='pdf',
    key='input_pdf',
    accept_multiple_files=False
)

# Skip everything else if there is no file.
if (input_file is None):
    st.write("Please upload a PDF file to process.")
    st.stop()

# Save file to session state
if (ss.pdf_ref != input_file):
    ss.new_pdf_ref = True

    ss.pdf_ref = input_file
    # Save file to directory
    with open('data/input.pdf', 'wb') as f:
        f.write(input_file.read())

# View PDF
pdf_viewer(input=ss.pdf_ref.getvalue(), width=700)

### Read in PDF and save to Storage.
# Get PDF Reader
if (ss.pdf_reader is None):
    ss.pdf_reader = UnstructuredPDFReader(embed_model=ss.embed_model, node_parser=ss.node_parser)

# Get Vector Store
if (ss.vector_store is None):
    ss.vector_store = get_vector_store()

# Get Observability
if (ss.observability is None):
    ss.observability = get_obs()

# Get Storage Context
if ((ss.storage_ctx is None) or (ss.new_pdf_ref)):
    ss.storage_ctx, ss.vector_store_index = pdf_to_storage(pdf_file=ss.pdf_ref, pdf_file_path=None, _pdf_reader=ss.pdf_reader, _vector_store=ss.vector_store)

### Get Retrieval on Vector Store Index
if ((ss.retriever is None) or (ss.new_pdf_ref)):
    ss.retriever = get_retriever(
        _vector_store_index=ss.vector_store_index,
        semantic_top_k=10,
        sparse_top_k=6,
        fusion_similarity_top_k=10,
        semantic_weight_fraction=0.6,
        merge_up_thresh=0.5,
        verbose=True,
        _callback_manager=ss.callback_manager
    )

### Get Node Postprocessor Pipeline
if (ss.node_postprocessors is None):
    ss.node_postprocessors = [
        SimilarityPostprocessor(similarity_cutoff=0.01),  # remove any nodes unrelated to the query
        ss.reranker_model,  # rerank
        SentenceEmbeddingOptimizer(percentile_cutoff=0.2),  # remove sentences less related to query. lower is stricter!?
    ]

### Get Response Synthesizer
if (ss.response_synthesizer is None):
    ss.response_synthesizer = get_response_synthesizer(
        response_mode=ResponseMode.COMPACT,
    )

### Get LLM Query Engine
if ((ss.engine is None) or (ss.new_pdf_ref)):
    ss.engine = get_engine(
        retriever=ss.retriever,
        response_synthesizer=ss.response_synthesizer,
        callback_manager=ss.callback_manager
    )

### Get LLM Agent
# TODO
if ((ss.new_pdf_ref)):
    ss.new_pdf_ref = False  # this is the last step in the new-pdf recalculation

# Send question to Agent
def handle_chat_message(user_message):
    # Get Response
    response = ss.engine.query(user_message)
    return response


################################################################################
# Agent Chat UI
# Initialize observability
observability = ss.observability

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
prompt = st.chat_input("Say something")
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = handle_chat_message(prompt)
    # Add assistant message to chat history
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append(
        {"role": "assistant", "content": response})
