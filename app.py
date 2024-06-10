#####################
# Minimum viable document reader product.
#####################
import config

from streamlit_pdf_viewer import pdf_viewer
from streamlit import session_state as ss
import streamlit as st
from llama_index.core.agent import ReActAgent
from llama_index.core.agent.runner.base import AgentRunner
from llama_index.core.tools import BaseTool, QueryEngineTool, FunctionTool, ToolMetadata
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.query_engine import BaseQueryEngine, SubQuestionQueryEngine
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler, CBEventType
from llama_index.llms.huggingface import HuggingFaceLLM, HuggingFaceInferenceAPI
from llama_index.llms.groq import Groq
from transformers import AutoTokenizer, BitsAndBytesConfig
from llama_index.core import Settings
from llama_index.core import (
    # SimpleDirectoryReader,
    # load_index_from_storage,
    QueryBundle,
    Response,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding, HuggingFaceInferenceAPIEmbedding
from llama_index.core.node_parser import (
    SentenceSplitter,
    SemanticSplitterNodeParser
)
from llama_index.readers.file import PDFReader, UnstructuredReader

import qdrant_client
import nest_asyncio
import asyncio
import re
import glob
from pathlib import Path
from joblib import Parallel, delayed
from collections import defaultdict
from typing import Optional, Any, Dict, List, Set, Tuple, Callable, Dict, cast
from torch import manual_seed
import random
from torch.cuda import empty_cache, list_gpu_processes
import gc
import os

import urllib3, socket
from urllib3.connection import HTTPConnection

import sys
import logging
from llama_index.core.instrumentation import get_dispatcher
from obs_logging import ExampleEventHandler, ExampleSpanHandler

empty_cache()
gc.collect()
empty_cache()

# import pickle

# VectorDB

# Unstructured Document Parsing
# from unstructured.partition.pdf import partition_pdf
# from unstructured.cleaners.core import clean_extra_whitespace, group_broken_paragraphs #, clean_ordered_bullets, clean_bullets, clean_dashes
# from unstructured.chunking.title import chunk_by_title

# Llamaindex Document Parsing
# from llama_index.core import Document
# from llama_index.core.schema import BaseNode, TextNode, ImageNode, IndexNode, NodeWithScore, QueryBundle, NodeRelationship, RelatedNodeInfo

# Embedding and Vector Store

# LLM

# Retrieval, Query Engine

# Engine Tools, Agents


################################################################################
# SETTINGS

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# CUDA GPU memory avoid fragmentation.
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['MAX_SPLIT_SIZE_MB'] = '128'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

# needed to fix some llamaindex issues with asyncio https://github.com/run-llama/llama_index/issues/9978
nest_asyncio.apply()

# Increase socket buffer size for crappy internet.
HTTPConnection.default_socket_options = ( 
    HTTPConnection.default_socket_options + [
    (socket.SOL_SOCKET, socket.SO_SNDBUF, 2000000), 
    (socket.SOL_SOCKET, socket.SO_RCVBUF, 2000000)
    ])


# SESSION STATE INITIALIZATION
if 'pdf_ref' not in ss:
    ss.pdf_ref = None
if 'llm' not in ss:
    ss.llm = None
if 'embed_model' not in ss:
    ss.embed_model = None
if 'node_parser' not in ss:
    ss.node_parser = None
if 'vector_store' not in ss:
    ss.vector_store = None
if 'storage_ctx' not in ss:
    ss.storage_ctx = None
if 'callback_manager' not in ss:
    ss.callback_manager = None
if 'agent' not in ss:
    ss.agent = None
if 'obs_dispatcher' not in ss:
    ss.obs_dispatcher = None
################################################################################
# SCRIPT
st.title("Minimum viable document reader.")

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
ss.pdf_ref = input_file
# Save file to directory
with open('data/input.pdf', 'wb') as f:
    f.write(input_file.read())

# View PDF
pdf_binary_data = ss.pdf_ref.getvalue()
pdf_viewer(input=pdf_binary_data, width=700)

################################################################################
# Set Seeds
if (random.getstate() is None or random.getstate() != 31415926):
    random.seed(31415926)
    manual_seed(31415926)

################################################################################
# Get LLM
@st.cache_resource
def get_llm():
    
    # Groq API (reluctantly okay.)
    llm = Groq(
        model='llama3-8b-8192',
        api_key=config.groq_api_key
    )

    ## HF API (doesn't work with REACT or other Chat Agents)
    # llm = HuggingFaceInferenceAPI(
    #     model_name='meta-llama/Meta-Llama-3-8B-Instruct',
    #     token=config.hf_api_key
    # )


    ## LOCAL GPU (I'm currently too poor for this)
    # bnb_config = BitsAndBytesConfig(
    #     load_in_8bit=True,
    #     # load_in_4bit=True,
    #     # bnb_4bit_use_double_quant=True,
    #     # bnb_4bit_quant_type="nf4",
    #     # bnb_4bit_compute_dtype=torch.bfloat16,
    # )

    # llm = HuggingFaceLLM(
    #     model_name="/content/models/Meta-Llama-3-8B-Instruct/",
    #     tokenizer_name="/content/models/Meta-Llama-3-8B-Instruct/",
    #     tokenizer_kwargs={},  # trust remote code
    #     model_kwargs={'quantization_config': bnb_config},  # trust remote code
    #     # generate_kwargs={'temperature': 0},
    #     generate_kwargs={'do_sample': False}
    # )
    return (llm)


if (ss.llm is None):
    llm = get_llm()
    ss.llm = llm
    Settings.llm = llm

# Get Embedding Model
@st.cache_resource
def get_embed_model() -> HuggingFaceEmbedding:
    embed_model = HuggingFaceEmbedding(
        model_name="mixedbread-ai/mxbai-embed-large-v1",
        # model_kwargs={"device": "cpu"},
    )
    Settings.embed_model = embed_model
    return (embed_model)


if (ss.embed_model is None):
    embed_model = get_embed_model()
    ss.embed_model = embed_model
    Settings.embed_model = embed_model

# Get Callback Manager
@st.cache_resource
def get_callback_manager():
    callback_manager = CallbackManager([LlamaDebugHandler()])
    ss.callback_manager = callback_manager
    Settings.callback_manager = callback_manager


# Get Service Context
@st.cache_resource
def get_node_parser():
    # TODO: callback handlers
    # Node Parser
    node_parser = SemanticSplitterNodeParser(
        buffer_size=1,
        breakpoint_percentile_threshold=95,
        embed_model=ss.embed_model,
        include_metadata=True,
        include_prev_next_rel=True
    )
    ss.node_parser = node_parser
    Settings.node_parser = node_parser
    return (node_parser)
if ss.node_parser is None:
    node_parser = get_node_parser()
    ss.node_parser = node_parser
    Settings.node_parser = node_parser


# Get Vector Store
@st.cache_resource
def get_vector_store():
    qdr_client = qdrant_client.QdrantClient(
        location=":memory:"
    )
    qdr_aclient = qdrant_client.AsyncQdrantClient(
        location=":memory:"
    )
    return QdrantVectorStore(client=qdr_client, aclient=qdr_aclient, collection_name='pdf', prefer_grpc=True)
if ss.vector_store is None:
    ss.vector_store = get_vector_store()


# Get Storage Context
@st.cache_resource
def get_storage_context(persist_dir: Optional[str] = None):
    return StorageContext.from_defaults(
        persist_dir=persist_dir, vector_store=ss.vector_store
    )
if ss.storage_ctx is None:
    ss.storage_ctx = get_storage_context()


# Get Observability
@st.cache_resource
def get_obs():
    dispatcher = get_dispatcher()
    event_handler = ExampleEventHandler()
    span_handler = ExampleSpanHandler()

    dispatcher.add_event_handler(event_handler)
    dispatcher.add_span_handler(span_handler)
    return dispatcher
if (ss.obs_dispatcher is None):
    ss.obs_dispatcher = get_obs()


# Read PDF into Agent
def pdf_to_agent(pdf_doc) -> ReActAgent:
    # Save into Storage
    ss.storage_ctx.docstore.add_documents(pdf_doc)

    index = VectorStoreIndex.from_documents(
        pdf_doc,
        storage_context=ss.storage_ctx,
        # service_context=ss.service_ctx,
        use_async=True
    )

    # Query Engine
    base_query_engine = index.as_query_engine(
        similarity_top_k=3,
        # , "filters": filters
        use_async=True
    )
    response_synth = get_response_synthesizer(response_mode='compact')

    # Query Engine Tools
    sqe_tools = [
        QueryEngineTool(
            query_engine=base_query_engine,
            metadata=ToolMetadata(
                name="Document_Query_Engine",
                description="""A query engine that can answer a question about the user-submitted document. 
Single questions about the document should be asked here.""".strip()
            ),
        )
    ]

    sub_question_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=sqe_tools,
        # service_context=ss.service_ctx,
        response_synthesizer=response_synth,
        use_async=True,
    )

    # Agent Tools
    query_engine_tools = [
        QueryEngineTool(
            query_engine=sub_question_engine,
            metadata=ToolMetadata(
                name="Document_Question_Engine",
                description="""A query engine that can answer questions about a user-submitted document.
Any questions about the document should be asked here.""".strip()
            ),
        )
    ]

    # Agent
    agent = ReActAgent.from_tools(
        tools=query_engine_tools, # type: ignore
        llm=ss.llm,
        verbose=True,
    )
    return agent
if ss.agent is None:
    pdf_loader = PDFReader()
    pdf_doc = pdf_loader.load_data(file=Path('data/input.pdf'))
    ss.agent = pdf_to_agent(pdf_doc)


# Send question to Agent
def handle_chat_message(user_message):
    # Get Response
    response = ss.agent.chat(user_message)
    return response

################################################################################
# Agent Chat UI
# Initialize observability
obs_dispatcher = ss.obs_dispatcher

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
