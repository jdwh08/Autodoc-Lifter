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

################################################################################
### PROGRAM SETTINGS
import gc

import config
import os
import sys
import logging
import random
from torch.cuda import empty_cache, manual_seed

import asyncio
import nest_asyncio

# Logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# API Keys
os.environ['OPENAI_API_KEY'] = config.openai_api_key
os.environ['GROQ_API_KEY'] = config.groq_api_key

# CUDA GPU memory avoid fragmentation.
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'  # CUDA GPU memory avoid fragmentation.
os.environ['MAX_SPLIT_SIZE_MB'] = '128'
os.environ['SCARF_NO_ANALYTICS'] = 'true'  # get rid of data collection from Unstructured-IO. be sure this is set, otherwise unstructured will take ages pinging the telemetry.
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

# Asyncio: fix some issues with nesting https://github.com/run-llama/llama_index/issues/9978
nest_asyncio.apply()

# Set seeds
if (random.getstate() is None or random.getstate() != 31415926):
    random.seed(31415926)
    manual_seed(31415926)


#####################################################
### PROGRAM IMPORTS

from streamlit_pdf_viewer import pdf_viewer
from streamlit import session_state as ss
import streamlit as st
import base64

from typing import Optional, Any, Dict, List, Set, Tuple, Callable, Dict, cast, IO

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
from pdf_reader_utils import UnstructuredPDFPostProcessor, RegexMetadataAdder, KeywordMetadataAdder, TextSummaryMetadataAdder, TableSummaryMetadataAdder, ImageSummaryMetadataAdder
from pdf_reader_utils import DATE_REGEX, TIME_REGEX, EMAIL_REGEX, MAIL_ADDR_REGEX, PHONE_REGEX
from storage import get_vector_store, pdf_to_storage
from parsers import get_parser
from retriever import get_retriever
from prompts import get_qa_prompt, get_refine_prompt
from models import get_sat_sentence_splitter, get_embedder, get_reranker, get_llm, get_multimodal_llm
from engine import get_engine
from obs_logging import get_callback_manager, get_obs

#########################################################################
### SESSION STATE INITIALIZATION
st.set_page_config(layout="wide")

if 'pdf_ref' not in ss:
    ss.input_pdf = None
if 'sentence_model' not in ss:
    ss.sentence_model = None  # sentence splitting model, as alternative to nltk/PySBD
if 'embed_model' not in ss:
    ss.embed_model = None
if 'reranker_model' not in ss:
    ss.reranker_model = None
if 'llm' not in ss:
    ss.llm = None
if 'multimodal_llm' not in ss:
    ss.multimodal_llm = None
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
if 'pdf_postprocessor' not in ss:
    ss.pdf_postprocessor = None

################################################################################
### SCRIPT

### Get Models and Settings
# Get Vision LLM
if (ss.multimodal_llm is None):
    vision_llm = get_multimodal_llm(
        # model_name='dwb2023/phi-3-vision-128k-instruct-quantized',
        # tokenizer_model_name='microsoft/Phi-3-vision-128k-instruct',
    )
    ss.multimodal_llm = vision_llm

# Get LLM
if (ss.llm is None):
    llm = get_llm()
    ss.llm = llm
    Settings.llm = llm

# Get Sentence Splitting Model.
if (ss.sentence_model is None):
    sent_splitter = get_sat_sentence_splitter('sat-3l-sm')
    ss.sentence_model = sent_splitter

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
    node_parser = get_parser(embed_model=Settings.embed_model, sentence_model=ss.sentence_model, callback_manager=ss.callback_manager)
    ss.node_parser = node_parser
    Settings.node_parser = node_parser

#### Get Observability
if (ss.observability is None):
    obs = get_obs()

### Get PDF Reader
if (ss.pdf_reader is None):
    ss.pdf_reader = UnstructuredPDFReader()

### Get PDF Reader Postprocessing
if (ss.pdf_postprocessor is None):
    # Get embedding
    # regex_adder = RegexMetadataAdder(regex_pattern=)  # Are there any that I need?
    keyword_adder = KeywordMetadataAdder()
    # table_summary_adder = TableSummaryMetadataAdder(llm=ss.llm)
    # image_summary_adder = ImageSummaryMetadataAdder(llm=ss.multimodal_llm)
    
    pdf_postprocessor = UnstructuredPDFPostProcessor(
        embed_model=ss.embed_model, 
        metadata_adders=[keyword_adder] #, table_summary_adder, image_summary_adder]
    )
    ss.pdf_postprocessor = pdf_postprocessor

#### Get Vector Store
if (ss.vector_store is None):
    ss.vector_store = get_vector_store()

#### Get Observability
if (ss.observability is None):
    ss.observability = get_obs()
    observability = ss.observability

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

# @st.cache_resource
def pdf_to_agent(file_io) -> None:
    """Handles processing a new source PDF file document."""
    ### Get file name
    file_name = file_io.name
    
    ### Save Locally
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/input.pdf'), 'wb') as f:
        f.write(file_io.getbuffer())
        
    ### Get Storage Context
    with (st.spinner(f"Processing input file, this make take some time...")):
        ss.storage_ctx, ss.vector_store_index = pdf_to_storage(
            pdf_file=None, 
            pdf_file_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/input.pdf'), 
            _pdf_reader=ss.pdf_reader, 
            _embed_model=ss.embed_model,
            _pdf_postprocessor=ss.pdf_postprocessor, 
            _vector_store=ss.vector_store
        )
    ### Get Retrieval on Vector Store Index
    with (st.spinner(f"Building retriever for the input file...")):
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
    ### Get LLM Query Engine
    with (st.spinner(f"Building query responder for the input file...")):
        ss.engine = get_engine(
            retriever=ss.retriever,
            response_synthesizer=ss.response_synthesizer,
            callback_manager=ss.callback_manager
        )
    # TODO:
    ### Get LLM Agent
    
    # All done!
    st.toast("All done!")
    ### Cleaning
    empty_cache()
    gc.collect()


###############################################################################
### UI
col_left, col_right = st.columns([1, 1])

if 'uploaded_files' not in ss:
    ss.uploaded_files = []
if 'selected_file' not in ss:
    ss.selected_file = None

if 'chat_messages' not in ss:
    ss.chat_messages = []

################################################################################
### PDF Upload UI (Left Panel)

with st.sidebar:
    uploaded_files = st.file_uploader(
        label='Upload a PDF file.',
        type='pdf',
        accept_multiple_files=True,
        label_visibility='collapsed',
    )

    uploaded_files = uploaded_files or []  # handle case when no file is uploaded
    for uploaded_file in uploaded_files:
        if (uploaded_file not in ss.uploaded_files):
            ss.uploaded_files.append(uploaded_file)
            pdf_to_agent(uploaded_file)
        
    if (ss.selected_file is None and ss.uploaded_files):
        ss.selected_file = ss.uploaded_files[0]
    
    file_names = [file.name for file in ss.uploaded_files]
    selected_file_name = st.radio("Uploaded Files:", file_names)
    if selected_file_name:
        ss.selected_file = [file for file in ss.uploaded_files if file.name == selected_file_name][0]

################################################################################
### PDF Display UI (Middle Panel)
@st.cache_data
def get_pdf_display(file, app_width: str = "100%", app_height: str = "500", starting_page_number: Optional[int] = None) -> str:
    # Read file as binary
    file_bytes = file.getbuffer()
    base64_pdf = base64.b64encode(file_bytes).decode('utf-8')
    
    pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}"'  # iframe vs embed
    if starting_page_number is not None:
        pdf_display += f'#page={starting_page_number}'
    pdf_display += f' width={app_width} height="{app_height}" type="application/pdf"></iembed>'  # iframe vs embed
    return (pdf_display)

with col_left:
    if (ss.selected_file is not None):
        selected_file = ss.selected_file
        selected_file_name = selected_file.name
        
        if (selected_file.type == "application/pdf"):
            pdf_display = get_pdf_display(selected_file, app_width="100%", app_height="550")
            st.markdown(pdf_display, unsafe_allow_html=True)
    else:
        selected_file_name = "Upload a file."
        st.markdown(f"## {selected_file_name}")

################################################################################
### Chat UI (Right Panel)
def handle_chat_message(user_message):
    # Get Response
    response = ss.engine.query(user_message)
    return response

with col_right:
    messages_container = st.container(height=475, border=False)
    input_container = st.container(height=80, border=False)

with messages_container:
    for message in ss.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

with input_container:
    # Accept user input
    prompt = st.chat_input("Say something")

if prompt:
    with messages_container:
        with st.chat_message("user"):
            st.markdown(prompt)
            ss.chat_messages.append({"role": "user", "content": prompt})

        with st.spinner(f"Generating response..."):
            # Get Response
            response = handle_chat_message(prompt)

        if response:
            ss.chat_messages.append(
                {"role": "assistant", "content": response}
            )
            with st.chat_message("assistant"):
                st.markdown(response)