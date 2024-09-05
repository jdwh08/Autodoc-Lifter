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
# Try ColPali? https://huggingface.co/vidore/colpali 

#####################################################
### PROGRAM IMPORTS
from __future__ import annotations

import base64
import gc
import logging
import os
import random
import sys
import warnings
from pathlib import Path
from typing import Any, cast

import nest_asyncio
import numpy as np
import streamlit as st
from llama_index.core import Settings, get_response_synthesizer
from llama_index.core.base.llms import BaseLLM
from llama_index.core.postprocessor import (
    SentenceEmbeddingOptimizer,
    SimilarityPostprocessor,
)
from llama_index.core.response_synthesizers import ResponseMode
from streamlit import session_state as ss
from summary import (
    ImageSummaryMetadataAdder,
    TableSummaryMetadataAdder,
    get_tree_summarizer,
)
from torch.cuda import (
    empty_cache,
    get_device_name,
    is_available,
    manual_seed,
    mem_get_info,
)
from transformers import set_seed

# Own Modules
from agent import doclist_to_agent
from citation import get_citation_builder
from full_doc import FullDocument
from keywords import KeywordMetadataAdder
from metadata_adder import UnstructuredPDFPostProcessor
from models import get_embedder, get_llm, get_multimodal_llm, get_reranker
from obs_logging import get_callback_manager, get_obs
from pdf_reader import UnstructuredPDFReader
from pdf_reader_utils import (
    chunk_by_header,
    clean_abbreviations,
    combine_listitem_chunks,
    dedupe_title_chunks,
    remove_header_footer_repeated,
)
from parsers import get_parser
from prompts import get_qa_prompt, get_refine_prompt

#####################################
### SETTINGS
# Logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# CUDA GPU memory avoid fragmentation.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # avoid vram frag
os.environ["MAX_SPLIT_SIZE_MB"] = "128"
os.environ["SCARF_NO_ANALYTICS"] = "true"  # get rid of data collection from Unstructured
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

os.environ["HF_HOME"] = "/data/.huggingface"  # save cached models on disk.

SEED = 31415926

print(f"CUDA Availablility: {is_available()}")
print(f"CUDA Device Name: {get_device_name()}")
print(f"CUDA Memory: {mem_get_info()}")

gc.collect()
empty_cache()

# Asyncio: fix some issues with nesting https://github.com/run-llama/llama_index/issues/9978
nest_asyncio.apply()

# Set seeds
if (random.getstate() is None):
    random.seed(SEED)  # python
    np.random.seed(SEED)  # numpy  # TODO(Jonathan Wang): Replace with generator
    manual_seed(SEED)  # pytorch
    set_seed(SEED)  # transformers

# API Keys
os.environ["HF_TOKEN"] = st.secrets["huggingface_api_token"]
os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
os.environ["GROQ_API_KEY"] = st.secrets["groq_api_key"]

#########################################################################
### SESSION STATE INITIALIZATION
st.set_page_config(layout="wide")

if "pdf_ref" not in ss:
    ss.input_pdf = []
if "doclist" not in ss:
    ss.doclist = []
if "pdf_reader" not in ss:
    ss.pdf_reader = None
if "pdf_postprocessor" not in ss:
    ss.pdf_postprocessor = None
# if 'sentence_model' not in ss:
    # ss.sentence_model = None  # sentence splitting model, as alternative to nltk/PySBD
if "embed_model" not in ss:
    ss.embed_model = None
    gc.collect()
    empty_cache()
if "reranker_model" not in ss:
    ss.reranker_model = None
    gc.collect()
    empty_cache()
if "llm" not in ss:
    ss.llm = None
    gc.collect()
    empty_cache()
if "multimodal_llm" not in ss:
    ss.multimodal_llm = None
    gc.collect()
    empty_cache()
if "callback_manager" not in ss:
    ss.callback_manager = None
if "node_parser" not in ss:
    ss.node_parser = None
if "node_postprocessors" not in ss:
    ss.node_postprocessors = None
if "response_synthesizer" not in ss:
    ss.response_synthesizer = None
if "tree_summarizer" not in ss:
    ss.tree_summarizer = None
if "citation_builder" not in ss:
    ss.citation_builder = None
if "agent" not in ss:
    ss.agent = None
if "observability" not in ss:
    ss.observability = None

if "uploaded_files" not in ss:
    ss.uploaded_files = []
if "selected_file" not in ss:
    ss.selected_file = None

if "chat_messages" not in ss:
    ss.chat_messages = []

################################################################################
### SCRIPT

st.markdown("""
        <style>
                .block-container {
                    padding-top: 3rem;
                    padding-bottom: 0rem;
                    padding-left: 3rem;
                    padding-right: 3rem;
                }
        </style>
        """, unsafe_allow_html=True)

### # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
### UI
st.text("Autodoc Lifter Local PDF Chatbot (Built with Meta🦙3)")
col_left, col_right = st.columns([1, 1])

### # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
### PDF Upload UI (Left Panel)
with st.sidebar:
    uploaded_files = st.file_uploader(
        label="Upload a PDF file.",
        type="pdf",
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

### # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
### PDF Display UI (Middle Panel)
# NOTE: This currently only displays the PDF, which requires user interaction (below)

### # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
### Chat UI (Right Panel)

with col_right:
    messages_container = st.container(height=475, border=False)
    input_container = st.container(height=80, border=False)

with messages_container:
    for message in ss.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

with input_container:
    # Accept user input
    prompt = st.chat_input("Ask your question about the document here.")

### # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
### Get Models and Settings
# Get Vision LLM
if (ss.multimodal_llm is None):
    print(f"CUDA Memory Pre-VLLM: {mem_get_info()}")
    vision_llm = get_multimodal_llm()
    ss.multimodal_llm = vision_llm

# Get LLM
if (ss.llm is None):
    print(f"CUDA Memory Pre-LLM: {mem_get_info()}")
    llm = get_llm()
    ss.llm = llm
    Settings.llm = cast(llm, BaseLLM)

# Get Sentence Splitting Model.
# if (ss.sentence_model is None):
#     sent_splitter = get_sat_sentence_splitter('sat-3l-sm')
#     ss.sentence_model = sent_splitter

# Get Embedding Model
if (ss.embed_model is None):
    print(f"CUDA Memory Pre-Embedding: {mem_get_info()}")
    embed_model = get_embedder()
    ss.embed_model = embed_model
    Settings.embed_model = embed_model

# Get Reranker
if (ss.reranker_model is None):
    print(f"CUDA Memory Pre-Reranking: {mem_get_info()}")
    ss.reranker_model = get_reranker()

# Get Callback Manager
if (ss.callback_manager is None):
    callback_manager = get_callback_manager()
    ss.callback_manager = callback_manager
    Settings.callback_manager = callback_manager

# Get Node Parser
if (ss.node_parser is None):
    node_parser = get_parser(
        embed_model=Settings.embed_model,
        callback_manager=ss.callback_manager
    )
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
    keyword_adder = KeywordMetadataAdder(metadata_name="keywords")
    table_summary_adder = TableSummaryMetadataAdder(llm=ss.llm)
    image_summary_adder = ImageSummaryMetadataAdder(llm=ss.multimodal_llm)

    pdf_postprocessor = UnstructuredPDFPostProcessor(
        embed_model=ss.embed_model,
        metadata_adders=[keyword_adder, table_summary_adder, image_summary_adder]
    )
    ss.pdf_postprocessor = pdf_postprocessor

#### Get Observability
if (ss.observability is None):
    ss.observability = get_obs()
    observability = ss.observability

### Get Node Postprocessor Pipeline
if (ss.node_postprocessors is None):
    from nltk.tokenize import PunktTokenizer
    punkt_tokenizer = PunktTokenizer()
    ss.node_postprocessors = [
        SimilarityPostprocessor(similarity_cutoff=0.01),  # remove nodes unrelated to query
        ss.reranker_model,  # rerank
        # remove sentences less related to query. lower is stricter
        SentenceEmbeddingOptimizer(tokenizer_fn=punkt_tokenizer.tokenize, percentile_cutoff=0.2),
    ]

### Get Response Synthesizer
if (ss.response_synthesizer is None):
    ss.response_synthesizer = get_response_synthesizer(
        response_mode=ResponseMode.COMPACT,
        text_qa_template=get_qa_prompt(),
        refine_template=get_refine_prompt()
    )

### Get Tree Summarizer
if (ss.tree_summarizer is None):
    ss.tree_summarizer = get_tree_summarizer()

### Get Citation Builder
if (ss.citation_builder is None):
    ss.citation_builder = get_citation_builder()

### # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
### Handle User Interaction
def handle_new_pdf(file_io: Any) -> None:
    """Handle processing a new source PDF file document."""
    with st.sidebar:
        with (st.spinner("Reading input file, this make take some time...")):
            ### Save Locally
            # TODO(Jonathan Wang): Get the user to upload their file with a reference name in a separate tab.
            if not Path(__file__).parent.joinpath("data").exists():
                print("NEWPDF: Making data directory...")
                Path(__file__).parent.joinpath("data").mkdir(parents=True)
            with open(Path(__file__).parent.joinpath("data/input.pdf"), "wb") as f:
                print("NEWPDF: Writing input file...")
                f.write(file_io.getbuffer())

            ### Create Document
            print("NEWPDF: Building Document...")
            new_document = FullDocument(
                name="input.pdf",
                file_path=Path(__file__).parent.joinpath("data/input.pdf"),
            )

            #### Process document.
            print("NEWPDF: Writing input file...")
            new_document.file_to_nodes(
                reader=ss.pdf_reader,
                postreaders=[
                    clean_abbreviations, dedupe_title_chunks, combine_listitem_chunks,
                    remove_header_footer_repeated, chunk_by_header
                ],
                node_parser=ss.node_parser,
                postparsers=[ss.pdf_postprocessor],
            )

        ### Get Storage Context
        with (st.spinner("Processing input file, this make take some time...")):
            new_document.nodes_to_summary(summarizer=ss.tree_summarizer)
            new_document.summary_to_oneline(summarizer=ss.tree_summarizer)
            new_document.nodes_to_document_keywords()
            new_document.nodes_to_storage()
    ### Get Retrieval on Vector Store Index
        with (st.spinner("Building retriever for the input file...")):
            new_document.storage_to_retriever(callback_manager=ss.callback_manager)
    ### Get LLM Query Engine
        with (st.spinner("Building query responder for the input file...")):
            new_document.retriever_to_engine(
                response_synthesizer=ss.response_synthesizer,
                callback_manager=ss.callback_manager
            )
            new_document.engine_to_sub_question_engine()

    ### Officially Add to Document List
        ss.uploaded_files.append(uploaded_file)  # Left UI Bar
        ss.doclist.append(new_document)  # Document list for RAG.  # TODO(Jonathan Wang): Fix potential duplication.

    ### Get LLM Agent
        with (st.spinner("Building LLM Agent for the input file...")):
            agent = doclist_to_agent(ss.doclist)
            ss.agent = agent

    # All done!
    st.toast("All done!")

    # Display summary of new document in chat.
    with messages_container:
        ss.chat_messages.append(
            {"role": "assistant", "content": new_document.summary_oneline}
        )
        with st.chat_message("assistant"):
            st.markdown(new_document.summary_oneline)

    ### Cleaning
    empty_cache()
    gc.collect()


def handle_chat_message(user_message: str) -> str:
    # Get Response
    if (not hasattr(ss, "doclist") or len(ss.doclist) == 0):
        return "Please upload a document to get started."

    if (not hasattr(ss, "agent")):
        warnings.warn("No LLM Agent found. Attempting to create one.", stacklevel=2)
        with st.sidebar, (st.spinner("Building LLM Agent for the input file...")):
            agent = doclist_to_agent(ss.doclist)
            ss.agent = agent

    response = ss.agent.query(user_message)
    # Get citations if available
    response = ss.citation_builder.get_citations(response, citation_threshold=60)
    # Add citations to response text
    response_with_citations = ss.citation_builder.add_citations_to_response(response)
    return str(response_with_citations.response)

@st.cache_data
def get_pdf_display(
    file: Any,
    app_width: str = "100%",
    app_height: str = "500",
    starting_page_number: int | None = None
) -> str:
    # Read file as binary
    file_bytes = file.getbuffer()
    base64_pdf = base64.b64encode(file_bytes).decode("utf-8")

    pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}"'  # TODO(Jonathan Wang): iframe vs embed
    if starting_page_number is not None:
        pdf_display += f"#page={starting_page_number}"
    pdf_display += f' width={app_width} height="{app_height}" type="application/pdf"></iembed>'  # iframe vs embed
    return (pdf_display)

# Upload
with st.sidebar:
    uploaded_files = uploaded_files or []  # handle case when no file is uploaded
    for uploaded_file in uploaded_files:
        if (uploaded_file not in ss.uploaded_files):
            handle_new_pdf(uploaded_file)

    if (ss.selected_file is None and ss.uploaded_files):
        ss.selected_file = ss.uploaded_files[-1]

    file_names = [file.name for file in ss.uploaded_files]
    selected_file_name = st.radio("Uploaded Files:", file_names)
    if selected_file_name:
        ss.selected_file = [file for file in ss.uploaded_files if file.name == selected_file_name][-1]

with col_left:
    if (ss.selected_file is None):
        selected_file_name = "Upload a file."
        st.markdown(f"## {selected_file_name}")

    elif (ss.selected_file is not None):
        selected_file = ss.selected_file
        selected_file_name = selected_file.name

        if (selected_file.type == "application/pdf"):
            pdf_display = get_pdf_display(selected_file, app_width="100%", app_height="550")
            st.markdown(pdf_display, unsafe_allow_html=True)

# Chat
if prompt:
    with messages_container:
        with st.chat_message("user"):
            st.markdown(prompt)
            ss.chat_messages.append({"role": "user", "content": prompt})

        with st.spinner("Generating response..."):
            # Get Response
            response = handle_chat_message(prompt)

        if response:
            ss.chat_messages.append(
                {"role": "assistant", "content": response}
            )
            with st.chat_message("assistant"):
                st.markdown(response)
