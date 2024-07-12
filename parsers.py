#####################################################
# LIFT DOCUMENT INTAKE PROCESSOR [PARSERS]
#####################################################
# Jonathan Wang (jwang15, jonathan.wang@discover.com)
# Greenhouse GenAI Modeling Team

# ABOUT: 
# This project automates the processing of legal documents 
# requesting information about our customers.
# These are handled by LIFT
# (Legal, Investigation, and Fraud Team)

# This is the PARSERS.
# It chunks Raw Text into LlamaIndex nodes
# E.g., by embedding meaning, by sentence, ...
#####################################################
# TODO Board:
# Add more stuff

#####################################################
## IMPORTS
from typing import Optional, Callable, List, Any

import streamlit as st
from streamlit import session_state as ss

from wtpsplit import SaT

from llama_index.core import Settings
from llama_index.core.node_parser.interface import NodeParser

from llama_index.core.callbacks import CallbackManager

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SemanticSplitterNodeParser, SentenceWindowNodeParser

# Lazy Loading

#####################################################
## CODE
def sentence_splitter_from_SaT(model: Optional[SaT]) -> Callable[[str], List[str]]:
    """Convert a SaT model into a sentence splitter function.

    Args:
        model (SaT): The Segment Anything model.

    Returns:
        Callable[[str], List[str]]: The sentence splitting function using the SaT model.
    """
    model = model or ss.model
    if model is None:
        raise ValueError("Sentence splitting model is not set.")
    
    def sentence_splitter(text: str) -> List[str]:
        segments = model.split(text_or_texts=text)
        if isinstance(segments, list):
            return segments
        else:
            return list(segments)  # type: ignore (generator is the other option?)
    
    return (sentence_splitter)

# @st.cache_resource  # can't cache because embed_model is not hashable.
def get_parser(
        embed_model: BaseEmbedding,
        sentence_model: Optional[SaT] = None,
        sentence_splitter: Optional[Callable[[str], List[str]]] = None,
        callback_manager: Optional[CallbackManager] = None
    ) -> NodeParser:
    """Main parser to use throughout the RAG document processing."""
    if (sentence_model is not None) and (sentence_splitter is not None):
        sentence_splitter = sentence_splitter_from_SaT(sentence_model)
    
    parser = SemanticSplitterNodeParser.from_defaults(
        embed_model=embed_model,
        breakpoint_percentile_threshold=93,
        buffer_size=3,
        sentence_splitter=sentence_splitter,
        callback_manager=callback_manager or Settings.callback_manager,
    )
    return (parser)


# @st.cache_resource
# def get_sentence_parser() -> SentenceWindowNodeParser:
#     """Special sentence-level parser to get the document requested info section."""
#     sentence_parser = SentenceWindowNodeParser.from_defaults(
#         # sentence_splitter=split_by_sentence_newline_tokenizer,
#         window_size=1,
#         window_metadata_key="window",
#         original_text_metadata_key="original_text",
#     )
#     return (sentence_parser)