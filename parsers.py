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
import streamlit as st

from llama_index.core.node_parser.interface import NodeParser

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SemanticSplitterNodeParser, SentenceWindowNodeParser

# Lazy Loading

#####################################################
## CODE
# @st.cache_resource  # can't cache because embed_model is not hashable.
def get_parser(embed_model: HuggingFaceEmbedding) -> NodeParser:
    """Main parser to use throughout the RAG document processing."""
    semantic_parser = SemanticSplitterNodeParser(
        buffer_size=3,
        breakpoint_percentile_threshold=93,
        embed_model=embed_model,
        # sentence_splitter=_sentence_splitter,
    )
    return (semantic_parser)


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