#####################################################
### DOCUMENT PROCESSOR [PARSERS]
#####################################################
# Jonathan Wang

# ABOUT:
# This project creates an app to chat with PDFs.

# This is the PARSERS.
# It chunks Raw Text into LlamaIndex nodes
# E.g., by embedding meaning, by sentence, ...
#####################################################
# TODO Board:
# Add more stuff

#####################################################
## IMPORTS
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List, Optional

from llama_index.core import Settings
from llama_index.core.node_parser import (
    SemanticSplitterNodeParser,
    SentenceWindowNodeParser,
)

if TYPE_CHECKING:
    from llama_index.core.base.embeddings.base import BaseEmbedding
    from llama_index.core.callbacks import CallbackManager
    from llama_index.core.node_parser.interface import NodeParser

# from wtpsplit import SaT

# Lazy Loading

#####################################################
## CODE
# def sentence_splitter_from_SaT(model: Optional[SaT]) -> Callable[[str], List[str]]:
#     """Convert a SaT model into a sentence splitter function.

#     Args:
#         model (SaT): The Segment Anything model.

#     Returns:
#         Callable[[str], List[str]]: The sentence splitting function using the SaT model.
#     """
#     model = model or ss.model
#     if model is None:
#         raise ValueError("Sentence splitting model is not set.")

#     def sentence_splitter(text: str) -> List[str]:
#         segments = model.split(text_or_texts=text)
#         if isinstance(segments, list):
#             return segments
#         else:
#             return list(segments)  # type: ignore (generator is the other option?)

#     return (sentence_splitter)

# @st.cache_resource  # can't cache because embed_model is not hashable.
def get_parser(
        embed_model: BaseEmbedding,
        # sentence_model: Optional[SaT] = None,
        sentence_splitter: Optional[Callable[[str], List[str]]] = None,
        callback_manager: Optional[CallbackManager] = None
    ) -> NodeParser:
    """Parse RAG document processing (main one)."""
    # if (sentence_model is not None) and (sentence_splitter is not None):
        # sentence_splitter = sentence_splitter_from_SaT(sentence_model)

    return SemanticSplitterNodeParser.from_defaults(
        embed_model=embed_model,
        breakpoint_percentile_threshold=95,
        buffer_size=3,
        sentence_splitter=sentence_splitter,
        callback_manager=callback_manager or Settings.callback_manager,
        include_metadata=True,
        include_prev_next_rel=True,
    )


# @st.cache_resource
# def get_sentence_parser(splitter_model: Optional[SaT] = None) -> SentenceWindowNodeParser:
#     """Special sentence-level parser to get the document requested info section."""
#     if (splitter_model is not None):
#         sentence_splitter = sentence_splitter_from_SaT(splitter_model)

#     sentence_parser = SentenceWindowNodeParser.from_defaults(
#         sentence_splitter=sentence_splitter,
#         window_size=0,
#         window_metadata_key="window",
#         original_text_metadata_key="original_text",
#     )
#     return (sentence_parser)

def get_sentence_parser() -> SentenceWindowNodeParser:
    """Parse sentences to get the document requested info section."""
    # if (splitter_model is not None):
    #     sentence_splitter = sentence_splitter_from_SaT(splitter_model)
    return SentenceWindowNodeParser.from_defaults(
        # sentence_splitter=sentence_splitter,
        window_size=0,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
