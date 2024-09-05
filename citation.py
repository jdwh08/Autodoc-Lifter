#####################################################
### DOCUMENT PROCESSOR [CITATION]
#####################################################
# Jonathan Wang

# ABOUT: 
# This project creates an app to chat with PDFs.

# This is the CITATION
# which adds citation information to the LLM response
#####################################################
## TODO Board:
# Investigate using LLM model weights with attention to determien citations.

# https://gradientscience.org/contextcite/
# https://github.com/MadryLab/context-cite/blob/main/context_cite/context_citer.py#L25
# https://github.com/MadryLab/context-cite/blob/main/context_cite/context_partitioner.py
# https://github.com/MadryLab/context-cite/blob/main/context_cite/solver.py

#####################################################
## IMPORTS
from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING
import warnings

import numpy as np
from llama_index.core.base.response.schema import RESPONSE_TYPE, Response

if TYPE_CHECKING:
    from llama_index.core.schema import NodeWithScore

# Own Modules
from merger import _merge_on_scores
from rapidfuzz import fuzz, process, utils


# Lazy Loading:
# from nltk import sent_tokenize  # noqa: ERA001

#####################################################
## CODE

class CitationBuilder:
    """Class that builds citations from responses."""

    text_splitter: Callable[[str], list[str]]

    def __init__(self, text_splitter: Callable[[str], list[str]] | None = None) -> None:
        if not text_splitter:
            from nltk import sent_tokenize
            text_splitter = sent_tokenize
        self.text_splitter = text_splitter

    @classmethod
    def class_name(cls) -> str:
        return "CitationBuilder"

    def convert_to_response(self, input_response: RESPONSE_TYPE) -> Response:
        # Convert all other response types into the baseline response
        # Otherwise, we won't have the full response text generated.
        if not isinstance(input_response, Response):
            response = input_response.get_response()
            if isinstance(response, Response):
                return response
            else:
                # TODO(Jonathan Wang): Handle async responses with Coroutines
                msg = "Expected Response object, got Coroutine"
                raise TypeError(msg)
        else:
            return input_response

    def find_nearest_whitespace(
        self,
        input_text: str,
        input_index: int,
        right_to_left: bool=False
    ) -> int:
        """Given a sting and an index, find the index of whitespace closest to the string."""
        if (input_index < 0  or input_index >= len(input_text)):
            msg = "find_nearest_whitespace: index beyond string."
            raise ValueError(msg)

        find_text = ""
        if (right_to_left):
            find_text = input_text[:input_index]
            for index, char in enumerate(reversed(find_text)):
                if (char.isspace()):
                    return (len(find_text)-1 - index)
            return (0)
        else:
            find_text = input_text[input_index:]
            for index, char in enumerate(find_text):
                if (char.isspace()):
                    return (input_index + index)
            return (len(input_text))

    def get_citations(
        self,
        input_response: RESPONSE_TYPE,
        citation_threshold: int = 70,
        citation_len: int = 128
    ) -> Response:
        response = self.convert_to_response(input_response)

        if not response.response or not response.source_nodes:
            return response

        # Get current response text:
        response_text = response.response
        source_nodes = response.source_nodes

        # 0. Get candidate nodes for citation.
        # Fuzzy match each source node text against the respone text.
        source_texts: dict[str, list[NodeWithScore]] = defaultdict(list)
        for node in source_nodes:
            if (
                (len(getattr(node.node, "text", "")) > 0) and
                (len(node.node.metadata) > 0)
            ):  # filter out non-text nodes and intermediate nodes from SubQueryQuestionEngine
                source_texts[node.node.text].append(node)  # type: ignore

        fuzzy_matches = process.extract(
            response_text,
            list(source_texts.keys()),
            scorer=fuzz.partial_ratio,
            processor=utils.default_process,
            score_cutoff=max(10, citation_threshold - 10)
        )

        # Convert extracted matches of form (Match, Score, Rank) into scores for all source_texts.
        if fuzzy_matches:
            fuzzy_texts, _, _ = zip(*fuzzy_matches)
            fuzzy_nodes = [source_texts[text][0] for text in fuzzy_texts]
        else:
            return response

        # 1. Combine fuzzy score and source text semantic/reranker score.
        # NOTE: for our merge here, we value the nodes with strong fuzzy text matching over other node types.
        cited_nodes = _merge_on_scores(
            a_list=fuzzy_nodes,
            b_list=source_nodes,  # same nodes, different scores (fuzzy vs semantic/bm25/reranker)
            a_scores_input=[getattr(node, "score", np.nan) for node in fuzzy_nodes],
            b_scores_input=[getattr(node, "score", np.nan) for node in source_nodes],
            a_weight=0.85,  # we want to heavily prioritize the fuzzy text for matches
            top_k=3  # maximum of three source options.
        )

        # 2. Add cited nodes text to the response text, and cited nodes as metadata.
        # For each sentence in the response, if there is a match in the source text, add a citation tag.
        response_sentences = self.text_splitter(response_text)
        output_text = ""
        output_citations = ""
        citation_tag = 0

        for response_sentence in response_sentences:
            # Get fuzzy citation at sentence level
            best_alignment = None
            best_score = 0
            best_node = None

            for _, source_node in enumerate(source_nodes):
                source_node_text = getattr(source_node.node, "text", "")
                new_alignment = fuzz.partial_ratio_alignment(
                    response_sentence,
                    source_node_text,
                    processor=utils.default_process, score_cutoff=citation_threshold
                )
                new_score = 0.0

                if (new_alignment is not None and (new_alignment.src_end - new_alignment.src_start) > 0):
                    new_score = fuzz.ratio(
                        source_node_text[new_alignment.src_start:new_alignment.src_end],
                        response_sentence[new_alignment.dest_start:new_alignment.dest_end],
                        processor=utils.default_process
                    )
                    new_score = new_score * (new_alignment.src_end - new_alignment.src_start) / float(len(response_sentence))

                    if (new_score > best_score):
                        best_alignment = new_alignment
                        best_score = new_score
                        best_node = source_node

            if (best_score <= 0 or best_node is None or best_alignment is None):
                # No match
                output_text += response_sentence
                continue

            # Add citation tag to text
            citation_tag_position = self.find_nearest_whitespace(response_sentence, best_alignment.dest_start, right_to_left=True)
            output_text += response_sentence[:citation_tag_position]  # response up to the quote
            output_text += f" [{citation_tag}] "  # add citation tag
            output_text += response_sentence[citation_tag_position:]  # reposnse after the quote

            # Add citation text to citations
            citation = getattr(best_node.node, "text", "")
            citation_margin = round((citation_len - (best_alignment.src_end - best_alignment.src_start)) / 2)
            nearest_whitespace_pre = self.find_nearest_whitespace(citation, max(0, best_alignment.src_start), right_to_left=True)
            nearest_whitespace_post = self.find_nearest_whitespace(citation, min(len(citation)-1, best_alignment.src_end), right_to_left=False)
            nearest_whitespace_prewindow = self.find_nearest_whitespace(citation, max(0, nearest_whitespace_pre - citation_margin), right_to_left=True)
            nearest_whitespace_postwindow = self.find_nearest_whitespace(citation, min(len(citation)-1, nearest_whitespace_post + citation_margin), right_to_left=False)

            citation_text = (
                citation[nearest_whitespace_prewindow+1: nearest_whitespace_pre+1]
                + "|||||"
                + citation[nearest_whitespace_pre+1:nearest_whitespace_post]
                + "|||||"
                + citation[nearest_whitespace_post:nearest_whitespace_postwindow]
                + f"… <<{best_node.node.metadata.get('name', '')}, Page(s) {best_node.node.metadata.get('page_number', '')}>>"
            )
            output_citations += f"[{citation_tag}]: {citation_text}\n\n"
            citation_tag += 1

        # Create output
        if response.metadata is not None:
            # NOTE: metadata is certainly existant by now, but the schema allows None...
            response.metadata["cited_nodes"] = cited_nodes
            response.metadata["citations"] = output_citations
        response.response = output_text  # update response to include citation tags
        return response

    def add_citations_to_response(self, input_response: Response) -> Response:
        if not hasattr(input_response, "metadata"):
            msg = "Input response does not have metadata."
            raise ValueError(msg)
        elif input_response.metadata is None or "citations" not in input_response.metadata:
            warnings.warn("Input response does not have citations.", stacklevel=2)
            input_response = self.get_citations(input_response)

        # Add citation text to response
        if (hasattr(input_response, "metadata") and input_response.metadata.get("citations", "") != ""):
            input_response.response = (
                input_response.response
                + "\n\n----- CITATIONS -----\n\n"
                + input_response.metadata.get('citations', "")
            )  # type: ignore
        return input_response

    def __call__(self, input_response: RESPONSE_TYPE, *args: Any, **kwds: Any) -> Response:
        return self.get_citations(input_response, *args, **kwds)


def get_citation_builder() -> CitationBuilder:
    return CitationBuilder()