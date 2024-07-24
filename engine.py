#####################################################
# LIFT DOCUMENT INTAKE PROCESSOR [ENGINE]
#####################################################
# Jonathan Wang (jwang15, jonathan.wang@discover.com)
# Greenhouse GenAI Modeling Team

# ABOUT: 
# This project automates the processing of legal documents 
# requesting information about our customers.
# These are handled by LIFT
# (Legal, Investigation, and Fraud Team)

# This is the ENGINE
# which defines how LLMs handle processing.
#####################################################
## TODO Board:
# Handle citations from multiple documents

# Move Citation outside of RAGQueryEngine and instead add it after LLM response
    # Allows us to get citations from more advanced engines.

#####################################################
## IMPORTS
import gc
from torch.cuda import empty_cache

from typing import Optional, List, Callable, Dict
from collections import defaultdict
from copy import deepcopy

import numpy as np
from rapidfuzz import process, utils, fuzz

import streamlit as st

from llama_index.core.callbacks import CallbackManager

from llama_index.core.schema import BaseNode, NodeWithScore, QueryBundle
from llama_index.core.base.response.schema import RESPONSE_TYPE, Response
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.llms.llm import LLM
from llama_index.core.response_synthesizers import (
    BaseSynthesizer,
    ResponseMode,
    get_response_synthesizer,
)
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.query_engine import CustomQueryEngine

from llama_index.core.settings import (
    Settings,
    callback_manager_from_settings_or_context,
    llm_from_settings_or_context,
)

from merger import _merge_on_scores

# Lazy Loading:
# from nltk import sent_tokenize

#####################################################
## CODE
class RAGQueryEngine(CustomQueryEngine):
    """Custom RAG Query Engine."""
    retriever: BaseRetriever
    response_synthesizer: BaseSynthesizer
    node_postprocessors: Optional[List[BaseNodePostprocessor]] = []
    citation_sentence_splitter: Callable[[str], List[str]]

    # def __init__(
    #     self,
    #     retriever: BaseRetriever,
    #     response_synthesizer: Optional[BaseSynthesizer] = None,
    #     node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
    #     callback_manager: Optional[CallbackManager] = None,
    # ) -> None:
    #     self._retriever = retriever
    #     # callback_manager = (
    #     #     callback_manager
    #     #     or callback_manager_from_settings_or_context(Settings, service_context)
    #     # )
    #     # llm = llm or llm_from_settings_or_context(Settings, service_context)

    #     self._response_synthesizer = response_synthesizer or get_response_synthesizer(
    #         # llm=llm,
    #         # service_context=service_context,
    #         # callback_manager=callback_manager,
    #     )
    #     self._node_postprocessors = node_postprocessors or []
    #     self._metadata_mode = metadata_mode

    #     for node_postprocessor in self._node_postprocessors:
    #         node_postprocessor.callback_manager = callback_manager

    #     super().__init__(callback_manager=callback_manager)
    
    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "RAGQueryEngine"

    # taken from Llamaindex CustomEngine: 
    # https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/query_engine/retriever_query_engine.py#L134
    def _apply_node_postprocessors(
        self, nodes: List[NodeWithScore], query_bundle: QueryBundle
    ) -> List[NodeWithScore]:
        if self.node_postprocessors is None:
            return nodes

        for node_postprocessor in self.node_postprocessors:
            nodes = node_postprocessor.postprocess_nodes(
                nodes, query_bundle=query_bundle
            )
        return nodes
    
    def _add_citation(
        self,
        input_response: RESPONSE_TYPE,
        text_splitter: Callable[[str], List[str]],
        citation_threshold: int = 70,
        citation_len: int = 256
    ) -> Response:
        # Append the node chunks which have near-exact quotes into the the "citations" page
        # along with their position on the page.
        
        # Convert all other response types into the baseline response
        # Otherwise, we won't have the full response text generated.
        if (not isinstance(input_response, Response)):
            response: Response = input_response.get_response()  # TODO: handling async.
        else:
            response: Response = input_response
        
        # Get current response text:
        if (not response.response or not response.source_nodes):
            # No citation.
            return response

        response_text = response.response
        source_nodes = response.source_nodes
        
        # 0. Fuzzy match each source node text against the respone text.
        source_texts: Dict[str, List[NodeWithScore]] = defaultdict(list)
        for node in source_nodes:
            if getattr(node.node, 'text', '') != '':
                source_texts[getattr(node.node, 'text')].append(node)

        fuzzy_matches = process.extract(response_text, list(source_texts.keys()), scorer=fuzz.partial_ratio, processor=utils.default_process, score_cutoff=min(0, citation_threshold-10))
        
        # Convert extracted matches of form (Match, Score, Rank) into scores for all source_texts.
        if fuzzy_matches:
            fuzzy_texts, fuzzy_scores, _ = zip(*fuzzy_matches)
            fuzzy_nodes: List[NodeWithScore] = []
            for text in fuzzy_texts:
                fuzzy_nodes.append(source_texts[text][0])  # NOTE: Node choice is arbitrary because all have the same text.
        else:
            return response
        
        # 1. Combine fuzzy score and source text semantic/reranker score.
        # NOTE: for our merge here, we value the nodes with strong fuzzy text matching over other node types.
        cited_nodes =_merge_on_scores(
            a_list=fuzzy_nodes, 
            b_list=source_nodes, # same nodes, different scores (fuzzy vs semantic/bm25/reranker)
            a_scores=[float(getattr(node, 'score')) for node in fuzzy_nodes],
            b_scores=[float(getattr(node, 'score')) for node in source_nodes],
            a_weight=0.85,  # we want to heavily prioritize the text
            top_k=3  # maximum of three source options.
        )
        
        # 2. Add cited nodes text to the response text, and cited nodes as metadata.
        # Identify the **sentences** in the response text which have significant overlap with the output.
        response_sentences = text_splitter(response_text)
        source_texts_sentences = [text_splitter(getattr(node.node, 'text', '')) for node in cited_nodes]
        
        output_text = ""
        output_citations = ""
        citation_tag = 0
        
        for response_sentence in response_sentences:
            source_sentence = ""
            max_source_score = 0
            max_source_index = -1

            for source_index, source_sentences in enumerate(source_texts_sentences):
                # Get fuzzy at sentence level

                # NOTE: I assume only one sentence per source should be checked, and only one source is relevant per sentence.
                sentence_fuzzy_matches = process.extract(response_sentence, source_sentences, scorer=fuzz.token_ratio, limit=1, score_cutoff=citation_threshold)  # no processing
                if ((len(sentence_fuzzy_matches) > 0) and (sentence_fuzzy_matches[0][1] > max_source_score)):
                    # We have a better source.
                    source_sentence = sentence_fuzzy_matches[0][0]
                    max_source_score = sentence_fuzzy_matches[0][1]
                    max_source_index = source_index
            
            if (source_sentence == ""):
                output_text += response_sentence
            else:
                # ... yeah we have to hit Rapidfuzz again to get where the sentence is in the source
                source_sentence_alignment = fuzz.partial_ratio_alignment(source_sentence, response_sentence)
                if (source_sentence_alignment is None):
                    Warning(f"""_add_citation: Could not find citation alignment position. 
Source: {source_sentence}
Response: {response_sentence}""")
                    return response

                # Add citation to text
                # 1. Find the index of the last whitespace (space, enter, tab) in the source text before the citation position
                citation_position = max(response_sentence.rfind(' ', 0, source_sentence_alignment.dest_end), response_sentence.rfind('\n', 0, source_sentence_alignment.dest_end), response_sentence.rfind('\t', 0, source_sentence_alignment.dest_end))
                
                output_text += response_sentence[:citation_position+1]  # reposnse up to the quote
                output_text += f" [{citation_tag}] "  # add citation tag
                output_text += response_sentence[citation_position+1:]  # reposnse after the quote

                citation_margin = round((citation_len - (source_sentence_alignment.src_end - source_sentence_alignment.src_start)) / 2)
                citation_text = source_sentence[max(0, source_sentence_alignment.src_start - citation_margin):min(len(source_sentence), source_sentence_alignment.src_end + citation_margin)]
                output_citations += (f"[{citation_tag}]: …{citation_text}… (Page {cited_nodes[max_source_index].metadata.get('page_number', '')})" + "\n\n")   ### TODO: Handle citation from multiple documents by including document name.
                citation_tag += 1
        
        # Add output citations to the end of the text.
        if output_citations:
            output_text += "\n\n--- CITATIONS ---\n\n" + output_citations
        
        # Create output
        output_response = deepcopy(response)
        output_response.metadata['cited_nodes'] = cited_nodes
        output_response.response = output_text  # update response with citations block
        return output_response

    def retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        nodes = self.retriever.retrieve(query_bundle)
        return self._apply_node_postprocessors(nodes, query_bundle=query_bundle)

    async def aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        nodes = await self.retriever.aretrieve(query_bundle)
        return self._apply_node_postprocessors(nodes, query_bundle=query_bundle)

    def custom_query(self, query_str: str):
        # Convert query string into query bundle
        query_bundle = QueryBundle(query_str=query_str)
        nodes = self.retrieve(query_bundle)  # also does the postprocessing.

        response_obj = self.response_synthesizer.synthesize(query_bundle, nodes)
        response_obj = self._add_citation(response_obj, text_splitter=self.citation_sentence_splitter)
        
        empty_cache()
        gc.collect()
        return response_obj


# @st.cache_resource  # none of these can be hashable or cached :(
def get_engine(
    retriever: BaseRetriever,
    response_synthesizer: BaseSynthesizer,
    node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
    citation_sentence_splitter: Optional[Callable[[str], List[str]]] = None,
    callback_manager: Optional[CallbackManager] = None,
) -> RAGQueryEngine:
    if (citation_sentence_splitter is None):
        from nltk import sent_tokenize
    
    engine = RAGQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=node_postprocessors,
        citation_sentence_splitter=citation_sentence_splitter or sent_tokenize,
        callback_manager=callback_manager or Settings.callback_manager,
    )
    return (engine)