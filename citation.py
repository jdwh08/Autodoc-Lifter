#####################################################
# LIFT DOCUMENT INTAKE PROCESSOR [CITATION]
#####################################################
# Jonathan Wang (jwang15, jonathan.wang@discover.com)
# Greenhouse GenAI Modeling Team

# ABOUT: 
# This project automates the processing of legal documents 
# requesting information about our customers.
# These are handled by LIFT
# (Legal, Investigation, and Fraud Team)

# This is the CITATION
# which adds citation information to the LLM response
#####################################################
## TODO Board:
# Move Citation outside of RAGQueryEngine and instead add it after LLM response
    # Allows us to get citations from more advanced engines.

#####################################################
## IMPORTS
import os

from typing import Callable, List, Dict, Any, Optional
from collections import defaultdict
from copy import deepcopy

from rapidfuzz import process, utils, fuzz

import numpy as np

from llama_index.core.base.response.schema import RESPONSE_TYPE, Response
from llama_index.core.schema import NodeWithScore

# Own Modules
from merger import _merge_on_scores

# Lazy Loading:
# from nltk import sent_tokenize

#####################################################
## CODE

class CitationBuilder():
    """Class that builds citations from responses."""

    text_splitter: Callable[[str], List[str]]
    
    def __init__(self, text_splitter: Optional[Callable[[str], List[str]]] = None) -> None:
        if not text_splitter:
            from nltk import sent_tokenize
            os.environ['NLTK_DATA'] = './nltk_data'
            text_splitter = sent_tokenize
        self.text_splitter = text_splitter
    
    @classmethod
    def class_name(cls) -> str:
        return "CitationBuilder"

    def convert_to_response(self, input_response: RESPONSE_TYPE) -> Response:
        # Convert all other response types into the baseline response
        # Otherwise, we won't have the full response text generated.
        if (not isinstance(input_response, Response)):
            response: Response = input_response.get_response()  # TODO: handling async.
        else: 
            response = input_response
        return response
    
    def get_citations(
        self, 
        input_response: RESPONSE_TYPE,
        citation_threshold: int = 70,
        citation_len: int = 256
    ) -> Response:
        response = self.convert_to_response(input_response)
        
        if (not response.response or not response.source_nodes):
            return response
        
        # Get current response text:
        response_text = response.response
        source_nodes = response.source_nodes

        # 0. Fuzzy match each source node text against the respone text.
        source_texts: Dict[str, List[NodeWithScore]] = defaultdict(list)
        for node in source_nodes:
            if getattr(node.node, 'text', '') != '':
                source_texts[getattr(node.node, 'text')].append(node)

        fuzzy_matches = process.extract(response_text, list(source_texts.keys()), scorer=fuzz.partial_ratio, processor=utils.default_process, score_cutoff=max(0, citation_threshold-10))
        
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
        cited_nodes = _merge_on_scores(
            a_list=fuzzy_nodes, 
            b_list=source_nodes, # same nodes, different scores (fuzzy vs semantic/bm25/reranker)
            a_scores_input=[getattr(node, 'score', np.nan) for node in fuzzy_nodes],
            b_scores_input=[getattr(node, 'score', np.nan) for node in source_nodes],
            a_weight=0.85,  # we want to heavily prioritize the fuzzy text for matches
            top_k=3  # maximum of three source options.
        )
        
        # 2. Add cited nodes text to the response text, and cited nodes as metadata.
        # Identify the **sentences** in the response text which have significant overlap with the output.
        response_sentences = self.text_splitter(response_text)
        source_texts_sentences = [self.text_splitter(getattr(node.node, 'text', '')) for node in cited_nodes]
        
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
                period_citation_position = response_sentence.lstrip().find('.', source_sentence_alignment.dest_end)
                space_citation_position = response_sentence.lstrip().find(' ', source_sentence_alignment.dest_end)
                newline_citation_position = response_sentence.lstrip().find('\n', source_sentence_alignment.dest_end)
                tab_citation_position = response_sentence.lstrip().find('\t', source_sentence_alignment.dest_end)
                citation_position = min(period_citation_position, space_citation_position, newline_citation_position, tab_citation_position)
                
                output_text += response_sentence[:citation_position+1]  # reposnse up to the quote
                if (citation_position == period_citation_position):
                    output_text += " "  # add a leading space before the citation.
                output_text += f" [{citation_tag}] "  # add citation tag
                output_text += response_sentence[citation_position+1:]  # reposnse after the quote

                citation_margin = round((citation_len - (source_sentence_alignment.src_end - source_sentence_alignment.src_start)) / 2)
                citation_text = source_sentence[max(0, source_sentence_alignment.src_start - citation_margin):min(len(source_sentence), source_sentence_alignment.src_end + citation_margin)]
                output_citations += (f"[{citation_tag}]: …{citation_text}… ({cited_nodes[max_source_index].metadata.get('name', '')} Page {cited_nodes[max_source_index].metadata.get('page_number', '')})" + "\n\n")   ### TODO: Handle citation from multiple documents by including document name.
                citation_tag += 1

        # Create output
        response.metadata['cited_nodes'] = cited_nodes
        response.metadata['citations'] = output_citations
        response.response = output_text  # update response to include citation tags
        return response
    
    def add_citations_to_response(self, input_response: Response) -> Response:
        if not hasattr(input_response, 'metadata'):
            raise ValueError("Input response does not have metadata.")
        elif input_response.metadata is None or 'citations' not in input_response.metadata:
            UserWarning("Input response does not have citations.")
            input_response = self.get_citations(input_response)
        
        # Add citation text to response
        citations = input_response.metadata.get('citations', "")  # type: ignore
        if (citations != ""):
            input_response.response = input_response.response + "\n\n----- CITATIONS -----\n\n"+ input_response.metadata.get('citations', "")  # type: ignore
        return input_response
    
    def __call__(self, input_response: RESPONSE_TYPE, *args: Any, **kwds: Any) -> Response:
        return self.get_citations(input_response, *args, **kwds)


def get_citation_builder() -> CitationBuilder:
    return CitationBuilder()