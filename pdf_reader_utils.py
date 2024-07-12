#####################################################
# LIFT DOCUMENT INTAKE PROCESSOR [PDF READER UTILS]
#####################################################
# Jonathan Wang (jwang15, jonathan.wang@discover.com)
# Greenhouse GenAI Modeling Team

# ABOUT: 
# This project automates the processing of legal documents 
# requesting information about our customers.
# These are handled by LIFT
# (Legal, Investigation, and Fraud Team)

# This is the PDF READER UTILITIES.
# It defines helper functions for the PDF reader,
# such as getting Keywords or finding Contact Info.
#####################################################
### TODO Board:
# Better Summarizer than T5, which has been stripped out?
# Better keywords than the RAKE+YAKE fusion we're currently using?
# Consider using GPE/GSP tagging with spacy to confirm mailing addresses?

# Handle FigureCaption somehow.
# Skip Header if it has a Page X or other page number construction.

#####################################################
### Imports
import logging

from abc import ABC, abstractmethod
from typing import List, Any, Sequence, List, Dict, Optional, Callable

import os
import re, regex
from copy import deepcopy

import asyncio

# Keywords
from multi_rake import Rake
import yake
from merger import _merge_on_scores

# Unstructured Document Parsing
from unstructured.partition.pdf import partition_pdf
from unstructured.cleaners.core import clean_extra_whitespace, group_broken_paragraphs #, clean_ordered_bullets, clean_bullets, clean_dashes
from unstructured.chunking.title import chunk_by_title
# Unstructured Element Types
from unstructured.documents import elements, email_elements

# Llamaindex Extactor
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.multi_modal_llms import MultiModalLLM

from llama_index.core.schema import BaseComponent, BaseNode, TextNode, ImageNode, TransformComponent, RelatedNodeInfo, NodeRelationship
from llama_index.core.instrumentation import DispatcherSpanMixin
from llama_index.core.async_utils import run_jobs, run_async_tasks
from llama_index.core.bridge.pydantic import Field, PrivateAttr

from llama_index.core.settings import (
    Settings,
    llm_from_settings_or_context
)

# test
from llama_index.core.indices.prompt_helper import PromptHelper
from llama_index.core.callbacks.base import CallbackManager
# from llama_index.core.prompts.mixin import PromptMixin
# from llama_index.core.instrumentation import DispatcherSpanMixin

#####################################################
### Constants
# ah how beautiful the regex
# handy visualizer and checker: https://www.debuggex.com/, https://www.regexpr.com/
logger = logging.getLogger(__name__)

DATE_REGEX = re.compile(r'(?:(?<!\:)(?<!\:\d)[0-3]?\d(?:st|nd|rd|th)?\s+(?:of\s+)?(?:jan\.?|january|feb\.?|february|mar\.?|march|apr\.?|april|may|jun\.?|june|jul\.?|july|aug\.?|august|sep\.?|september|oct\.?|october|nov\.?|november|dec\.?|december)|(?:jan\.?|january|feb\.?|february|mar\.?|march|apr\.?|april|may|jun\.?|june|jul\.?|july|aug\.?|august|sep\.?|september|oct\.?|october|nov\.?|november|dec\.?|december)\s+(?<!\:)(?<!\:\d)[0-3]?\d(?:st|nd|rd|th)?)(?:\,)?\s*(?:\d{4})?|[0-3]?\d[-\./][0-3]?\d[-\./]\d{2,4}', re.IGNORECASE)
TIME_REGEX = re.compile(r'\d{1,2}:\d{2} ?(?:[ap]\.?m\.?)?|\d[ap]\.?m\.?', re.IGNORECASE)
EMAIL_REGEX = re.compile(r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)")
PHONE_REGEX = re.compile(r'((?:(?<![\d-])(?:\+?\d{1,3}[-.\s*]?)?(?:\(?\d{3}\)?[-.\s*]?)?\d{3}[-.\s*]?\d{4}(?![\d-]))|(?:(?<![\d-])(?:(?:\(\+?\d{2}\))|(?:\+?\d{2}))\s*\d{2}\s*\d{3}\s*\d{4}(?![\d-])))')
MAIL_ADDR_REGEX = re.compile(r'\d{1,4}.{1,10}[\w\s]{1,20}[\s]+(?:street|st|avenue|ave|road|rd|highway|hwy|square|sq|trail|trl|drive|dr|court|ct|parkway|pkwy|circle|cir|boulevard|blvd)\W?(?=\s|$)', re.IGNORECASE)

# DEFAULT_NUM_WORKERS = os.cpu_count() - 1 if os.cpu_count() else 1  # type: ignore
DEFAULT_SUMMARY_TEMPLATE = """You are an expert summarizer of information. You are given some information. Summarize the information, and then provide the key information that can be drawn from it. The information is below:
{context_str}
"""

DEFAULT_TABLE_SUMMARY_TEMPLATE = """You are an expert summarizer of tables. You are given a table or part of a table in HTML format. The table is below:
{context_str}
----------------
Summarize the table, and then provide the key insights that can be drawn directly from the table. If this is not actually an HTML table or part of an HTML table, please do not respond.
"""

DEFAULT_IMAGE_SUMMARY_TEMPLATE = """You are an expert image summarizer. You are given an image. Summarize the image, and then provide the key insights that can be drawn directly from the image, if there are any.
"""

#####################################################
## FUNCTIONS

def clean_pdf_chunk(self, pdf_chunk: elements.Element) -> elements.Element:
        """
        Given a single element of text from a pdf read by Unstructured, clean its text.
        """
        ### TODO: Is it better to have this outside the reader as a TransformComponent? A: No, we'd still need to clean bad characters.
        
        chunk_text = pdf_chunk.text
        if (len(chunk_text) > 0):
            # Clean any control characters which break the language detection for other parts of the reader.
            RE_BAD_CHARS = regex.compile(r"[\p{Cc}\p{Cs}]+")
            chunk_text = RE_BAD_CHARS.sub("", chunk_text)
    
            # Remove PDF citations text
            chunk_text = re.sub("\\(cid:\\d+\\)", "", chunk_text)  # matches (cid:###)
            # Clean whitespace and broken paragraphs
            # chunk_text = clean_extra_whitespace(chunk_text)
            # chunk_text = group_broken_paragraphs(chunk_text)
            # Save cleaned text.
            pdf_chunk.text = chunk_text
    
        return pdf_chunk


def dedupe_title_chunks(pdf_chunks: List[BaseNode]) -> List[BaseNode]:
    """Given a list of chunks, return a list of chunks without any title duplicates.

    Args:
        pdf_chunks (List[BaseNode]): The list of chunks to have titles deduped.

    Returns:
        List[BaseNode]: The deduped list of chunks.
    """
    index = 0
    while (index < len(pdf_chunks)):
        if (
            (pdf_chunks[index].metadata['type'] == 'Title') # is title
            and (index > 0) # is not first chunk
            and (pdf_chunks[index - 1].metadata['type'] == 'Title')  # previous chunk is also title
        ):
            if (getattr(pdf_chunks[index], 'text', None) != getattr(pdf_chunks[index - 1], 'text', '')):
                pdf_chunks[index].text = getattr(pdf_chunks[index - 1], 'text', '') + '\n' + getattr(pdf_chunks[index], 'text', '')

            # NOTE: We'll remove the PRIOR title, since duplicates AND child relationships are built on the CURRENT title.
            pdf_chunks.pop(index - 1)

            # Update relationships to prior prior node if it exists.
            if (index - 2 >= 0):
                pdf_chunks[index - 2].relationships[NodeRelationship.NEXT] = RelatedNodeInfo(
                    node_id=pdf_chunks[index].node_id,
                    metadata={"filename": pdf_chunks[index].metadata['filename']}
                )
                pdf_chunks[index].relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(
                    node_id=pdf_chunks[index - 2].node_id,
                    metadata={"filename": pdf_chunks[index - 2].metadata['filename']}
                )
            # NOTE: there shouldn't be any PARENT/CHILD relationships to the title that we are deleting, so this seems fine.
            # NOTE: don't need to shift index because we removed an element.
        else:
            # We don't care about any situations other than consecutive title chunks.
            index += 1
            pass

    return (pdf_chunks)


def combine_listitem_chunks(pdf_chunks: List[BaseNode]) -> List[BaseNode]:
    """Given a list of chunks, combine any adjacent chunks which are ListItems into one List.

    Args:
        pdf_chunks (List[BaseNode]): The list of chunks to combine.

    Returns:
        List[BaseNode]: The list of chunks with ListItems combined into one List chunk.
    """
    index = 0
    while (index < len(pdf_chunks)):
        if (
            (pdf_chunks[index].metadata['type'] == 'ListItem') # is list item
            and (index > 0) # is not first chunk
            and (pdf_chunks[index - 1].metadata['type'] == 'ListItem')  # previous chunk is also list item
        ):
            # Okay, we have a consecutive list item. Combine into one list.
            # NOTE: We'll remove the PRIOR list item, since duplicates AND child relationships are built on the CURRENT list item.
            # 1. Append prior list item's text to the current list item's text
            pdf_chunks[index].text = getattr(pdf_chunks[index - 1], 'text', '') + '\n' + getattr(pdf_chunks[index], 'text', '')
            # 2. Remove PRIOR list item
            pdf_chunks.pop(index - 1)
            # 3. Replace NEXT relationship from PRIOR list item with the later list item node ID, if prior prior node exists.
            if (index - 2 >= 0):
                pdf_chunks[index - 2].relationships[NodeRelationship.NEXT] = RelatedNodeInfo(
                    node_id=pdf_chunks[index].node_id,
                    metadata={"filename": pdf_chunks[index].metadata['filename']}
                )
            # 4. Replace PREVIOUS relationship from LATER list item with the prior prior node ID, if prior prior node exists.
                pdf_chunks[index].relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(
                    node_id=pdf_chunks[index - 2].node_id,
                    metadata={"filename": pdf_chunks[index - 2].metadata['filename']}
                )
            # NOTE: the PARENT/CHILD relationships should be the same as the previous list item, so this seems fine.
        else:
            # We don't care about any situations other than consecutive list item chunks.
            index += 1
            pass
    return (pdf_chunks)

def remove_header_footer_pagenum(pdf_chunks: List[BaseNode]) -> List[BaseNode]:
    """Given a list of chunks, remove any header or footer type chunks whose text consists only of the current page number.

    Args:
        pdf_chunks (List[BaseNode]): The list of chunks

    Returns:
        List[BaseNode]: The list of chunks with any page number only headers and footers removed
    """
    output_chunks = []
    for chunk in pdf_chunks:
        if (chunk.metadata['type'] not in ['Header', 'Footer', 'PageNumber']):
            output_chunks.append(chunk)
            continue
        
        # chunk_text = getattr(chunk, 'text', '')
        # if (len(chunk_text.replace(' ', '').replace('\n', '')) == 0):
        #     # Do not keep empty headers or footers
        #     continue
        
        # if (re.match(r'-?((p|p.?|p.?p.?|p.?g.?|page|pages)? ?\d+( ? of \d+)? ?)-?', chunk_text, re.IGNORECASE)):
        #     # Strip out the page number.
        #     chunk.text = re.sub(r'-?((p|p.?|p.?p.?|p.?g.?|page|pages)? ?\d+( ? of \d+)? ?)-?', '', chunk_text)
        
        # output_chunks.append(chunk)
        # continue
        
    return (output_chunks)


def chunk_by_header(
    pdf_chunks_in: List[BaseNode],
    combine_text_under_n_chars: int, 
    multipage_sections: bool = True,
# ) -> Tuple[List[BaseNode], List[BaseNode]]:
):
    """Combine chunks together that are part of the same header and have similar meaning.

    Args:
        pdf_chunks (List[BaseNode]): List of chunks to be combined.

    Returns:
        List[BaseNode]: List of combined chunks.
        List[BaseNode]: List of original chunks, with node references updated.
    """
    
    # TODO: Handle semantic chunking between elements within a Header chunk.
    # TODO: Handle splitting element chunks if they are over `max_characters` in length (does this ever really happen?)
    # TODO: Handle relationships between nodes.
    
    pdf_chunks = deepcopy(pdf_chunks_in)
    output = []
    id_to_index = {}
    index = 0
    
    # Pass 1: Combine chunks together that are part of the same title chunk.
    while (index < len(pdf_chunks)):
        chunk = pdf_chunks[index]
        if (chunk.metadata['type'] in ['Header', 'Footer', 'Image', 'Table']):
            # These go immediately into the semantic title chunks and also reset the new node.
            
            # Let's add a newline to distinguish from any other content.
            if (chunk.metadata['type'] in ['Header', 'Footer', 'Table']):
                setattr(chunk, 'text', getattr(chunk, 'text', '') + "\n")
            
            output.append(chunk)
            index += 1
            continue

        # Make a new node if we have a new title (or if we don't have a title).
        elif (
            (chunk.metadata['type'] == 'Title') 
        ):
            # We're good, this node can stay as a TitleChunk.
            chunk.metadata['type'] = 'Composite'
            if (not isinstance(chunk.metadata['page number'], list)):
                chunk.metadata['page number'] = [chunk.metadata['page number']]

            # Let's add a newline to distinguish the title from the content.
            setattr(chunk, 'text', getattr(chunk, 'text', '') + "\n")

            output.append(chunk)
            id_to_index[chunk.id_] = len(output) - 1
            index += 1
            continue
        
        elif (
            (chunk.metadata['parent_id'] is None) 
            and (
                len(getattr(chunk, 'text', '')) > combine_text_under_n_chars  # big enough text section to stand alone
                or (len(id_to_index.keys()) <= 0)  # no prior title
            )
        ):
            # Okay, so either we don't have a title, or it was interrupted by an image / table.
            # This chunk can stay as a TextChunk.
            chunk.metadata['type'] = 'Composite-TextOnly'
            if (not isinstance(chunk.metadata['page number'], list)):
                chunk.metadata['page number'] = [chunk.metadata['page number']]

            output.append(chunk)
            id_to_index[chunk.id_] = len(output) - 1
            index += 1
            continue
        
        elif (chunk.metadata['parent_id'] in id_to_index):
            # This chunk is part of the same title as a prior chunk.
            # Add this text into the prior title node.
            jndex = id_to_index[chunk.metadata['parent_id']]

            if (not isinstance(output[jndex].metadata['page number'], list)):
                output[jndex].metadata['page number'] = [chunk.metadata['page number']]
            
            output[jndex].text = getattr(output[jndex], 'text', '') + ' ' + getattr(chunk, 'text', '')
            output[jndex].metadata['page number'] = list(set(output[jndex].metadata['page number'] + [chunk.metadata['page number']]))
            output[jndex].metadata['languages'] = list(set(output[jndex].metadata['languages'] + chunk.metadata['languages']))
            
            pdf_chunks.remove(chunk)
            # TODO: Update relationships between nodes.
            continue

        else:
            # Add the text to the prior node that isn't a table or image.
            jndex = len(output) - 1
            while (
                (jndex >= 0) 
                and (output[jndex].metadata['type'] in ['Table', 'Image'])
            ):
                # for title_chunk in output:
                    # print(f'{title_chunk.id_}: {title_chunk.metadata['type']}, text: {title_chunk.text}, parent: {title_chunk.metadata['parent_id']}')
                jndex -= 1
            
            if (jndex <= 0):
                raise Exception(f'Prior title chunk not found: {index}, {chunk.metadata['parent_id']}')
            
            # Add this text into the prior title node.
            if (not isinstance(output[jndex].metadata['page number'], list)):
                output[jndex].metadata['page number'] = [chunk.metadata['page number']]
            
            output[jndex].text = getattr(output[jndex], 'text', '') + ' ' + getattr(chunk, 'text', '')
            output[jndex].metadata['page number'] = list(set(output[jndex].metadata['page number'] + [chunk.metadata['page number']]))
            output[jndex].metadata['languages'] = list(set(output[jndex].metadata['languages'] + chunk.metadata['languages']))
            
            pdf_chunks.remove(chunk)
            # TODO: Update relationships between nodes.
            continue
    
    return (output)


def get_keywords(input_text: str, top_k: int=5) -> str:
    """
    Given a string, get its keywords using RAKE+YAKE w/ Distribution Based Fusion.  
    Inputs:  
        input_text (str): the input text to get keywords from  
        top_k (int): the number of keywords to get  

    Returns:  
        str: A list of the keywords, joined into a string.
    """

    # RAKE
    kw_extractor = Rake()
    keywords_rake = kw_extractor.apply(input_text)
    keywords_rake = dict(keywords_rake)

    # YAKE
    kw_extractor = yake.KeywordExtractor(lan="en", dedupLim=0.9, n=3)
    keywords_yake = kw_extractor.extract_keywords(input_text)
    # reorder scores so that higher is better
    keywords_yake = {keyword[0].lower(): (1 - keyword[1]) for keyword in keywords_yake}
    keywords_yake = dict(sorted(keywords_yake.items(), key=lambda x: x[1], reverse=True))

    # Merge RAKE and YAKE based on scores.
    keywords_merged = _merge_on_scores(list(keywords_yake.keys()), list(keywords_rake.keys()), list(keywords_yake.values()), list(keywords_rake.values()), a_weight=0.5, top_k=top_k)

    # return (list(keywords_rake.keys())[:top_k], list(keywords_yake.keys())[:top_k], keywords_merged)
    return ', '.join(keywords_merged)


class MetadataAdder(TransformComponent):
    """Adds metadata to a node.

    Args:
        metadata_name: The name of the metadata to add to the node. Defaults to 'metadata'.
        # num_workers: The number of workers to use for parallel processing. By default, use all available cores minus one. currently WIP.
    """

    metadata_name: str = Field(
        default="metadata",
        description="The name of the metadata field to add to the document. Defaults to 'metadata'.",
    )
    # num_workers: int = Field(
    #     default=DEFAULT_NUM_WORKERS,
    #     description="The number of workers to use for parallel processing. By default, use all available cores minus one.",
    # )

    def __init__(
        self, metadata_name: str = "metadata", **kwargs
    ):
        super().__init__(**kwargs)
        self.metadata_name = metadata_name
        # self.num_workers = num_workers
    
    @classmethod
    def class_name(cls) -> str:
        return "MetadataAdder"

    @abstractmethod
    def get_node_metadata(self, node: BaseNode) -> Optional[str]:
        """Given a node, get the metadata for the node."""

    def add_node_metadata(self, node: BaseNode, metadata_value: Optional[Any]) -> BaseNode:
        """Given a node and the metadata, add the metadata to the node's `metadata_name` field."""
        if (metadata_value is None):
            return node
        else:
            node.metadata[self.metadata_name] = metadata_value
        return node

    def process_nodes(self, nodes: List[BaseNode]) -> List[BaseNode]:
        """Process the list of nodes. This gets called by __call__.

        Args:
            nodes (List[BaseNode]): The nodes to process.

        Returns:
            List[BaseNode]: The processed nodes, with metadata field metadata_name added.
        """
        output_nodes = []
        for node in nodes:
            node_metadata = self.get_node_metadata(node)
            node = self.add_node_metadata(node, node_metadata)
            output_nodes.append(node)
        return(output_nodes)

    def __call__(self, nodes: List[BaseNode], **kwargs: Any) -> List[BaseNode]:
        """Check whether nodes have the specified regex pattern."""
        return self.process_nodes(nodes)


class RegexMetadataAdder(MetadataAdder):
    """Adds regex metadata to a document.

    Args:
        regex_pattern: The regex pattern to search for.
        metadata_name: The name of the metadata to add to the document. Defaults to 'regex_metadata'.
        # num_workers: The number of workers to use for parallel processing. By default, use all available cores minus one.
    """

    _regex_pattern: re.Pattern = PrivateAttr()
    _boolean_mode: bool = PrivateAttr()
    # num_workers: int = Field(
    #     default=DEFAULT_NUM_WORKERS,
    #     description="The number of workers to use for parallel processing. By default, use all available cores minus one.",
    # )

    def __init__(
        self,
        regex_pattern: re.Pattern | str = DATE_REGEX,
        metadata_name: str = "regex_metadata",
        boolean_mode: bool = False,
        # num_workers: int = DEFAULT_NUM_WORKERS,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        if (isinstance(regex_pattern, str)):
            regex_pattern = re.compile(regex_pattern)
        # self.num_workers = num_workers
        super().__init__(metadata_name=metadata_name, **kwargs)  # ah yes i love oop :)
        self._regex_pattern=regex_pattern
        self._boolean_mode=boolean_mode

    @classmethod
    def class_name(cls) -> str:
        return "RegexMetadataAdder"
    
    def get_node_metadata(self, node: BaseNode) -> Optional[str]:
        """Given a node with text, return the regex match if it exists.

        Args:
            node (BaseNode): The base node to extract from.

        Returns:
            Optional[str]: The regex match if it exists. If not, return None.
        """
        if ((not hasattr(node, 'text')) or (node.text is None)):
            return None

        if (self._boolean_mode):
            return str(self._regex_pattern.match(node.text) is not None)
        else:
            return str(self._regex_pattern.findall(node.text))  # NOTE: we are saving these as a string'd list since this is easier


class KeywordMetadataAdder(MetadataAdder):
    """Adds keyword metadata to a document.

    Args:
        metadata_name: The name of the metadata to add to the document. Defaults to 'keyword_metadata'.
        keywords_function: A function for keywords, given a source string and the number of keywords to get.
    """
    keywords_function: Callable[[str, int], str] = Field(
        description="The function to use to extract keywords from the text. Input is string and number of keywords to extract. Ouptut is string of keywords.",
        default=get_keywords,
    )
    num_keywords: int = Field(
        default=5,
        description="The number of keywords to extract from the text. Defaults to 5.",
    )
    
    def __init__(
        self,
        metadata_name: str = "keyword_metadata",
        keywords_function: Callable[[str, int], str] = get_keywords,
        num_keywords: int = 5,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        super().__init__(metadata_name=metadata_name, keywords_function=keywords_function, num_keywords=num_keywords, **kwargs)  # ah yes i love oop :)
        
    @classmethod
    def class_name(cls) -> str:
        return "KeywordMetadataAdder"

    def get_node_metadata(self, node: BaseNode) -> Optional[str]:
        if ((not hasattr(node, 'text')) or (node.text is None)):
            return None
        return(self.keywords_function(node.text, self.num_keywords))


class ModelMetadataAdder(MetadataAdder):
    """Adds metadata to nodes based on a language model."""
    prompt_template: str = Field(
        description="The prompt to use to generate the metadata. Defaults to DEFAULT_SUMMARY_TEMPLATE.",
        default=DEFAULT_SUMMARY_TEMPLATE
    )
    
    def __init__(
        self, 
        metadata_name: str,
        prompt_template: Optional[str] = DEFAULT_SUMMARY_TEMPLATE, 
        **kwargs: Any
    ) -> None:
        """Init params."""
        prompt_template = prompt_template if prompt_template is not None else DEFAULT_SUMMARY_TEMPLATE
        super().__init__(metadata_name=metadata_name, prompt_template=prompt_template, **kwargs)

    @classmethod
    def class_name(cls) -> str:
        return "ModelMetadataAdder"

    @abstractmethod
    def get_node_metadata(self, node: BaseNode) -> Optional[str]:
        """Given a node, get the metadata for the node.

        Args:
            node (BaseNode): The node to add metadata to.

        Returns:
            Optional[str]: The metadata if it exists. If not, return None.
        """


class TextSummaryMetadataAdder(ModelMetadataAdder):
    """Adds metadata to nodes based on a language model."""
    
    _llm: BaseLLM = PrivateAttr()
    
    def __init__(
        self, 
        metadata_name: str,
        llm: Optional[BaseLLM] = None,
        prompt_template: Optional[str] = DEFAULT_SUMMARY_TEMPLATE, 
        **kwargs: Any
    ) -> None:
        """Init params."""
        llm = llm or Settings.llm
        prompt_template = prompt_template if prompt_template is not None else DEFAULT_SUMMARY_TEMPLATE
        super().__init__(metadata_name=metadata_name, prompt_template=prompt_template, **kwargs)

    @classmethod
    def class_name(cls) -> str:
        return "TextSummaryMetadataAdder"

    def get_node_metadata(self, node: BaseNode) -> Optional[str]:
        if (getattr(node, 'text', None) is None):
            return None
        
        response = self._llm.complete(prompt=self.prompt_template.format(context_str=node.text))
        return response.text


class TableSummaryMetadataAdder(ModelMetadataAdder):
    """Adds table summary metadata to a document.

    Args:
        metadata_name: The name of the metadata to add to the document. Defaults to 'table_summary'.
        llm: The LLM to use to generate the table summary. Defaults to Settings llm.
        prompt_template: The prompt template to use to generate the table summary. Defaults to DEFAULT_TABLE_SUMMARY_TEMPLATE.
    """
    
    _llm: BaseLLM = PrivateAttr()
    
    def __init__(
        self,
        metadata_name: str = "table_summary",  ## TODO: This is a bad pattern, string should not be hardcoded like this
        llm: Optional[BaseLLM] = None,
        prompt_template: Optional[str] = DEFAULT_TABLE_SUMMARY_TEMPLATE,
        # num_workers: int = 1,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        llm = llm or Settings.llm
        prompt_template = prompt_template or DEFAULT_TABLE_SUMMARY_TEMPLATE
        super().__init__(metadata_name=metadata_name, prompt_template=prompt_template, **kwargs)
        self._llm = llm
    
    @classmethod
    def class_name(cls) -> str:
        return "TableSummaryMetadataAdder"

    def get_node_metadata(self, node: BaseNode) -> Optional[str]:
        """Given a node, get the metadata for the node using the language model."""
        ## NOTE: Our PDF Reader parser distringuishes between TextNode and TableNode using the 'orignal_table_text' attribute. 
        ## BUG (future): `orignal_table_text` should not be hardcoded. 
        if (not isinstance(node, TextNode)):
            return None
        if (node.metadata.get('orignal_table_text') is None):
            return None
        if (getattr(node, 'text', None) is None):
            return None
        
        response = self._llm.complete(
            self.prompt_template.format(context_str=node.text)
        )
        return response.text


class ImageSummaryMetadataAdder(ModelMetadataAdder):
    """Adds image summary metadata to a document.

    Args:
        metadata_name: The name of the metadata to add to the document. Defaults to 'table_summary_metadata'.
    """
    _llm: MultiModalLLM = PrivateAttr()
    
    def __init__(
        self,
        llm: MultiModalLLM,
        prompt_template: str = DEFAULT_IMAGE_SUMMARY_TEMPLATE,
        metadata_name: str = 'image_summary',
        **kwargs: Any,
    ) -> None:
        """Init params."""
        super().__init__(metadata_name=metadata_name, prompt_template=prompt_template, **kwargs)
        self._llm = llm

    @classmethod
    def class_name(cls) -> str:
        return "ImageSummaryMetadataAdder"
    
    def _get_image_node_metadata(self, node: BaseNode) -> Optional[str]:
        """Handles getting images from image nodes.

        Args:
            node (BaseNode): The image node to get the image summary for. NOTE: This can technically be any type of node so long as it has an image stored.

        Returns:
            Optional[str]: The image summary if it exists. If not, return None.
        """
        if (
            ((getattr(node, 'image', None) is None) and (getattr(node, 'image_path', None) is None))
            or (not callable(getattr(node, "resolve_image", None)))  # method used to convert node to PILImage for model.
        ):
            # Not a valid image node with image attributes and image conversion.
            return None

        ## NOTE: We're assuming that the llm complete function has a parameter `images` to send image node(s) as input.
        ## This is NOT necessarily true if the end user decides to create their own implementation of a MultiModalLLM.
        response = self._llm.complete(
            prompt=self.prompt_template,
            image_documents=[node]
        )
        return response.text
    
    def _get_composite_node_metadata(self, node: BaseNode) -> Optional[str]:
        """Handles getting images from composite nodes (i.e., where an image is stored as a original node inside a composite node).

        Args:
            node (TextNode): The composite node to get the image summary for.

        Returns:
            Optional[str]: The image summary if it exists. If not, return None.
        """
        if ('orig_nodes' not in node.metadata):
            return None  # no image nodes in the composite node.
        
        output = ""
        for orig_node in node.metadata['orig_nodes']:
            output += str(self._get_image_node_metadata(orig_node) or "")
        
        if (output == ""):
            return None
        return output
    
    def get_node_metadata(self, node: BaseNode) -> Optional[str]:
        """Get the image summary for a node (or subnodes)."""
        
        if (node.metadata['type'].startswith('Composite')):
            return self._get_composite_node_metadata(node)
        else:
            return self._get_image_node_metadata(node)


class UnstructuredPDFPostProcessor(TransformComponent):
    """Handles postprocessing of PDF which was read in using UnstructuredIO."""
    ### NOTE: okay technically we could have done this in the IngestionPipeline abstraction, but I want to keep my stuff separate.
    # This component doesn't play nice with multi-processing due to having LLMs.
    
    # _embed_model: Optional[BaseEmbedding] = PrivateAttr()
    _metadata_adders: List[MetadataAdder] = PrivateAttr()
    
    def __init__(
        self, 
        # embed_model: Optional[BaseEmbedding] = None,
        metadata_adders: Optional[List[MetadataAdder]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # self._embed_model = embed_model or Settings.embed_model
        self._metadata_adders = metadata_adders or []

    @classmethod
    def class_name(cls) -> str:
        return "UnstructuredPDFPostProcessor"
    
    # def _apply_embed_model(self, nodes: List[BaseNode]) -> List[BaseNode]:
    #     if (self._embed_model is not None):
    #         nodes = self._embed_model(nodes)
    #     return nodes
    
    def _apply_metadata_adders(self, nodes: List[BaseNode]) -> List[BaseNode]:
        for metadata_adder in self._metadata_adders:
            nodes = metadata_adder(nodes)
        return nodes
    
    def __call__(self, nodes: List[BaseNode], **kwargs: Any) -> List[BaseNode]:
        nodes = self._apply_metadata_adders(nodes)
        # nodes = self._apply_embed_model(nodes)  # this goes second in case we want to embed the metadata.
        return nodes


# def has_email(input_text: str) -> bool:
#     """
#     Given a chunk of text, determine whether it has an email address or not.

#     We're using the long complex email regex from https://emailregex.com/index.html
#     but supposedly for LIFT there is a bijective relationship between @ and email address.
#     i.e., @ only occurs for email, and all emails have @
#     """
#     return (EMAIL_REGEX.search(input_text) is not None)


# def has_phone(input_text: str) -> bool:
#     """
#     Given a chunk of text, determine whether it has a phone number or not.
#     """
#     has_phone = PHONE_REGEX.search(input_text)
#     return (has_phone is not None)


# def has_mail_addr(input_text: str) -> bool:
#     """
#     Given a chunk of text, determine whether it has a mailing address or not.

#     NOTE: This is difficult to do with regex.
#         ... We could use spacy's English language NER model instead / as well:
#         Assume that addresses will have a GSP (geospatial political) or GPE (geopolitical entity).
#         DOCS SEE: https://www.nltk.org/book/ch07.html | https://spacy.io/usage/linguistic-features
#     """
#     has_addr = MAIL_ADDR_REGEX.search(input_text)
#     return (has_addr is not None)


# def has_date(input_text: str) -> bool:
#     """
#     Given a chunk of text, determine whether it has a date or not.
#     NOTE: relative dates are stuff like "within 30 days"
#     """
#     has_date = DATE_REGEX.search(input_text)
#     return (has_date is not None)