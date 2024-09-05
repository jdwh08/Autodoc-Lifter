#####################################################
### DOCUMENT PROCESSOR [Metadata Adders]
#####################################################
### Jonathan Wang

# ABOUT:
# This creates an app to chat with PDFs.

# This is the Metadata Adders
# Which are classes that add metadata fields to documents.
# This often is used for summaries or keywords.
#####################################################
### TODO Board:
# Seems like this overlaps well with the `metadata extractors` interface from llama_index.
# These are TransformComponents which take a Sequence of Nodes as input, and returns a list of Dicts as output (with the dicts storing metdata for each node).
# We should add a wrapper which adds this metadata to nodes.
# We should also add a wrapper

# https://github.com/run-llama/llama_index/blob/be3bd619ec114d26cf328d12117c033762695b3f/llama-index-core/llama_index/core/extractors/interface.py#L21
# https://github.com/run-llama/llama_index/blob/be3bd619ec114d26cf328d12117c033762695b3f/llama-index-core/llama_index/core/extractors/metadata_extractors.py#L332

#####################################################
### PROGRAM SETTINGS


#####################################################
### PROGRAM IMPORTS
from __future__ import annotations

import logging
import re
from abc import abstractmethod
from typing import Any, List, Optional, TypeVar, Sequence

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.schema import BaseNode, TransformComponent

# Own modules


#####################################################
### CONSTANTS
# ah how beautiful the regex
# handy visualizer and checker: https://www.debuggex.com/, https://www.regexpr.com/
logger = logging.getLogger(__name__)
GenericNode = TypeVar("GenericNode", bound=BaseNode)

DATE_REGEX = re.compile(r"(?:(?<!\:)(?<!\:\d)[0-3]?\d(?:st|nd|rd|th)?\s+(?:of\s+)?(?:jan\.?|january|feb\.?|february|mar\.?|march|apr\.?|april|may|jun\.?|june|jul\.?|july|aug\.?|august|sep\.?|september|oct\.?|october|nov\.?|november|dec\.?|december)|(?:jan\.?|january|feb\.?|february|mar\.?|march|apr\.?|april|may|jun\.?|june|jul\.?|july|aug\.?|august|sep\.?|september|oct\.?|october|nov\.?|november|dec\.?|december)\s+(?<!\:)(?<!\:\d)[0-3]?\d(?:st|nd|rd|th)?)(?:\,)?\s*(?:\d{4})?|[0-3]?\d[-\./][0-3]?\d[-\./]\d{2,4}", re.IGNORECASE)
TIME_REGEX = re.compile(r"\d{1,2}:\d{2} ?(?:[ap]\.?m\.?)?|\d[ap]\.?m\.?", re.IGNORECASE)
EMAIL_REGEX = re.compile(r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)")
PHONE_REGEX = re.compile(r"((?:(?<![\d-])(?:\+?\d{1,3}[-.\s*]?)?(?:\(?\d{3}\)?[-.\s*]?)?\d{3}[-.\s*]?\d{4}(?![\d-]))|(?:(?<![\d-])(?:(?:\(\+?\d{2}\))|(?:\+?\d{2}))\s*\d{2}\s*\d{3}\s*\d{4}(?![\d-])))")
MAIL_ADDR_REGEX = re.compile(r"\d{1,4}.{1,10}[\w\s]{1,20}[\s]+(?:street|st|avenue|ave|road|rd|highway|hwy|square|sq|trail|trl|drive|dr|court|ct|parkway|pkwy|circle|cir|boulevard|blvd)\W?(?=\s|$)", re.IGNORECASE)

# DEFAULT_NUM_WORKERS = os.cpu_count() - 1 if os.cpu_count() else 1  # type: ignore


#####################################################
### SCRIPT

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
        self, metadata_name: str = "metadata", **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.metadata_name = metadata_name
        # self.num_workers = num_workers

    @classmethod
    def class_name(cls) -> str:
        return "MetadataAdder"

    @abstractmethod
    def get_node_metadata(self, node: BaseNode) -> str | None:
        """Given a node, get the metadata for the node."""

    def add_node_metadata(self, node: GenericNode, metadata_value: Any | None) -> GenericNode:
        """Given a node and the metadata, add the metadata to the node's `metadata_name` field."""
        if (metadata_value is None):
            return node
        else:
            node.metadata[self.metadata_name] = metadata_value
        return node

    def process_nodes(self, nodes: list[GenericNode]) -> list[GenericNode]:
        """Process the list of nodes. This gets called by __call__.

        Args:
            nodes (List[GenericNode]): The nodes to process.

        Returns:
            List[GenericNode]: The processed nodes, with metadata field metadata_name added.
        """
        output_nodes = []
        for node in nodes:
            node_metadata = self.get_node_metadata(node)
            node_with_metadata = self.add_node_metadata(node, node_metadata)
            output_nodes.append(node_with_metadata)
        return(output_nodes)

    def __call__(self, nodes: Sequence[BaseNode], **kwargs: Any) -> list[BaseNode]:
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

    def get_node_metadata(self, node: BaseNode) -> str | None:
        """Given a node with text, return the regex match if it exists.

        Args:
            node (BaseNode): The base node to extract from.

        Returns:
            Optional[str]: The regex match if it exists. If not, return None.
        """
        if (getattr(node, "text", None) is None):
            return None

        if (self._boolean_mode):
            return str(self._regex_pattern.match(node.text) is not None)
        else:
            return str(self._regex_pattern.findall(node.text))  # NOTE: we are saving these as a string'd list since this is easier


class ModelMetadataAdder(MetadataAdder):
    """Adds metadata to nodes based on a language model."""

    prompt_template: str = Field(
        description="The prompt to use to generate the metadata. Defaults to DEFAULT_SUMMARY_TEMPLATE.",
    )

    def __init__(
        self,
        metadata_name: str,
        prompt_template: str | None = None,
        **kwargs: Any
    ) -> None:
        """Init params."""
        super().__init__(metadata_name=metadata_name, prompt_template=prompt_template, **kwargs)

    @classmethod
    def class_name(cls) -> str:
        return "ModelMetadataAdder"

    @abstractmethod
    def get_node_metadata(self, node: BaseNode) -> str | None:
        """Given a node, get the metadata for the node.

        Args:
            node (BaseNode): The node to add metadata to.

        Returns:
            Optional[str]: The metadata if it exists. If not, return None.
        """


class UnstructuredPDFPostProcessor(TransformComponent):
    """Handles postprocessing of PDF which was read in using UnstructuredIO."""

    ### NOTE: okay technically we could have done this in the IngestionPipeline abstraction. Maybe we integrate in the future?
    # This component doesn't play nice with multi-processing due to having non-async LLMs.

    # _embed_model: Optional[BaseEmbedding] = PrivateAttr()
    _metadata_adders: list[MetadataAdder] = PrivateAttr()

    def __init__(
        self,
        # embed_model: Optional[BaseEmbedding] = None,
        metadata_adders: list[MetadataAdder] | None = None,
        **kwargs: Any,
    ) -> None:
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

    def _apply_metadata_adders(self, nodes: list[GenericNode]) -> list[GenericNode]:
        for metadata_adder in self._metadata_adders:
            nodes = metadata_adder(nodes)
        return nodes
    
    def __call__(self, nodes: list[GenericNode], **kwargs: Any) -> Sequence[BaseNode]:
        return self._apply_metadata_adders(nodes)
        # nodes = self._apply_embed_model(nodes)  # this goes second in case we want to embed the metadata.

# def has_email(input_text: str) -> bool:
#     """
#     Given a chunk of text, determine whether it has an email address or not.

#     We're using the long complex email regex from https://emailregex.com/index.html
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