#####################################################
### DOCUMENT PROCESSOR [FULLDOC]
#####################################################
### Jonathan Wang

# ABOUT:
# This creates an app to chat with PDFs.

# This is the FULLDOC
# which is a class that associates documents
# with their critical information
# and their tools. (keywords, summary, queryengine, etc.)
#####################################################
### TODO Board:
# Automatically determine which reader to use for each document based on the file type.

#####################################################
### PROGRAM SETTINGS

#####################################################
### PROGRAM IMPORTS
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, TypeVar
from uuid import UUID, uuid4

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.schema import BaseNode, TransformComponent
from llama_index.core.settings import Settings
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from streamlit import session_state as ss

if TYPE_CHECKING:
    from llama_index.core.base.base_query_engine import BaseQueryEngine
    from llama_index.core.callbacks import CallbackManager
    from llama_index.core.node_parser import NodeParser
    from llama_index.core.readers.base import BaseReader
    from llama_index.core.response_synthesizers import BaseSynthesizer
    from llama_index.core.retrievers import BaseRetriever

# Own Modules
from engine import get_engine
from keywords import KeywordMetadataAdder
from retriever import get_retriever
from storage import get_docstore, get_vector_store
from summary import DEFAULT_ONELINE_SUMMARY_TEMPLATE, DEFAULT_TREE_SUMMARY_TEMPLATE

#####################################################
### SCRIPT

GenericNode = TypeVar("GenericNode", bound=BaseNode)

class FullDocument:
    """Bundles all the information about a document together.

    Args:
        name (str): The name of the document.
        file_path (Path): The path to the document.
        summary (str): The summary of the document.
        keywords (List[str]): The keywords of the document.
        entities (List[str]): The entities of the document.
        vector_store (BaseDocumentStore): The vector store of the document.
    """

    # Identifiers
    id: UUID
    name: str
    file_path: Path
    file_name: str

    # Basic Contents
    summary: str
    summary_oneline: str  # A one line summary of the document.
    keywords: set[str]  # List of keywords in document.
    # entities: Set[str]  # list of entities in document  ## TODO: Add entities
    metadata: dict[str, Any] | None
    # NOTE: other metdata that might be useful:
        # Document Creation / Last Date (e.g., recency important for legal/medical questions)
        # Document Source and Trustworthiness
        # Document Access Level (though this isn't important for us here.)
        # Document Citations?
        # Document Format? (text/spreadsheet/presentation/image/etc.)

    # RAG Components
    nodes: list[BaseNode]
    storage_context: StorageContext  # NOTE: current setup has single storage context per document.
    vector_store_index: VectorStoreIndex
    retriever: BaseRetriever  # TODO(Jonathan Wang): Consider multiple retrievers for keywords vs semantic.
    engine: BaseQueryEngine  # TODO(Jonathan Wang): Consider mulitple engines. 
    subquestion_engine: SubQuestionQueryEngine

    def __init__(
        self,
        name: str,
        file_path: Path | str,
        metadata: dict[str, Any] | None = None
    ) -> None:
        self.id = uuid4()
        self.name = name

        if (isinstance(file_path, str)):
            file_path = Path(file_path)
        self.file_path = file_path
        self.file_name = file_path.name

        self.metadata = metadata


    @classmethod
    def class_name(cls) -> str:
        return "FullDocument"

    def add_name_to_nodes(self, nodes: list[GenericNode]) -> list[GenericNode]:
        """Add the name of the document to the nodes.

        Args:
            nodes (List[GenericNode]): The nodes to add the name to.

        Returns:
            List[GenericNode]: The nodes with the name added.
        """
        for node in nodes:
            node.metadata["name"] = self.name
        return nodes

    def file_to_nodes(
        self,
        reader: BaseReader,
        postreaders: list[Callable[[list[GenericNode]], list[GenericNode]] | TransformComponent] | None=None,  # NOTE: these should be used in order. and probably all TransformComponent instead.
        node_parser: NodeParser | None=None,
        postparsers: list[Callable[[list[GenericNode]], list[GenericNode]] | TransformComponent] | None=None,  # Stuff like chunking, adding Embeddings, etc.
    ) -> None:
        """Read in the file path and get the nodes.

        Args:
            file_path (Optional[Path], optional): The path to the file. Defaults to file_path from init.
            reader (Optional[BaseReader], optional): The reader to use. Defaults to reader from init.
        """
        # Use the provided reader to read in the file.
        print("NEWPDF: Reading input file...")
        nodes = reader.load_data(file_path=self.file_path)

        # Use node postreaders to post process the nodes.
        if (postreaders is not None):
            for node_postreader in postreaders:
                nodes = node_postreader(nodes)  # type: ignore  (TransformComponent allows a list of nodes)

        # Use node parser to parse the nodes.
        if (node_parser is None):
            node_parser = Settings.node_parser
            nodes = node_parser(nodes)  # type: ignore  (Document is a child of BaseNode)

        # Use node postreaders to post process the nodes. (also add the common name to the nodes)
        if (postparsers is None):
            postparsers = [self.add_name_to_nodes]
        else:
            postparsers.append(self.add_name_to_nodes)

        for node_postparser in postparsers:
            nodes = node_postparser(nodes)  # type: ignore  (TransformComponent allows a list of nodes)

        # Save nodes
        self.nodes = nodes  # type: ignore

    def nodes_to_summary(
        self,
        summarizer: BaseSynthesizer,  # NOTE: this is typically going to be a TreeSummarizer / SimpleSummarize for our use case
        query_str: str = DEFAULT_TREE_SUMMARY_TEMPLATE,
    ) -> None:
        """Summarize the nodes.

        Args:
            summarizer (BaseSynthesizer): The summarizer to use. Takes in nodes and returns summary.
        """
        if (not hasattr(self, "nodes")):
            msg = "Nodes must be extracted from document using `file_to_nodes` before calling `nodes_to_summary`."
            raise ValueError(msg)

        text_chunks = [getattr(node, "text", "") for node in self.nodes if hasattr(node, "text")]
        summary_responses = summarizer.aget_response(query_str=query_str, text_chunks=text_chunks)

        loop = asyncio.get_event_loop()
        summary = loop.run_until_complete(summary_responses)

        if (not isinstance(summary, str)):
            # TODO(Jonathan Wang): ... this should always give us a string, right? we're not doing anything fancy with TokenGen/TokenAsyncGen/Pydantic BaseModel...
            msg = f"Summarizer must return a string summary. Actual type: {type(summary)}, with value {summary}."
            raise TypeError(msg)

        self.summary = summary

    def summary_to_oneline(
        self,
        summarizer: BaseSynthesizer,  # NOTE: this is typically going to be a SimpleSummarize / TreeSummarizer for our use case
        query_str: str = DEFAULT_ONELINE_SUMMARY_TEMPLATE,
    ) -> None:

        if (not hasattr(self, "summary")):
            msg = "Summary must be extracted from document using `nodes_to_summary` before calling `summary_to_oneline`."
            raise ValueError(msg)

        oneline = summarizer.get_response(query_str=query_str, text_chunks=[self.summary])  # There's only one chunk.
        self.summary_oneline = oneline  # type: ignore | shouldn't have fancy TokenGenerators / TokenAsyncGenerators / Pydantic BaseModels

    def nodes_to_document_keywords(self, keyword_extractor: Optional[KeywordMetadataAdder] = None) -> None:
        """Save the keywords from the nodes into the document.

        Args:
            keyword_extractor (Optional[BaseKeywordExtractor], optional): The keyword extractor to use. Defaults to None.
        """
        if (not hasattr(self, "nodes")):
            msg = "Nodes must be extracted from document using `file_to_nodes` before calling `nodes_to_keywords`."
            raise ValueError(msg)

        if (keyword_extractor is None):
            keyword_extractor = KeywordMetadataAdder()

        # Add keywords to nodes using KeywordMetadataAdder
        keyword_extractor.process_nodes(self.nodes)

        # Save keywords
        keywords: list[str] = []
        for node in self.nodes:
            node_keywords = node.metadata.get("keyword_metadata", "").split(", ")  # NOTE: KeywordMetadataAdder concatinates b/c required string output
            keywords = keywords + node_keywords

        # TODO(Jonathan Wang): handle dedupling keywords which are similar to each other (fuzzy?)
        self.keywords = set(keywords)

    def nodes_to_storage(self, create_new_storage: bool = True) -> None:
        """Save the nodes to storage."""
        if (not hasattr(self, "nodes")):
            msg = "Nodes must be extracted from document using `file_to_nodes` before calling `nodes_to_storage`."
            raise ValueError(msg)

        if (create_new_storage):
            docstore = get_docstore(documents=self.nodes)
            self.docstore = docstore

            vector_store = get_vector_store()

            storage_context = StorageContext.from_defaults(
                docstore=docstore,
                vector_store=vector_store
            )
            self.storage_context = storage_context

            vector_store_index = VectorStoreIndex(
                self.nodes, storage_context=storage_context
            )
            self.vector_store_index = vector_store_index

        else:
            ### TODO(Jonathan Wang): use an existing storage instead of creating a new one.
            msg = "Currently creates new storage for every document."
            raise NotImplementedError(msg)

    # TODO(Jonathan Wang): Create multiple different retrievers based on the question type(?)
    # E.g., if the question is focused on specific keywords or phrases, use a retriever oriented towards sparse scores.
    def storage_to_retriever(
        self,
        semantic_nodes: int = 6,
        sparse_nodes: int = 3,
        fusion_nodes: int = 3,
        semantic_weight: float = 0.6,
        merge_up_thresh: float = 0.5,
        callback_manager: CallbackManager | None=None
    ) -> None:
        """Create retriever from storage."""
        if (not hasattr(self, "vector_store_index")):
            msg = "Vector store must be extracted from document using `nodes_to_storage` before calling `storage_to_retriever`."
            raise ValueError(msg)

        retriever = get_retriever(
            _vector_store_index=self.vector_store_index,
            semantic_top_k=semantic_nodes,
            sparse_top_k=sparse_nodes,
            fusion_similarity_top_k=fusion_nodes,
            semantic_weight_fraction=semantic_weight,
            merge_up_thresh=merge_up_thresh,
            verbose=True,
            _callback_manager=callback_manager or ss.callback_manager
        )
        self.retriever = retriever

    def retriever_to_engine(
        self,
        response_synthesizer: BaseSynthesizer,
        callback_manager: CallbackManager | None=None
    ) -> None:
        """Create query engine from retriever."""
        if (not hasattr(self, "retriever")):
            msg = "Retriever must be extracted from document using `storage_to_retriever` before calling `retriver_to_engine`."
            raise ValueError(msg)

        engine = get_engine(
            retriever=self.retriever,
            response_synthesizer=response_synthesizer,
            callback_manager=callback_manager or ss.callback_manager
        )
        self.engine = engine

    # TODO(Jonathan Wang): Create Summarization Index and Engine.
    def engine_to_sub_question_engine(self) -> None:
        """Convert a basic query engine into a sub-question query engine for handling complex, multi-step questions.

        Args:
            query_engine (BaseQueryEngine): The Base Query Engine to convert.
        """
        if (not hasattr(self, "summary_oneline")):
            msg = "One Line Summary must be created for the document before calling `engine_to_sub_query_engine`"
            raise ValueError(msg)
        elif (not hasattr(self, "engine")):
            msg = "Basic Query Engine must be created before calling `engine_to_sub_query_engine`"
            raise ValueError(msg)

        sqe_tools = [
            QueryEngineTool(
                query_engine=self.engine,  # TODO(Jonathan Wang): handle mulitple engines?
                metadata=ToolMetadata(
                    name=(self.name + "simple query answerer"),
                    description=f"""A tool that answers simple questions about the following document: {self.summary_oneline}"""
                )
            )
            # TODO(Jonathan Wang): add more tools
        ]

        subquestion_engine = SubQuestionQueryEngine.from_defaults(
            query_engine_tools=sqe_tools, 
            verbose=True,
            use_async=True
        )
        self.subquestion_engine = subquestion_engine
