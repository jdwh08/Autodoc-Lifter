#####################################################
### DOCUMENT PROCESSOR [OBSERVATION/LOGGING]
#####################################################
# Jonathan Wang

# ABOUT: 
# This project creates an app to chat with PDFs.

# This is the Observation and Logging
# to see the actions undertaken in the RAG pipeline.
#####################################################
## TODOS:
# Why does FullRAGEventHandler keep producing duplicate output?

#####################################################
## IMPORTS:
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar, Sequence

import streamlit as st

# Callbacks
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler

# Pretty Printing
# from llama_index.core.response.notebook_utils import display_source_node
# End user handler
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.core.instrumentation.events.agent import (
    AgentChatWithStepEndEvent,
    AgentChatWithStepStartEvent,
    AgentRunStepEndEvent,
    AgentRunStepStartEvent,
    AgentToolCallEvent,
)
from llama_index.core.instrumentation.events.chat_engine import (
    StreamChatDeltaReceivedEvent,
    StreamChatErrorEvent,
)
from llama_index.core.instrumentation.events.embedding import (
    EmbeddingEndEvent,
    EmbeddingStartEvent,
)
from llama_index.core.instrumentation.events.llm import (
    LLMChatEndEvent,
    LLMChatInProgressEvent,
    LLMChatStartEvent,
    LLMCompletionEndEvent,
    LLMCompletionStartEvent,
    LLMPredictEndEvent,
    LLMPredictStartEvent,
    LLMStructuredPredictEndEvent,
    LLMStructuredPredictStartEvent,
)
from llama_index.core.instrumentation.events.query import (
    QueryEndEvent,
    QueryStartEvent,
)
from llama_index.core.instrumentation.events.rerank import (
    ReRankEndEvent,
    ReRankStartEvent,
)
from llama_index.core.instrumentation.events.retrieval import (
    RetrievalEndEvent,
    RetrievalStartEvent,
)
from llama_index.core.instrumentation.events.span import (
    SpanDropEvent,
)
from llama_index.core.instrumentation.events.synthesis import (
    # GetResponseEndEvent,
    GetResponseStartEvent,
    SynthesizeEndEvent,
    SynthesizeStartEvent,
)
from llama_index.core.instrumentation.span import SimpleSpan
from llama_index.core.instrumentation.span_handlers.base import BaseSpanHandler
from treelib import Tree

if TYPE_CHECKING:
    from llama_index.core.instrumentation.dispatcher import Dispatcher
    from llama_index.core.instrumentation.events import BaseEvent
    from llama_index.core.schema import BaseNode, NodeWithScore

#####################################################
## Code
logger = logging.getLogger(__name__)

@st.cache_resource
def get_callback_manager() -> CallbackManager:
    """Create the callback manager for the code."""
    return CallbackManager([LlamaDebugHandler()])


def display_source_node(source_node: NodeWithScore, max_length: int = 100) -> str:
    source_text = source_node.node.get_content().strip()
    source_text = source_text[:max_length] + "..." if len(source_text) > max_length else source_text
    return (
        f"**Node ID:** {source_node.node.node_id}<br>"
        f"**Similarity:** {source_node.score}<br>"
        f"**Text:** {source_text}<br>"
    )

class RAGEventHandler(BaseEventHandler):
    """Pruned RAG Event Handler."""

    # events: List[BaseEvent] = []  # TODO: handle removing historical events if they're too old.

    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "RAGEventHandler"

    def handle(self, event: BaseEvent, **kwargs: Any) -> None:
        """Logic for handling event."""
        print("-----------------------")
        # all events have these attributes
        print(event.id_)
        print(event.timestamp)
        print(event.span_id)

        # event specific attributes
        if isinstance(event, LLMChatStartEvent):
            # initial
            print(event.messages)
            print(event.additional_kwargs)
            print(event.model_dict)
        elif isinstance(event, LLMChatInProgressEvent):
            # streaming
            print(event.response.delta)
        elif isinstance(event, LLMChatEndEvent):
            # final response
            print(event.response)

        # self.events.append(event)
        print("-----------------------")

class FullRAGEventHandler(BaseEventHandler):
    """RAG event handler. Built off the example custom event handler.

    In general, logged events are treated as single events in a point in time,
    that link to a span. The span is a collection of events that are related to
    a single task. The span is identified by a unique span_id.

    While events are independent, there is some hierarchy.
    For example, in query_engine.query() call with a reranker attached:
    - QueryStartEvent
    - RetrievalStartEvent
    - EmbeddingStartEvent
    - EmbeddingEndEvent
    - RetrievalEndEvent
    - RerankStartEvent
    - RerankEndEvent
    - SynthesizeStartEvent
    - GetResponseStartEvent
    - LLMPredictStartEvent
    - LLMChatStartEvent
    - LLMChatEndEvent
    - LLMPredictEndEvent
    - GetResponseEndEvent
    - SynthesizeEndEvent
    - QueryEndEvent
    """

    events: ClassVar[list[BaseEvent]] = []
    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "RAGEventHandler"

    def _print_event_nodes(self, event_nodes: Sequence[NodeWithScore | BaseNode]) -> str:
        """Print a list of nodes nicely."""
        output_str = "["
        for node in event_nodes:
            output_str += (str(display_source_node(node, 1000)) + "\n")
            output_str += "* * * * * * * * * * * *"
        output_str += "]"
        return (output_str)

    def handle(self, event: BaseEvent, **kwargs: Any) -> None:
        """Logic for handling event."""
        logger.info("-----------------------")
        # all events have these attributes
        logger.info(event.id_)
        logger.info(event.timestamp)
        logger.info(event.span_id)

        # event specific attributes
        logger.info(f"Event type: {event.class_name()}")
        if isinstance(event, AgentRunStepStartEvent):
            # logger.info(event.task_id)
            logger.info(event.step)
            logger.info(event.input)
        if isinstance(event, AgentRunStepEndEvent):
            logger.info(event.step_output)
        if isinstance(event, AgentChatWithStepStartEvent):
            logger.info(event.user_msg)
        if isinstance(event, AgentChatWithStepEndEvent):
            logger.info(event.response)
        if isinstance(event, AgentToolCallEvent):
            logger.info(event.arguments)
            logger.info(event.tool.name)
            logger.info(event.tool.description)
        if isinstance(event, StreamChatDeltaReceivedEvent):
            logger.info(event.delta)
        if isinstance(event, StreamChatErrorEvent):
            logger.info(event.exception)
        if isinstance(event, EmbeddingStartEvent):
            logger.info(event.model_dict)
        if isinstance(event, EmbeddingEndEvent):
            logger.info(event.chunks)
            logger.info(event.embeddings[0][:5])  # avoid printing all embeddings
        if isinstance(event, LLMPredictStartEvent):
            logger.info(event.template)
            logger.info(event.template_args)
        if isinstance(event, LLMPredictEndEvent):
            logger.info(event.output)
        if isinstance(event, LLMStructuredPredictStartEvent):
            logger.info(event.template)
            logger.info(event.template_args)
            logger.info(event.output_cls)
        if isinstance(event, LLMStructuredPredictEndEvent):
            logger.info(event.output)
        if isinstance(event, LLMCompletionStartEvent):
            logger.info(event.model_dict)
            logger.info(event.prompt)
            logger.info(event.additional_kwargs)
        if isinstance(event, LLMCompletionEndEvent):
            logger.info(event.response)
            logger.info(event.prompt)
        if isinstance(event, LLMChatInProgressEvent):
            logger.info(event.messages)
            logger.info(event.response)
        if isinstance(event, LLMChatStartEvent):
            logger.info(event.messages)
            logger.info(event.additional_kwargs)
            logger.info(event.model_dict)
        if isinstance(event, LLMChatEndEvent):
            logger.info(event.messages)
            logger.info(event.response)
        if isinstance(event, RetrievalStartEvent):
            logger.info(event.str_or_query_bundle)
        if isinstance(event, RetrievalEndEvent):
            logger.info(event.str_or_query_bundle)
            # logger.info(event.nodes)
            logger.info(self._print_event_nodes(event.nodes))
        if isinstance(event, ReRankStartEvent):
            logger.info(event.query)
            # logger.info(event.nodes)
            for node in event.nodes:
                logger.info(display_source_node(node))
            logger.info(event.top_n)
            logger.info(event.model_name)
        if isinstance(event, ReRankEndEvent):
            # logger.info(event.nodes)
            logger.info(self._print_event_nodes(event.nodes))
        if isinstance(event, QueryStartEvent):
            logger.info(event.query)
        if isinstance(event, QueryEndEvent):
            logger.info(event.response)
            logger.info(event.query)
        if isinstance(event, SpanDropEvent):
            logger.info(event.err_str)
        if isinstance(event, SynthesizeStartEvent):
            logger.info(event.query)
        if isinstance(event, SynthesizeEndEvent):
            logger.info(event.response)
            logger.info(event.query)
        if isinstance(event, GetResponseStartEvent):
            logger.info(event.query_str)
        self.events.append(event)
        logger.info("-----------------------")

    def _get_events_by_span(self) -> dict[str, list[BaseEvent]]:
        events_by_span: dict[str, list[BaseEvent]] = {}
        for event in self.events:
            if event.span_id in events_by_span:
                events_by_span[event.span_id].append(event)
            elif (event.span_id is not None):
                events_by_span[event.span_id] = [event]
        return events_by_span

    def _get_event_span_trees(self) -> list[Tree]:
        events_by_span = self._get_events_by_span()

        trees = []
        tree = Tree()

        for span, sorted_events in events_by_span.items():
            # create root node i.e. span node
            tree.create_node(
                tag=f"{span} (SPAN)",
                identifier=span,
                parent=None,
                data=sorted_events[0].timestamp,
            )
            for event in sorted_events:
                tree.create_node(
                    tag=f"{event.class_name()}: {event.id_}",
                    identifier=event.id_,
                    parent=event.span_id,
                    data=event.timestamp,
                )
            trees.append(tree)
            tree = Tree()
        return trees

    def print_event_span_trees(self) -> None:
        """View trace trees."""
        trees = self._get_event_span_trees()
        for tree in trees:
            logger.info(
                tree.show(
                    stdout=False, sorting=True, key=lambda node: node.data
                )
            )
            logger.info("")


class RAGSpanHandler(BaseSpanHandler[SimpleSpan]):
    span_dict: dict = {}

    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "ExampleSpanHandler"

    def new_span(
        self,
        id_: str,
        bound_args: Any,
        instance: Any | None = None,
        parent_span_id: str | None = None,
        **kwargs: Any,
    ) -> SimpleSpan | None:
        """Create a span."""
        # logic for creating a new MyCustomSpan
        if id_ not in self.span_dict:
            self.span_dict[id_] = []
        self.span_dict[id_].append(
            SimpleSpan(id_=id_, parent_id=parent_span_id)
        )

    def prepare_to_exit_span(
        self,
        id_: str,
        bound_args: Any,
        instance: Any | None = None,
        result: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """Logic for preparing to exit a span."""
        # if id in self.span_dict:
        #    return self.span_dict[id].pop()

    def prepare_to_drop_span(
        self,
        id_: str,
        bound_args: Any,
        instance: Any | None = None,
        err: BaseException | None = None,
        **kwargs: Any,
    ) -> Any:
        """Logic for preparing to drop a span."""
        # if id in self.span_dict:
        #    return self.span_dict[id].pop()


def get_obs() -> Dispatcher:
    """Get observability for the RAG pipeline."""
    dispatcher = get_dispatcher()
    event_handler = RAGEventHandler()
    span_handler = RAGSpanHandler()

    dispatcher.add_event_handler(event_handler)
    dispatcher.add_span_handler(span_handler)
    return dispatcher
