#####################################################
# LIFT DOCUMENT INTAKE PROCESSOR [OBSERVATION/LOGGING]
#####################################################
# Jonathan Wang (jwang15, jonathan.wang@discover.com)
# Greenhouse GenAI Modeling Team

# ABOUT: 
# This project automates the processing of legal documents 
# requesting information about our customers.
# These are handled by LIFT
# (Legal, Investigation, and Fraud Team)

# This is the Observation and Logging
# to see the actions undertaken in the RAG pipeline.
#####################################################
## TODOS:
# 

#####################################################
## IMPORTS:
from typing import Dict, List, Any, Optional
from treelib import Tree

import streamlit as st

# Callbacks
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler

# Loads of stuff for the CICD
from llama_index.core.instrumentation.events import BaseEvent
from llama_index.core.instrumentation.event_handlers import BaseEventHandler

from llama_index.core.instrumentation.events.agent import (
    AgentChatWithStepStartEvent,
    AgentChatWithStepEndEvent,
    AgentRunStepStartEvent,
    AgentRunStepEndEvent,
    AgentToolCallEvent,
)
from llama_index.core.instrumentation.events.chat_engine import (
    StreamChatErrorEvent,
    StreamChatDeltaReceivedEvent,
)
from llama_index.core.instrumentation.events.embedding import (
    EmbeddingStartEvent,
    EmbeddingEndEvent,
)
from llama_index.core.instrumentation.events.llm import (
    LLMPredictEndEvent,
    LLMPredictStartEvent,
    LLMStructuredPredictEndEvent,
    LLMStructuredPredictStartEvent,
    LLMCompletionEndEvent,
    LLMCompletionStartEvent,
    LLMChatEndEvent,
    LLMChatStartEvent,
    LLMChatInProgressEvent,
)
from llama_index.core.instrumentation.events.query import (
    QueryStartEvent,
    QueryEndEvent,
)
from llama_index.core.instrumentation.events.rerank import (
    ReRankStartEvent,
    ReRankEndEvent,
)
from llama_index.core.instrumentation.events.retrieval import (
    RetrievalStartEvent,
    RetrievalEndEvent,
)
from llama_index.core.instrumentation.events.span import (
    SpanDropEvent,
)
from llama_index.core.instrumentation.events.synthesis import (
    SynthesizeStartEvent,
    SynthesizeEndEvent,
    GetResponseEndEvent,
    GetResponseStartEvent,
)
from llama_index.core.instrumentation.span import SimpleSpan
from llama_index.core.instrumentation.span_handlers.base import BaseSpanHandler
# Pretty Printing
from llama_index.core.response.notebook_utils import display_source_node

# End user handler
from llama_index.core.instrumentation import get_dispatcher

#####################################################
## Code
@st.cache_resource
def get_callback_manager() -> CallbackManager:
    """Create the callback manager for the code."""
    callback_manager = CallbackManager([LlamaDebugHandler()])
    return callback_manager


class RAGEventHandler(BaseEventHandler):
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
    events: List[BaseEvent] = []

    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "RAGEventHandler"

    def _print_event_nodes(event_nodes: List) -> str:
        """Print a list of nodes nicely."""
        output_str = "["
        for node in event_nodes:
            output_str += (display_source_node(node, 1000) + "\n")
            output_str += "* * * * * * * * * * * *"
        output_str += "]"
        return (output_str)

    def handle(self, event: BaseEvent) -> None:
        """Logic for handling event."""
        print("-----------------------")
        # all events have these attributes
        print(event.id_)
        print(event.timestamp)
        print(event.span_id)

        # event specific attributes
        print(f"Event type: {event.class_name()}")
        if isinstance(event, AgentRunStepStartEvent):
            print(event.task_id)
            print(event.step)
            print(event.input)
        if isinstance(event, AgentRunStepEndEvent):
            print(event.step_output)
        if isinstance(event, AgentChatWithStepStartEvent):
            print(event.user_msg)
        if isinstance(event, AgentChatWithStepEndEvent):
            print(event.response)
        if isinstance(event, AgentToolCallEvent):
            print(event.arguments)
            print(event.tool.name)
            print(event.tool.description)
            print(event.tool.to_openai_tool())
        if isinstance(event, StreamChatDeltaReceivedEvent):
            print(event.delta)
        if isinstance(event, StreamChatErrorEvent):
            print(event.exception)
        if isinstance(event, EmbeddingStartEvent):
            print(event.model_dict)
        if isinstance(event, EmbeddingEndEvent):
            print(event.chunks)
            print(event.embeddings[0][:5])  # avoid printing all embeddings
        if isinstance(event, LLMPredictStartEvent):
            print(event.template)
            print(event.template_args)
        if isinstance(event, LLMPredictEndEvent):
            print(event.output)
        if isinstance(event, LLMStructuredPredictStartEvent):
            print(event.template)
            print(event.template_args)
            print(event.output_cls)
        if isinstance(event, LLMStructuredPredictEndEvent):
            print(event.output)
        if isinstance(event, LLMCompletionStartEvent):
            print(event.model_dict)
            print(event.prompt)
            print(event.additional_kwargs)
        if isinstance(event, LLMCompletionEndEvent):
            print(event.response)
            print(event.prompt)
        if isinstance(event, LLMChatInProgressEvent):
            print(event.messages)
            print(event.response)
        if isinstance(event, LLMChatStartEvent):
            print(event.messages)
            print(event.additional_kwargs)
            print(event.model_dict)
        if isinstance(event, LLMChatEndEvent):
            print(event.messages)
            print(event.response)
        if isinstance(event, RetrievalStartEvent):
            print(event.str_or_query_bundle)
        if isinstance(event, RetrievalEndEvent):
            print(event.str_or_query_bundle)
            # print(event.nodes)
            print(_print_event_nodes(event.nodes))
        if isinstance(event, ReRankStartEvent):
            print(event.query)
            # print(event.nodes)
            for node in event.nodes:
                print(display_source_node(node))
            print(event.top_n)
            print(event.model_name)
        if isinstance(event, ReRankEndEvent):
            # print(event.nodes)
            print(_print_event_nodes(event.nodes))
        if isinstance(event, QueryStartEvent):
            print(event.query)
        if isinstance(event, QueryEndEvent):
            print(event.response)
            print(event.query)
        if isinstance(event, SpanDropEvent):
            print(event.err_str)
        if isinstance(event, SynthesizeStartEvent):
            print(event.query)
        if isinstance(event, SynthesizeEndEvent):
            print(event.response)
            print(event.query)
        if isinstance(event, GetResponseStartEvent):
            print(event.query_str)
        self.events.append(event)
        print("-----------------------")
        return (None)

    def _get_events_by_span(self) -> Dict[str, List[BaseEvent]]:
        events_by_span: Dict[str, List[BaseEvent]] = {}
        for event in self.events:
            if event.span_id in events_by_span:
                events_by_span[event.span_id].append(event)
            else:
                events_by_span[event.span_id] = [event]
        return events_by_span

    def _get_event_span_trees(self) -> List[Tree]:
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
        """Method for viewing trace trees."""
        trees = self._get_event_span_trees()
        for tree in trees:
            print(
                tree.show(
                    stdout=False, sorting=True, key=lambda node: node.data
                )
            )
            print("")
        return (None)


class RAGSpanHandler(BaseSpanHandler[SimpleSpan]):
    span_dict = {}

    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "ExampleSpanHandler"

    def new_span(
        self,
        id_: str,
        bound_args: Any,
        instance: Optional[Any] = None,
        parent_span_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[SimpleSpan]:
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
        instance: Optional[Any] = None,
        result: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        """Logic for preparing to exit a span."""
        pass
        # if id in self.span_dict:
        #    return self.span_dict[id].pop()

    def prepare_to_drop_span(
        self,
        id_: str,
        bound_args: Any,
        instance: Optional[Any] = None,
        err: Optional[BaseException] = None,
        **kwargs: Any,
    ) -> Any:
        """Logic for preparing to drop a span."""
        pass
        # if id in self.span_dict:
        #    return self.span_dict[id].pop()


@st.cache_resource
def get_obs():
    """Get observability for the RAG pipeline."""
    dispatcher = get_dispatcher()
    event_handler = RAGEventHandler()
    span_handler = RAGSpanHandler()

    dispatcher.add_event_handler(event_handler)
    dispatcher.add_span_handler(span_handler)
    return dispatcher