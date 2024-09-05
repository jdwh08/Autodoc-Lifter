#####################################################
### DOCUMENT PROCESSOR [ENGINE]
#####################################################
# Jonathan Wang

# ABOUT:
# This project creates an app to chat with PDFs.

# This is the ENGINE
# which defines how LLMs handle processing.
#####################################################
## TODO Board:

#####################################################
## IMPORTS
from __future__ import annotations

import gc
from typing import TYPE_CHECKING, Callable, List, Optional, cast

from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.settings import (
    Settings,
)
from torch.cuda import empty_cache

if TYPE_CHECKING:
    from llama_index.core.base.response.schema import Response
    from llama_index.core.callbacks import CallbackManager
    from llama_index.core.postprocessor.types import BaseNodePostprocessor
    from llama_index.core.response_synthesizers import (
        BaseSynthesizer,
    )
    from llama_index.core.retrievers import BaseRetriever

# Own Modules

#####################################################
## CODE
class RAGQueryEngine(CustomQueryEngine):
    """Custom RAG Query Engine."""

    retriever: BaseRetriever
    response_synthesizer: BaseSynthesizer
    node_postprocessors: Optional[List[BaseNodePostprocessor]] = []

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
    #     #     Settings.callback_manager
    #     # )
    #     # llm = llm or Settings.llm

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
        self, nodes: list[NodeWithScore], query_bundle: QueryBundle
    ) -> list[NodeWithScore]:
        if self.node_postprocessors is None:
            return nodes

        for node_postprocessor in self.node_postprocessors:
            nodes = node_postprocessor.postprocess_nodes(
                nodes, query_bundle=query_bundle
            )
        return nodes

    def retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        nodes = self.retriever.retrieve(query_bundle)
        return self._apply_node_postprocessors(nodes, query_bundle=query_bundle)

    async def aretrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        nodes = await self.retriever.aretrieve(query_bundle)
        return self._apply_node_postprocessors(nodes, query_bundle=query_bundle)

    def custom_query(self, query_str: str) -> Response:
        # Convert query string into query bundle
        query_bundle = QueryBundle(query_str=query_str)
        nodes = self.retrieve(query_bundle)  # also does the postprocessing.

        response_obj = self.response_synthesizer.synthesize(query_bundle, nodes)

        empty_cache()
        gc.collect()
        return cast(Response, response_obj)  # type: ignore


# @st.cache_resource  # none of these can be hashable or cached :(
def get_engine(
    retriever: BaseRetriever,
    response_synthesizer: BaseSynthesizer,
    node_postprocessors: list[BaseNodePostprocessor] | None = None,
    callback_manager: CallbackManager | None = None,
) -> RAGQueryEngine:
    return RAGQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=node_postprocessors,
        callback_manager=callback_manager or Settings.callback_manager,
    )
