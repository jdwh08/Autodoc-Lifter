#####################################################
### DOCUMENT PROCESSOR [RETRIEVER]
#####################################################
# Jonathan Wang

# ABOUT: 
# This project creates an app to chat with PDFs.

# This is the RETRIEVER
# which defines the main way that document
# snippets are identified.

#####################################################
## TODO:

#####################################################
## IMPORTS:
import logging
from typing import Optional, List, Tuple, Dict, cast
from collections import defaultdict

import streamlit as st

import numpy as np

from llama_index.core.utils import truncate_text
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever
from llama_index.retrievers.bm25 import BM25Retriever

from llama_index.core import VectorStoreIndex #, StorageContext, 
from llama_index.core.schema import BaseNode, IndexNode, NodeWithScore, QueryBundle
from llama_index.core.callbacks.base import CallbackManager

# Own Modules:
from merger import _merge_on_scores

# Lazy Loading:

#####################################################
## CODE:
class RAGRetriever(BaseRetriever):
    """
    Jonathan Wang's custom built retriever over our vector store.
    Combination of Hybrid Retrieval (BM25 x Vector Embeddings) + AutoMergingRetriever
    https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/retrievers/auto_merging_retriever.py
    """    
    def __init__(
        self,
        vector_store_index: VectorStoreIndex,

        semantic_top_k: int = 10,
        sparse_top_k: int = 6,

        fusion_similarity_top_k: int = 10,  # total number of snippets to retrieve after the Reicprocal Rerank.
        semantic_weight_fraction: float = 0.6,  # percentage weight to give to semantic cosine vs sparse bm25
        merge_up_thresh: float = 0.5,  # fraction of nodes needed to be retrieved to merge up to semantic level

        verbose: bool = True,
        callback_manager: Optional[CallbackManager] = None,
        object_map: Optional[dict] = None,
        objects: Optional[List[IndexNode]] = None,
    ) -> None:
        """Init params."""
        self._vector_store_index = vector_store_index
        
        self.sentence_vector_retriever = VectorIndexRetriever(
            index=vector_store_index, similarity_top_k=semantic_top_k
        )
        self.sentence_bm25_retriever = BM25Retriever.from_defaults(
            # nodes=list(vector_store_index.storage_context.docstore.docs.values())
            index=vector_store_index  # TODO: Confirm this works.
            , similarity_top_k=sparse_top_k
        )

        self._fusion_similarity_top_k = fusion_similarity_top_k
        self._semantic_weight_fraction = semantic_weight_fraction
        self._merge_up_thresh = merge_up_thresh

        super().__init__(
            # callback_manager=callback_manager,
            object_map=object_map,
            objects=objects,
            verbose=verbose,
        )

    
    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "RAGRetriever"

    
    def _get_parents_and_merge(
        self, nodes: List[NodeWithScore]
    ) -> Tuple[List[NodeWithScore], bool]:
        """Get parents and merge nodes."""
        # retrieve all parent nodes
        parent_nodes: Dict[str, BaseNode] = {}
        parent_cur_children_dict: Dict[str, List[NodeWithScore]] = defaultdict(list)
        for node in nodes:
            if node.node.parent_node is None:
                continue
            parent_node_info = node.node.parent_node

            # Fetch actual parent node if doesn't exist in `parent_nodes` cache yet
            parent_node_id = parent_node_info.node_id
            if parent_node_id not in parent_nodes:
                parent_node = self._vector_store_index.storage_context.docstore.get_document(
                    parent_node_id
                )
                parent_nodes[parent_node_id] = cast(BaseNode, parent_node)

            # add reference to child from parent
            parent_cur_children_dict[parent_node_id].append(node)

        # compute ratios and "merge" nodes
        # merging: delete some children nodes, add some parent nodes
        node_ids_to_delete = set()
        nodes_to_add: Dict[str, BaseNode] = {}
        for parent_node_id, parent_node in parent_nodes.items():
            parent_child_nodes = parent_node.child_nodes
            parent_num_children = len(parent_child_nodes) if parent_child_nodes else 1
            parent_cur_children = parent_cur_children_dict[parent_node_id]
            ratio = len(parent_cur_children) / parent_num_children

            # if ratio is high enough, merge up to the next level in the hierarchy
            if ratio > self._merge_up_thresh:
                node_ids_to_delete.update(
                    set({n.node.node_id for n in parent_cur_children})
                )

                parent_node_text = truncate_text(getattr(parent_node, 'text', ''), 100)
                info_str = (
                    f"> Merging {len(parent_cur_children)} nodes into parent node.\n"
                    f"> Parent node id: {parent_node_id}.\n"
                    f"> Parent node text: {parent_node_text}\n"
                )
                # logger.info(info_str)
                if self._verbose:
                    print(info_str)

                # add parent node
                # can try averaging score across embeddings for now
                avg_score = sum(
                    [n.get_score() or 0.0 for n in parent_cur_children]
                ) / len(parent_cur_children)
                parent_node_with_score = NodeWithScore(
                    node=parent_node, score=avg_score
                )
                nodes_to_add[parent_node_id] = parent_node_with_score  # type: ignore (NodesWithScore is a child of BaseNode)

        # delete old child nodes, add new parent nodes
        new_nodes = [n for n in nodes if n.node.node_id not in node_ids_to_delete]
        # add parent nodes
        new_nodes.extend(list(nodes_to_add.values()))  # type: ignore (NodesWithScore is a child of BaseNode)

        is_changed = len(node_ids_to_delete) > 0
        return new_nodes, is_changed

    
    def _fill_in_nodes(
        self, nodes: List[NodeWithScore]
    ) -> Tuple[List[NodeWithScore], bool]:
        """Fill in nodes."""
        new_nodes = []
        is_changed = False
        for idx, node in enumerate(nodes):
            new_nodes.append(node)
            if idx >= len(nodes) - 1:
                continue

            cur_node = cast(BaseNode, node.node)
            # if there's a node in the middle, add that to the queue
            if (
                cur_node.next_node is not None
                and cur_node.next_node == nodes[idx + 1].node.prev_node
            ):
                is_changed = True
                next_node = self._vector_store_index.storage_context.docstore.get_document(
                    cur_node.next_node.node_id
                )
                next_node = cast(BaseNode, next_node)

                next_node_text = truncate_text(getattr(next_node, 'text', ''), 100)  # TODO: why not higher?
                info_str = (
                    f"> Filling in node. Node id: {cur_node.next_node.node_id}"
                    f"> Node text: {next_node_text}\n"
                )
                # logger.info(info_str)
                if self._verbose:
                    print(info_str)

                # set score to be average of current node and next node
                avg_score = (node.get_score() + nodes[idx + 1].get_score()) / 2
                new_nodes.append(NodeWithScore(node=next_node, score=avg_score))
        return new_nodes, is_changed

    
    def _try_merging(
        self, nodes: List[NodeWithScore]
    ) -> Tuple[List[NodeWithScore], bool]:
        """Try different ways to merge nodes."""
        # first try filling in nodes
        nodes, is_changed_0 = self._fill_in_nodes(nodes)
        # then try merging nodes
        nodes, is_changed_1 = self._get_parents_and_merge(nodes)
        return nodes, is_changed_0 or is_changed_1


    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve."""
        # Get vector stores retrieved nodes
        vector_sentence_nodes = self.sentence_vector_retriever.retrieve(query_bundle)# , **kwargs)
        bm25_sentence_nodes = self.sentence_bm25_retriever.retrieve(query_bundle)# , **kwargs)

        # Get initial nodes from hybrid search.
        initial_nodes = _merge_on_scores(
            vector_sentence_nodes,
            bm25_sentence_nodes,
            [getattr(a, "score", np.nan) for a in vector_sentence_nodes],
            [getattr(b, "score", np.nan) for b in bm25_sentence_nodes],
            a_weight=self._semantic_weight_fraction,
            top_k=self._fusion_similarity_top_k
        )
        
        # Merge nodes
        cur_nodes, is_changed = self._try_merging(list(initial_nodes))  # technically _merge_on_scores returns a sequence.
        while is_changed:
            cur_nodes, is_changed = self._try_merging(cur_nodes)

        # sort by similarity
        cur_nodes.sort(key=lambda x: x.get_score(), reverse=True)

        # some other reranking and filtering node postprocessors here?
        # https://docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/root.html
        return cur_nodes

@st.cache_resource
def get_retriever(
    _vector_store_index: VectorStoreIndex,

    semantic_top_k: int = 10,
    sparse_top_k: int = 6,

    fusion_similarity_top_k: int = 10,  # total number of snippets to retrieve after the Reicprocal Rerank.
    semantic_weight_fraction: float = 0.6,  # percentage weight to give to semantic chunks over sentence chunks
    merge_up_thresh: float = 0.5,  # fraction of nodes needed to be retrieved to merge up to semantic level

    verbose: bool = True,
    _callback_manager: Optional[CallbackManager] = None,
    object_map: Optional[dict] = None,
    objects: Optional[List[IndexNode]] = None,
) -> BaseRetriever:
    """Get the retriver to use.

    Args:
        vector_store_index (VectorStoreIndex): The vector store to query on.
        semantic_top_k (int, optional): Top k nodes to retrieve semantically (cosine). Defaults to 10.
        sparse_top_k (int, optional): Top k nodes to retrieve sparsely (BM25). Defaults to 6.
        fusion_similarity_top_k (int, optional): Maximum number of nodes to retrieve after fusing. Defaults to 10.
        callback_manager (Optional[CallbackManager], optional): Callback manager. Defaults to None.
        object_map (Optional[dict], optional): Object map. Defaults to None.
        objects (Optional[List[IndexNode]], optional): Objects list. Defaults to None.

    Returns:
        BaseRetriever: Retriever to use.
    """
    retriever = RAGRetriever(
        vector_store_index=_vector_store_index, 
        semantic_top_k=semantic_top_k,
        sparse_top_k=sparse_top_k,
        fusion_similarity_top_k=fusion_similarity_top_k,
        semantic_weight_fraction=semantic_weight_fraction,
        merge_up_thresh=merge_up_thresh,
        verbose=verbose,
        callback_manager=_callback_manager,
        object_map=object_map,
        objects=objects
    )
    return (retriever)
