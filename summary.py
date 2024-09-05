#####################################################
### DOCUMENT PROCESSOR [Summarizer]
#####################################################
### Jonathan Wang

# ABOUT: 
# This creates an app to chat with PDFs.

# This is the Summarizer
# Which creates summaries based on documents.
#####################################################
### TODO Board:
# Summary Index for document?

# https://docs.llamaindex.ai/en/stable/examples/response_synthesizers/tree_summarize/
# https://sourajit16-02-93.medium.com/text-summarization-unleashed-novice-to-maestro-with-llms-and-instant-code-solutions-8d26747689c4

#####################################################
### PROGRAM SETTINGS


#####################################################
### PROGRAM IMPORTS
import logging

from typing import Optional, Sequence, Any, Callable, cast
from llama_index.core.bridge.pydantic import Field, PrivateAttr

from llama_index.core.settings import Settings
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.multi_modal_llms import MultiModalLLM
from llama_index.core.schema import BaseNode, TextNode, ImageDocument
from llama_index.core.callbacks.base import CallbackManager

from llama_index.core.response_synthesizers import TreeSummarize

# Own Modules
from metadata_adder import ModelMetadataAdder

#####################################################
### CONSTANTS
logger = logging.getLogger(__name__)

DEFAULT_SUMMARY_TEMPLATE = """You are an expert summarizer of information. You are given some information from a document. Summarize the information, and then provide the key information that can be drawn from it. The information is below:
{context_str}
"""

DEFAULT_ONELINE_SUMMARY_TEMPLATE = """You are an expert summarizer of information. You are given a summary of a document. In no more than three sentences, describe the subject of the document, the main ideas of the document, and what types of questions can be answered from it."""

DEFAULT_TREE_SUMMARY_TEMPLATE = """You are an expert summarizer of information. You are given some text from a document.
Please provide a comprehensive summary of the text.
Include the main subject of the text, the key points or topics, and the most important conclusions if there are any.
The summary should be detailed yet concise."""

DEFAULT_TABLE_SUMMARY_TEMPLATE = """You are an expert summarizer of tables. You are given a table or part of a table in HTML format. The table is below:
{context_str}
----------------
Summarize the table, and then provide the key insights that can be drawn directly from the table. If this is not actually an HTML table or part of an HTML table, please do not respond.
"""

DEFAULT_IMAGE_SUMMARY_TEMPLATE = """You are an expert image summarizer. You are given an image. Summarize the image, and then provide the key insights that can be drawn directly from the image, if there are any.
"""

#####################################################
### SCRIPT

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

        # Check whethr the image is of text or not
        ### TODO: Replace this with a text-overlap thing.
        image = node.resolve_image() # type: ignore | we check for this above.
        im_width, im_height = image.size
        if (im_width < 70):  # TODO: this really should be based on the average text width / whether this is overlapping text.
            return None

        ## NOTE: We're assuming that the llm complete function has a parameter `images` to send image node(s) as input.
        ## This is NOT necessarily true if the end user decides to create their own implementation of a MultiModalLLM.
        response = self._llm.complete(
            prompt=self.prompt_template,
            image_documents=[
                cast(ImageDocument, node)  # NOTE: This is a hack. Technically, node should be an ImageNode, a parent of ImageDocument; but I don't think we'll be using the Document features so this should be okay.
            ],
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


def get_tree_summarizer(
    llm: Optional[BaseLLM] = None,
    callback_manager: Optional[CallbackManager] = None,
):
    llm = llm or Settings.llm
    tree_summarizer = TreeSummarize(llm=llm, callback_manager=callback_manager)
    return (tree_summarizer)


def get_tree_summary(tree_summarizer: TreeSummarize, text_chunks: Sequence[BaseNode]) -> str:
    """Summarize the text nodes using a tree summarizer.

    Args:
        tree_summarizer (TreeSummarize): The tree summarizer to use.
        text_chunks (Sequence[BaseNode]): The text nodes to summarize.
    
    Returns:
        str: The summarized text.
    """
    response = tree_summarizer.aget_response(query_str=DEFAULT_TREE_SUMMARY_TEMPLATE, text_chunks=[getattr(text_chunks, 'text') for text_chunks in text_chunks if hasattr(text_chunks, 'text')])
    return response.response