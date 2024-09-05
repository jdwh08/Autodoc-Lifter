#####################################################
### DOCUMENT PROCESSOR [PDF READER]
#####################################################
# Jonathan Wang

# ABOUT: 
# This project creates an app to chat with PDFs.

# This is the PDF READER.
# It converts a PDF into LlamaIndex nodes
# using UnstructuredIO.
#####################################################
# TODO Board:
# I don't think the current code is elegent... :(
    
# TODO: Replace chunk_by_header with a custom solution replicating bySimilarity
# https://docs.unstructured.io/api-reference/api-services/chunking#by-similarity-chunking-strategy
# Some hybrid thing...
    

# Come up with a awy to handle summarizing images and tables using MultiModalLLM after the processing into nodes.
    # TODO: Put this into PDFReaderUtilities? Along with the other functions for stuff like email?

# Investigate PDFPlumber as a backup/alternative for Unstructured. 
    # `https://github.com/jsvine/pdfplumber`
    # nevermind, this is essentially pdfminer.six but nicer

# Chunk hierarchy from https://www.reddit.com/r/LocalLLaMA/comments/1dpb9ow/how_we_chunk_turning_pdfs_into_hierarchical/
# Investigate document parsing algorithms from https://github.com/BobLd/DocumentLayoutAnalysis?tab=readme-ov-file
# Investigate document parsing algorithms from https://github.com/Filimoa/open-parse?tab=readme-ov-file

# Competition:
    # https://github.com/infiniflow/ragflow
    # https://github.com/deepdoctection/deepdoctection

#####################################################
## IMPORTS
import os
import re
import regex
from copy import deepcopy

from abc import ABC, abstractmethod
from typing import Any, List, Tuple, IO, Optional, Type, Generic, TypeVar
from llama_index.core.bridge.pydantic import Field

import numpy as np

from io import BytesIO
from base64 import b64encode, b64decode
from PIL import Image as PILImage

# from pdf_reader_utils import clean_pdf_chunk, dedupe_title_chunks, combine_listitem_chunks

# Unstructured Document Parsing
from unstructured.partition.pdf import partition_pdf
# from unstructured.cleaners.core import clean_extra_whitespace, group_broken_paragraphs #, clean_ordered_bullets, clean_bullets, clean_dashes
# from unstructured.chunking.title import chunk_by_title
# Unstructured Element Types
from unstructured.documents import elements, email_elements
from unstructured.partition.utils.constants import PartitionStrategy

# Llamaindex Nodes
from llama_index.core.settings import Settings
from llama_index.core.schema import Document, BaseNode, TextNode, ImageNode, NodeRelationship, RelatedNodeInfo
from llama_index.core.readers.base import BaseReader
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.node_parser import NodeParser

# Parallelism for cleaning chunks
from joblib import Parallel, delayed

## Lazy Imports
# import nltk
#####################################################

# Additional padding around the PDF extracted images
PDF_IMAGE_HORIZONTAL_PADDING = 20
PDF_IMAGE_VERTICAL_PADDING = 20
os.environ['EXTRACT_IMAGE_BLOCK_CROP_HORIZONTAL_PAD'] = str(PDF_IMAGE_HORIZONTAL_PADDING)
os.environ['EXTRACT_IMAGE_BLOCK_CROP_VERTICAL_PAD'] = str(PDF_IMAGE_VERTICAL_PADDING)

# class TextReader(BaseReader):
#     def __init__(self, text: str) -> None:
#         """Init params."""
#         self.text = text


# class ImageReader(BaseReader):
#     def __init__(self, image: Any) -> None:
#         """Init params."""
#         self.image = image

GenericNode = TypeVar("GenericNode", bound=BaseNode)  # https://mypy.readthedocs.io/en/stable/generics.html

class UnstructuredPDFReader():
    # Yes, we could inherit from LlamaIndex BaseReader even though I don't think it's a good idea.
    # Have you seen the Llamaindex Base Reader? It's silly. """OOP"""
        # https://docs.llamaindex.ai/en/stable/api_reference/readers/

    # here I'm basically cargo culting off the (not-very-good) pre-built Llamaindex one.
        # https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/unstructured/base.py

    # yes I do want to bind these to the class. 
    # you better not be changing the embedding model or node parser on me across different PDFReaders. that's absurd.
    # embed_model: BaseEmbedding
    # _node_parser: NodeParser# = Field(
    #     description="Node parser to run on each Unstructured Title Chunk",
    #     default=Settings.node_parser,
    # )
    _max_characters: int# = Field(
    #     description="The maximum number of characters in a node",
    #     default=8192,
    # )
    _new_after_n_chars: int #= Field(
    #     description="The number of characters after which a new node is created",
    #     default=1024,
    # )
    _overlap_n_chars: int #= Field(
    #     description="The number of characters to overlap between nodes",
    #     default=128,
    # )
    _overlap: int #= Field(
    #     description="The number of characters to overlap between nodes",
    #     default=128,
    # )
    _overlap_all: bool #= Field(
    #     description="Whether to overlap all nodes",
    #     default=False,
    # )
    _multipage_sections: bool #= Field(
    #     description="Whether to include multipage sections",
    #     default=False,
    # )

    ## TODO: Fix this big ball of primiatives and turn it into a class.
    def __init__(
        self,
        # node_parser: Optional[NodeParser],  # Suggest using a SemanticNodeParser.
        max_characters: int = 2048, 
        new_after_n_chars: int = 512, 
        overlap_n_chars: int = 128, 
        overlap: int = 128, 
        overlap_all: bool = False, 
        multipage_sections: bool = True, 
        **kwargs: Any
    ) -> None:
        # node_parser = node_parser or Settings.node_parser
        """Init params."""
        super().__init__(**kwargs)

        self._max_characters = max_characters
        self._new_after_n_chars = new_after_n_chars
        self._overlap_n_chars = overlap_n_chars
        self._overlap = overlap
        self._overlap_all = overlap_all
        self._multipage_sections = multipage_sections
        # self._node_parser = node_parser or Settings.node_parser  # set node parser to run on each Unstructured Title Chunk

        # Prerequisites for Unstructured.io to work
        # import nltk
        # nltk.data.path = ['./nltk_data']
        # try: 
        #     if not nltk.data.find("tokenizers/punkt"):
        #         # nltk.download("punkt")
        #         print("Can't find punkt.")
        # except Exception as e:
        #     # nltk.download("punkt")
        #     print(e)
        # try: 
        #     if not nltk.data.find("taggers/averaged_perceptron_tagger"):
        #         # nltk.download("averaged_perceptron_tagger")
        #         print("Can't find averaged_perceptron_tagger.")
        # except Exception as e:
        #     # nltk.download("averaged_perceptron_tagger")
        #     print(e)


    # """DATA LOADING FUNCTIONS"""
    def _node_rel_prev_next(self, prev_node: GenericNode, next_node: GenericNode) -> Tuple[GenericNode, GenericNode]:
        """Update pre-next node relationships between two nodes."""
        prev_node.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(
            node_id=next_node.node_id,
            metadata={"filename": next_node.metadata['filename']}
        )
        next_node.relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(
            node_id=prev_node.node_id,
            metadata={"filename": prev_node.metadata['filename']}
        )
        return (prev_node, next_node)

    def _node_rel_parent_child(self, parent_node: GenericNode, child_node: GenericNode) -> Tuple[GenericNode, GenericNode]:
        """Update parent-child node relationships between two nodes."""
        parent_node.relationships[NodeRelationship.CHILD] = RelatedNodeInfo(
            node_id=child_node.node_id,
            metadata={"filename": child_node.metadata['filename']}
        )
        child_node.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(
            node_id=parent_node.node_id,
            metadata={"filename": parent_node.metadata['filename']}
        )
        return (parent_node, child_node)
    
    def _handle_metadata(
        self, 
        pdf_chunk: elements.Element, 
        node: GenericNode, 
        kept_metadata: List[str] = [
            'filename', 'file_directory', 'coordinates', 
            'page_number', 'page_name', 'section',
            'sent_from', 'sent_to', 'subject',
            'parent_id', 'category_depth', 
            'text_as_html', 'languages', 
            'emphasized_text_contents', 'link_texts', 'link_urls',
            'is_continuation', 'detection_class_prob',
    ]) -> GenericNode:
        """Add common unstructured element metadata to LlamaIndex node."""
        pdf_chunk_metadata = pdf_chunk.metadata.to_dict() if pdf_chunk.metadata else {}
        current_kept_metadata = deepcopy(kept_metadata)
        
        # Handle some interesting keys
        node.metadata['type'] = pdf_chunk.category
        if (('filename' in current_kept_metadata) and ('filename' in pdf_chunk_metadata) and ('file_directory' in pdf_chunk_metadata)):
            filename = os.path.join(str(pdf_chunk_metadata['file_directory']), str(pdf_chunk_metadata['filename']))
            node.metadata['filename'] = filename
            current_kept_metadata.remove('file_directory') if ('file_directory' in current_kept_metadata) else None
        if (('text_as_html' in current_kept_metadata) and ('text_as_html' in pdf_chunk_metadata)):
            node.metadata['orignal_table_text'] = getattr(node, 'text', '')
            node.text = pdf_chunk_metadata['text_as_html']
            current_kept_metadata.remove('text_as_html')
        if (('coordinates' in current_kept_metadata) and (pdf_chunk_metadata.get('coordinates') is not None)):
            node.metadata['coordinates'] = pdf_chunk_metadata['coordinates']
            current_kept_metadata.remove('coordinates')
        if (('page_number' in current_kept_metadata) and ('page_number' in pdf_chunk_metadata)):
            node.metadata['page_number'] = [pdf_chunk_metadata['page_number']]  # save as list to allow for multiple pages
            current_kept_metadata.remove('page_number')
        if (('page_name' in current_kept_metadata) and ('page_name' in pdf_chunk_metadata)):
            node.metadata['page_name'] = [pdf_chunk_metadata['page_name']]  # save as list to allow for multiple sheets
            current_kept_metadata.remove('page_name')
        
        # Handle the remaining keys
        for key in set(current_kept_metadata).intersection(set(pdf_chunk_metadata.keys())):
            node.metadata[key] = pdf_chunk_metadata[key]
        
        return node
    
    def _handle_text_chunk(self, pdf_text_chunk: elements.Element) -> TextNode:
        """Given a text chunk from Unstructured, convert it to a TextNode for LlamaIndex.

        Args:
            pdf_text_chunk (elements.Element): Input text chunk from Unstructured.

        Returns:
            TextNode: LlamaIndex TextNode which saves the text as HTML for structure.
        """
        new_node = TextNode(
            text=pdf_text_chunk.text, 
            id_=pdf_text_chunk.id,
            excluded_llm_metadata_keys=['type', 'parent_id', 'depth', 'filename', 'coordinates', 'link_texts', 'link_urls', 'link_start_indexes', 'orig_nodes', 'orignal_table_text', 'languages', 'detection_class_prob', 'keyword_metadata'],
            excluded_embed_metadata_keys=['type', 'parent_id', 'depth', 'filename', 'coordinates', 'page number', 'original_text', 'window', 'link_texts', 'link_urls', 'link_start_indexes', 'orig_nodes', 'orignal_table_text', 'languages', 'detection_class_prob']
        )
        new_node = self._handle_metadata(pdf_text_chunk, new_node)
        return (new_node)
    
    
    def _handle_table_chunk(self, pdf_table_chunk: elements.Table | elements.TableChunk) -> TextNode:
        """Given a table chunk from Unstructured, convert it to a TextNode for LlamaIndex.

        Args:
            pdf_table_chunk (elements.Table | elements.TableChunk): Input table chunk from Unstructured

        Returns:
            TextNode: LlamaIndex TextNode which saves the table as HTML for structure.
            
        NOTE: You will need to get the summary of the table for better performance.
        """
        new_node = TextNode(
            text=pdf_table_chunk.metadata.text_as_html if pdf_table_chunk.metadata.text_as_html else pdf_table_chunk.text,
            id_=pdf_table_chunk.id,
            excluded_llm_metadata_keys=['type', 'parent_id', 'depth', 'filename', 'coordinates', 'link_texts', 'link_urls', 'link_start_indexes', 'orig_nodes', 'orignal_table_text', 'languages', 'detection_class_prob', 'keyword_metadata'],
            excluded_embed_metadata_keys=['type', 'parent_id', 'depth', 'filename', 'coordinates', 'page number', 'original_text', 'window', 'link_texts', 'link_urls', 'link_start_indexes', 'orig_nodes', 'orignal_table_text', 'languages', 'detection_class_prob']
        )
        new_node = self._handle_metadata(pdf_table_chunk, new_node)
        return (new_node)
    
    
    def _handle_image_chunk(self, pdf_image_chunk: elements.Element) -> ImageNode:
        """Given an image chunk from UnstructuredIO, read it in and convert it into a Llamaindex ImageNode.

        Args:
            pdf_image_chunk (elements.Element): The input image element from UnstructuredIO. We'll allow all types, just in case you want to process some weird chunks.

        Returns:
            ImageNode: The image saved as a Llamaindex ImageNode.
        """
        pdf_image_chunk_data_available = pdf_image_chunk.metadata.to_dict()
        
        # Check for either saved image_path or image_base64/image_mime_type
        if (('image_path' not in pdf_image_chunk_data_available) and ('image_base64' not in pdf_image_chunk_data_available)):
            raise Exception('Image chunk does not have either image_path or image_base64/image_mime_type. Are you sure this is an image?')
        
        # Make the image node.
        new_node = ImageNode(
            text=pdf_image_chunk.text,
            id_=pdf_image_chunk.id,
            excluded_llm_metadata_keys=['type', 'parent_id', 'depth', 'filename', 'coordinates', 'link_texts', 'link_urls', 'link_start_indexes', 'orig_nodes', 'languages', 'detection_class_prob', 'keyword_metadata'],
            excluded_embed_metadata_keys=['type', 'parent_id', 'depth', 'filename', 'coordinates', 'page number', 'original_text', 'window', 'link_texts', 'link_urls', 'link_start_indexes', 'orig_nodes', 'languages', 'detection_class_prob']
        )
        new_node = self._handle_metadata(pdf_image_chunk, new_node)
        
        # Add image data to image node
        image = None
        if ('image_path' in pdf_image_chunk_data_available):
            # Save image path to image node
            new_node.image_path = pdf_image_chunk_data_available['image_path']
            
            # Load image from path, convert to base64
            image_pil = PILImage.open(pdf_image_chunk_data_available['image_path'])
            image_buffer = BytesIO()
            image_pil.save(image_buffer, format='JPEG')
            image = b64encode(image_buffer.getvalue()).decode('utf-8')
            
            new_node.image = image
            new_node.image_mimetype = 'image/jpeg'
            del image_buffer, image_pil
        elif ('image_base64' in pdf_image_chunk_data_available):
            # Save image base64 to image node
            new_node.image = pdf_image_chunk_data_available['image_base64']
            new_node.image_mimetype = pdf_image_chunk_data_available['image_mime_type']
        
        return (new_node)


    def _handle_composite_chunk(self, pdf_composite_chunk: elements.CompositeElement) -> BaseNode:
        """Given a composite chunk from Unstructured, convert it into a node and handle it dependencies as well."""
        # Start by getting a list of all the nodes which were combined into the composite chunk.
        # child_chunks = pdf_composite_chunk.metadata.to_dict()['orig_elements']
        child_chunks = pdf_composite_chunk.metadata.orig_elements or []
        child_nodes = []
        for chunk in child_chunks:
            child_nodes.append(self._handle_chunk(chunk))  # process all the child chunks.

        # Then build the Composite Chunk into a Node.
        composite_node = self._handle_text_chunk(pdf_text_chunk=pdf_composite_chunk)
        composite_node = self._handle_metadata(pdf_composite_chunk, composite_node)

        # Set relationships between chunks.
        for index in range(1, len(child_nodes)):
            child_nodes[index-1], child_nodes[index] = self._node_rel_prev_next(child_nodes[index-1], child_nodes[index])
        for index, node in enumerate(child_nodes):
            composite_node, child_nodes[index] = self._node_rel_parent_child(composite_node, child_nodes[index])

        composite_node.metadata['orig_nodes'] = child_nodes
        composite_node.excluded_llm_metadata_keys = ['filename', 'coordinates', 'chunk_number', 'window', 'orig_nodes', 'languages', 'detection_class_prob', 'keyword_metadata']
        composite_node.excluded_embed_metadata_keys = ['filename', 'coordinates', 'chunk_number', 'page number', 'original_text', 'window', 'summary', 'orig_nodes', 'languages', 'detection_class_prob']
        return(composite_node)


    def _handle_chunk(self, chunk: elements.Element) -> BaseNode:
        """Convert Unstructured element chunks to Llamaindex Node. Determine which chunk handling to use based on the element type."""
        # Composite (multiple nodes combined together by chunking)
        if (isinstance(chunk, elements.CompositeElement)):
            return (self._handle_composite_chunk(pdf_composite_chunk=chunk))
        # Tables
        elif ((chunk.category == 'Table') and isinstance(chunk, (elements.Table, elements.TableChunk))):
            return(self._handle_table_chunk(pdf_table_chunk=chunk))
        # Images
        elif (any(True for chunk_info in ['image', 'image_base64', 'image_path'] if chunk_info in chunk.metadata.to_dict())):
            return(self._handle_image_chunk(pdf_image_chunk=chunk))
        # Text
        else:
            return(self._handle_text_chunk(pdf_text_chunk=chunk))


    def pdf_to_chunks(
        self, 
        file_path: Optional[str],
        file: Optional[IO[bytes]],
    ) -> List[elements.Element]:
        """
        Given the file path to a PDF, read it in with UnstructuredIO and return its elements.
        """
        print("NEWPDF: Partitioning into Chunks...")
        # 1. attempt using AUTO to have it decide.
        # NOTE: this takes care of pdfminer, and also choses between using detectron2 vs tesseract only.
        # However, it sometimes gets confused by PDFs where text elements are added on later, e.g., CIDs for linking, or REDACTED
        pdf_chunks = partition_pdf(
            filename=file_path,
            file=file,
            unique_element_ids=True,  # UUIDs that are unique for each element
            strategy=PartitionStrategy.HI_RES,  # auto: it decides, hi_res: detectron2, but issues with multi-column, ocr_only: pytesseract, fast: pdfminer
            hi_res_model_name='yolox',
            include_page_breaks=False,
            metadata_filename=file_path,
            infer_table_structure=True,
            extract_images_in_pdf=True,
            extract_image_block_types=['Image', 'Table', 'Formula'],  # element types to save as images
            extract_image_block_to_payload=False,  # needs to be false; we'll convert into base64 later.
            extract_forms=False,  # not currently available
            extract_image_block_output_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/pdfimgs/')
        )
    
        # # 2. Check if it got good output.
        # pdf_read_in_okay = self.check_pdf_read_in(pdf_file_path=pdf_file_path, pdf_file=pdf_file, pdf_chunks=pdf_chunks)
        # if (pdf_read_in_okay):
        #     return pdf_chunks
    
        # # 3. Okay, PDF didn't read in well, so we'll use the back-up strategy
        # # According to Unstructured's Github: https://github.com/Unstructured-IO/unstructured/blob/main/unstructured/partition/pdf.py
        # # that is "OCR_ONLY" as opposed to "HI_RES".
        # pdf_chunks = partition_pdf(
        #     filename=pdf_file_path,
        #     file=pdf_file,
        #     strategy="ocr_only"  # auto: it decides, hi_res: detectron2, but issues with multi-column, ocr_only: pytesseract, fast: pdfminer
        # )
        return pdf_chunks


    def chunks_to_nodes(self, pdf_chunks: List[elements.Element]) -> List[BaseNode]:
        """
        Given a PDF from Unstructured broken by header,
        convert them into nodes using the node_parser.
        E.g., to have all sentences with similar meaning as a node, use the SemanticNodeParser
        """
        # 0. Setup.
        unstructured_chunk_nodes = []
        
        # Hash of node ID and index
        node_id_to_index = {}
    
        # 1. Convert each page's text to Nodes.
        for index, chunk in enumerate(pdf_chunks):
            # Create new node based on node type
            new_node = self._handle_chunk(chunk)

            # Update hash of node ID and index
            node_id_to_index[new_node.id_] = index

            # Add relationship to prior node
            if (len(unstructured_chunk_nodes) > 0):
                unstructured_chunk_nodes[-1], new_node = self._node_rel_prev_next(prev_node=unstructured_chunk_nodes[-1], next_node=new_node)

            # Add parent-child relationships for Title Chunks
            if (chunk.metadata.parent_id is not None):
                # Find the index of the parent node based on parent_id
                parent_index = node_id_to_index[chunk.metadata.parent_id]
                if (parent_index is not None):
                    unstructured_chunk_nodes[parent_index], new_node = self._node_rel_parent_child(parent_node=unstructured_chunk_nodes[parent_index], child_node=new_node)

            # Append to list
            unstructured_chunk_nodes.append(new_node)

        del node_id_to_index
    
        ## TODO: Move this chunk into a separate ReaderPostProcessor thing into PDFReaderUtils. Bundle in the sumamrization for tables and images into this.
        # 2. Node Parse each page to split when new information is different
        # NOTE: This was built for the Semantic Parser, but I guess we'll technically allow any parser here.
        # unstructured_parsed_nodes = self._node_parser.get_nodes_from_documents(unstructured_chunk_nodes)
    
        # 3. Node Attributes
        # for index, node in enumerate(unstructured_parsed_nodes):
        #     # Keywords and Summary
        #     # node_keywords = ', '.join(pdfrutils.get_keywords(node.text, top_k=5))
        #     # node_summary = get_t5_summary(node.text, summary_length=64)  # get_t5_summary
        #     node.metadata['keywords'] = node_keywords
        #     # node.metadata['summary'] = node_summary + (("\n" + node.metadata['summary']) if node.metadata['summary'] is not None else "")
    
        #     # Get additional information about the node.
        #     # Email: check for address.
        #     info_types = []
        #     if (pdfrutils.has_date(node.text)):
        #         info_types.append("date")
        #     if (pdfrutils.has_email(node.text)):
        #         info_types.append("contact email")
        #     if (pdfrutils.has_mail_addr(node.text)):
        #         info_types.append("mailing postal address")
        #     if (pdfrutils.has_phone(node.text)):
        #         info_types.append("contact phone")
    
        #     node.metadata['information types'] = ", ".join(info_types)
            # node.excluded_llm_metadata_keys = ['filename', 'coordinates', 'chunk_number', 'window', 'orig_nodes']
            # node.excluded_embed_metadata_keys = ['filename', 'coordinates', 'chunk_number', 'page number', 'original_text', 'window', 'keywords', 'summary', 'orig_nodes']
    
            # if (index > 0):
                # unstructured_parsed_nodes[index-1], node = self._node_rel_prev_next(unstructured_parsed_nodes[index-1], node)
        return(unstructured_chunk_nodes)

    # """Main user-interaction function"""
    def load_data(
        self, 
        file_path: Optional[str] = None,
        file: Optional[IO[bytes]] = None
    ) -> List: #[GenericNode]:
        """Given a path to a PDF file, load it with Unstructured and convert it into a list of Llamaindex Base Nodes.
        Input:
            - pdf_file_path (str): the path to the PDF file.
        Output:
            - List[GenericNode]: a list of LlamaIndex nodes. Creates one node for each parsed node, for each Unstructured Title Chunk.
        """
        # 1. PDF to Chunks
        print("NEWPDF: Reading Input File...")
        pdf_chunks = self.pdf_to_chunks(file_path=file_path, file=file)
        # return (pdf_chunks)
        
        # Chunk processing
        # pdf_chunks = clean_pdf_chunk, dedupe_title_chunks, combine_listitem_chunks, remove_header_footer_pagenum
        
        # 2. Chunks to titles
        # TODO: I hate this, make our own chunker.
        # pdf_titlechunks = chunk_by_title(
        #     pdf_chunks,
        #     max_characters=self._max_characters, 
        #     new_after_n_chars=self._new_after_n_chars,
        #     overlap=self._overlap, 
        #     overlap_all=self._overlap_all,
        #     multipage_sections=self._multipage_sections,
        #     include_orig_elements=True,
        #     combine_text_under_n_chars=self._new_after_n_chars
        # )
        # 3. Cleaning
        # pdf_titlechunks = Parallel(n_jobs=max(int(os.cpu_count())-1, 1))(  # type: ignore
        #     delayed(self.clean_pdf_chunk)(chunk) for chunk in pdf_chunks # pdf_titlechunks
        # )
        # pdf_titlechunks = list(pdf_titlechunks)
        # 4. Headlines to llamaindex nodes
        print("NEWPDF: Converting chunks to nodes...")
        parsed_chunks = self.chunks_to_nodes(pdf_chunks)
        return (parsed_chunks)