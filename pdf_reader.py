#####################################################
# LIFT DOCUMENT INTAKE PROCESSOR [PDF READER]
#####################################################
# Jonathan Wang (jwang15, jonathan.wang@discover.com)
# Greenhouse GenAI Modeling Team

# ABOUT: 
# This project automates the processing of legal documents 
# requesting information about our customers.
# These are handled by LIFT
# (Legal, Investigation, and Fraud Team)

# This is the PDF READER.
# It converts a PDF into LlamaIndex nodes
# using UnstructuredIO.
#####################################################
# TODO Board:
# I don't think the current code is elegent... :(

# Investigate PDFPlumber as a backup/alternative for Unstructured. 
    # `https://github.com/jsvine/pdfplumber`

#####################################################
## IMPORTS
import os
import re
import regex
from typing import Any, List, IO, Optional

import numpy as np

import pdf_reader_utils as pdfrutils

# Unstructured Document Parsing
from unstructured.partition.pdf import partition_pdf
from unstructured.cleaners.core import clean_extra_whitespace, group_broken_paragraphs #, clean_ordered_bullets, clean_bullets, clean_dashes
from unstructured.chunking.title import chunk_by_title

# Llamaindex Nodes
from llama_index.core.schema import Document, BaseNode, TextNode, NodeRelationship, RelatedNodeInfo
from llama_index.core.readers.base import BaseReader
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.node_parser import NodeParser

# Parallelism for cleaning chunks
from joblib import Parallel, delayed

## Lazy Imports
# import nltk
#####################################################
class UnstructuredPDFReader():
    # Yes, we could inherit from LlamaIndex BaseReader even though I don't think it's a good idea.
    # Have you seen the Llamaindex Base Reader? It's silly. """OOP"""
        # https://docs.llamaindex.ai/en/stable/api_reference/readers/

    # here I'm basically cargo culting off the (not-very-good) pre-built Llamaindex one.
        # https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/unstructured/base.py

    # yes I do want to bind these to the class. 
    # you better not be changing the embedding model or node parser on me across different PDFReaders. that's absurd.
    embed_model: BaseEmbedding
    node_parser: NodeParser

    def __init__(
        self,
        embed_model: BaseEmbedding,  # Built on HuggingfaceEmbeddings, but you can use others.
        node_parser: NodeParser,  # Suggest using a SemanticNodeParser.
        max_characters: int = 8192, new_after_n_chars: int = 1024, overlap_n_chars: int = 128, 
        overlap: int = 128, overlap_all: bool = False, multipage_sections: bool = False, 
        # *args: Any, **kwargs: Any
    ) -> None:
        """Init params."""
        self.max_characters = max_characters
        self.new_after_n_chars = new_after_n_chars
        self.overlap_n_chars = overlap_n_chars
        self.overlap = overlap
        self.overlap_all = overlap_all
        self.multipage_sections = multipage_sections
        self.embed_model = embed_model  # set the embedding model to convert text to vector.
        self.node_parser = node_parser  # set node parser to run on each Unstructured Title Chunk
        # super().__init__(*args)  # not passing kwargs to parent bc it cannot accept it

        # Prerequisites for Unstructured.io to work
        import nltk
        try: 
            if not nltk.data.find("tokenizers/punkt"):
                nltk.download("punkt")
        except Exception as e:
            nltk.download("punkt")
        try: 
            if not nltk.data.find("taggers/averaged_perceptron_tagger"):
                nltk.download("averaged_perceptron_tagger")
        except Exception as e:
            nltk.download("averaged_perceptron_tagger")


    # """DATA LOADING FUNCTIONS"""
    def check_pdf_read_in(
        self, 
        pdf_file_path: Optional[str],
        pdf_file: Optional[IO[bytes]],
        pdf_chunks: List[BaseNode],
    ) -> bool:
        """
        Given a list of PDFs from Unstructured's "auto",
        confirm that it correctly extracted information from each page.
        """
        # Get number of pages in PDF
        from pdfminer.pdfparser import PDFParser
        from pdfminer.pdfdocument import PDFDocument
        from pdfminer.pdfpage import PDFPage
        # from pdfminer.pdfinterp import resolve1
        from pdfminer.pdftypes import resolve1
        # 0. Get number of pages in PDF
        pdf_num_pages = resolve1(
            PDFDocument(
                PDFParser(
                    open(pdf_file_path, 'rb') if pdf_file_path else pdf_file
                )
            ).catalog['Pages']
        )['Count']
    
        # Get readable content from each page.
        content_per_page = []
        current_page_text = ""
        current_page_num = 1
    
        for chunk in pdf_chunks:
            if (int(chunk.metadata.to_dict()['page_number']) == current_page_num):
                current_page_text += (("\n" + chunk.text) if current_page_text != '' else chunk.text)
            else:
                content_per_page.append(current_page_text)
                current_page_text = chunk.text
                current_page_num += 1
    
        # Append last page.
        content_per_page.append(current_page_text)
    
        # Remove redacted, cid, and any symbols.
        symbols_re = re.compile(r'[^\w\s]')
        cid_re = re.compile(r'\(?cid\:? ?\d+\)?')
    
        def _clean_page_text(text: str) -> str:
            """
            Removes any commonly added-on text like CID/Censored
            that can confuse the reader into thinking the PDF is completely readable w/ text
            instead of being an unreadable low-quality scan.
            """
            text = re.sub(symbols_re, "", text)
            text = text.lower().replace('redacted', '').replace('censored', '')
            text = re.sub(cid_re, "", text)
            text = text.strip()
            return(text)
        content_per_page = [len(_clean_page_text(text)) for text in content_per_page]
    
        # Check if number of pages with text is equal to total number of pages in document.
        pdf_num_pages_w_text = (np.array(content_per_page) > 0).sum()
        return(pdf_num_pages_w_text == pdf_num_pages)


    def clean_pdf_chunk(self, pdf_chunk):  # TODO: type hinting for input as Unstructured.Documents.Elements
        """
        Given a single element of text from a pdf read by Unstructured, clean its text.
        """
        chunk_text = pdf_chunk.text
        if (len(chunk_text) > 0):
            # Clean any control characters which break language detection
            RE_BAD_CHARS = regex.compile(r"[\p{Cc}\p{Cs}]+")
            chunk_text = RE_BAD_CHARS.sub("", chunk_text)
    
            # Remove PDF citations text
            chunk_text = re.sub("\\(cid:\\d+\\)", "", chunk_text)  # matches (cid:###)
            # Clean whitespace and broken paragraphs
            chunk_text = clean_extra_whitespace(chunk_text)
            chunk_text = group_broken_paragraphs(chunk_text)
            # Save cleaned text.
            pdf_chunk.text = chunk_text
    
        return pdf_chunk
    
    
    def pdf_to_chunks(
        self, 
        pdf_file_path: Optional[str],
        pdf_file: Optional[IO[bytes]],
    ) -> List:
        """
        Given the file path to a PDF, read it in with UnstructuredIO and return its elements.
        """
        # 1. attempt using AUTO to have it decide.
        # NOTE: this takes care of pdfminer, and also choses between using detectron2 vs tesseract only.
        # However, it sometimes gets confused by PDFs where text elements are added on later, e.g., CIDs for linking, or REDACTED
        pdf_chunks = partition_pdf(
            filename=pdf_file_path,
            file=pdf_file,
            strategy="auto"  # auto: it decides, hi_res: detectron2, but issues with multi-column, ocr_only: pytesseract, fast: pdfminer
        )
    
        # 2. Check if it got good output.
        pdf_read_in_okay = self.check_pdf_read_in(pdf_file_path=pdf_file_path, pdf_file=pdf_file, pdf_chunks=pdf_chunks)
        if (pdf_read_in_okay):
            return pdf_chunks
    
        # 3. Okay, PDF didn't read in well, so we'll use the back-up strategy
        # According to Unstructured's Github: https://github.com/Unstructured-IO/unstructured/blob/main/unstructured/partition/pdf.py
        # that is "OCR_ONLY" as opposed to "HI_RES".
        pdf_chunks = partition_pdf(
            filename=pdf_file_path,
            file=pdf_file,
            strategy="ocr_only"  # auto: it decides, hi_res: detectron2, but issues with multi-column, ocr_only: pytesseract, fast: pdfminer
        )
        return pdf_chunks
    
    
    def titlechunks_to_chunks(self, pdf_chunks: List) -> List[BaseNode]:
        """
        Given a PDF from Unstructured broken by header,
        convert them into nodes using the node_parser.
        E.g., to have all sentences with similar meaning as a node, use the SemanticNodeParser
        """
        # 0. Setup.
        unstructured_chunk_nodes = []
    
        # 1. Convert each page's text to Nodes.
        for chunk in pdf_chunks:
            # Get filename if there is one
            filename = os.path.join(chunk.metadata.file_directory, chunk.metadata.filename) if (chunk.metadata.filename is not None) else ''
            
            new_node = TextNode(
                text=chunk.text, id_=chunk.id,
                metadata={
                    'filename': filename,
                    # 'coordinates': chunk.metadata.coordinates,
                    'page number': chunk.metadata.page_number,
                    # 'index number': '',
                    # 'keywords': '', # node_keywords,
                    # 'summary': '',
                    # 'orignal_text': chunk.text,
                    # 'window': '',
                },
                excluded_llm_metadata_keys=['filename', 'coordinates', 'chunk_number', 'window'],
                excluded_embed_metadata_keys=['filename', 'coordinates', 'chunk_number', 'page number', 'original_text', 'window', 'keywords', 'summary']
            )
            # Add relationship to prior node
            new_node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
                node_id=chunk.id,
                metadata={"filename": new_node.metadata['filename']}
            )
            if (len(unstructured_chunk_nodes) > 0):
                unstructured_chunk_nodes[-1].relationships[NodeRelationship.NEXT] = RelatedNodeInfo(
                    node_id=new_node.node_id,
                    metadata={"filename": new_node.metadata['filename']}
                )
                new_node.relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(
                    node_id=unstructured_chunk_nodes[-1].node_id,
                    metadata={"filename": new_node.metadata['filename']}
                )
            unstructured_chunk_nodes.append(new_node)
    
        # 2. Node Parse each page to split when new information is different
        # NOTE: This was built for the Semantic Parser, but I guess we'll technically allow any parser here.
        unstructured_parsed_nodes = self.node_parser.get_nodes_from_documents(unstructured_chunk_nodes)
    
        # 3. Node Attributes
        for index, node in enumerate(unstructured_parsed_nodes):
            # Keywords and Summary
            node_keywords = ', '.join(pdfrutils.get_keywords(node.text, top_k=5))
            # node_summary = get_t5_summary(node.text, summary_length=64)  # get_t5_summary
            node.metadata['keywords'] = node_keywords
            # node.metadata['summary'] = node_summary + (("\n" + node.metadata['summary']) if node.metadata['summary'] is not None else "")
    
            # Get additional information about the node.
            # Email: check for address.
            info_types = []
            if (pdfrutils.has_date(node.text)):
                info_types.append("date")
            if (pdfrutils.has_email(node.text)):
                info_types.append("contact email")
            if (pdfrutils.has_mail_addr(node.text)):
                info_types.append("mailing postal address")
            if (pdfrutils.has_phone(node.text)):
                info_types.append("contact phone")
    
            node.metadata['information types'] = ", ".join(info_types)
            
            node.excluded_llm_metadata_keys = ['filename', 'coordinates', 'chunk_number', 'window', 'docreq_start', 'docreq_cosine', 'format_cosine', 'legal_cosine']
            node.excluded_embed_metadata_keys = ['filename', 'coordinates', 'chunk_number', 'page number', 'original_text', 'window', 'keywords', 'summary', 'docreq_start', 'docreq_cosine', 'format_cosine', 'legal_cosine']
            # Add embeddings
            node_embedding = self.embed_model.get_text_embedding(
                node.get_content(metadata_mode="embed")  # embed # llm
            )
            node.embedding = node_embedding
    
            if (index > 0):
                unstructured_parsed_nodes[index-1].relationships[NodeRelationship.NEXT] = RelatedNodeInfo(
                    node_id=node.node_id,
                    metadata={"filename": node.metadata['filename']}
                )
                node.relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(
                    node_id=unstructured_parsed_nodes[index-1].node_id,
                    metadata={"filename": node.metadata['filename']}
                )
    
        return(unstructured_parsed_nodes)

    # """Main user-interaction function"""
    def load_data(
        self, 
        pdf_file_path: Optional[str] = None,
        pdf_file: Optional[IO[bytes]] = None
    ) -> List[BaseNode]:
        """Given a path to a PDF file, load it with Unstructured and convert it into a list of Llamaindex Base Nodes.
        Input:
            - pdf_file_path (str): the path to the PDF file.
        Output:
            - List[BaseNode]: a list of LlamaIndex nodes. Creates one node for each parsed node, for each Unstructured Title Chunk.
        """
        # 1. PDF to Chunks
        pdf_chunks = self.pdf_to_chunks(pdf_file_path=pdf_file_path, pdf_file=pdf_file)
        # 2. Chunks to titles
        pdf_titlechunks = chunk_by_title(
            pdf_chunks, 
            max_characters=self.max_characters, 
            new_after_n_chars=self.new_after_n_chars,
            overlap=self.overlap, 
            overlap_all=self.overlap_all, 
            multipage_sections=self.multipage_sections,
            # combine_text_under_n_chars=16
        )
        # 3. Cleaning
        pdf_titlechunks = Parallel(n_jobs=max(int(os.cpu_count())-1, 1))(
            delayed(self.clean_pdf_chunk)(chunk) for chunk in pdf_titlechunks
        )
        pdf_titlechunks = list(pdf_titlechunks)
        # 4. Headlines to llamaindex nodes
        parsed_chunks = self.titlechunks_to_chunks(pdf_titlechunks)
        return (parsed_chunks)