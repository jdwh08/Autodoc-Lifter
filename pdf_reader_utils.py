#####################################################
### DOCUMENT PROCESSOR [PDF READER UTILITIES]
#####################################################
# Jonathan Wang

# ABOUT: 
# This project creates an app to chat with PDFs.

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

# Detect images that are substantially overlapping according to coordinates.
# https://stackoverflow.com/questions/49897531/detect-overlapping-images-in-pil
# Keep them in the following order: no confidence score, larger image, higher confidence score

# Detect nodes whose text is substantially repeated at either the top or bottom of the page.
# Utilize the coordinates to ignore the text on the top and bottom two lines.

# Fix OCR issues with spell checking?

# Remove images that are too small in size, and overlapping with text boxes.

# Convert the List[BaseNode] -> List[BaseNode] functions into TransformComponents

#####################################################
### Imports
from __future__ import annotations

import difflib
import re
from collections import defaultdict
from copy import deepcopy
from typing import (
    TYPE_CHECKING,
    List,
    Optional,
    Tuple,
    TypeVar,
)

import rapidfuzz
import regex
from llama_index.core.schema import (
    BaseNode,
    NodeRelationship,
    RelatedNodeInfo,
)

if TYPE_CHECKING:
    from unstructured.documents import elements

#####################################################
### CODE

GenericNode = TypeVar("GenericNode", bound=BaseNode)

def clean_pdf_chunk(pdf_chunk: elements.Element) -> elements.Element:
    """Given a single element of text from a pdf read by Unstructured, clean its text."""
    ### NOTE: Don't think it's work making this a separate TransformComponent.
    # We'd still need to clean bad characters from the reader.
    chunk_text = pdf_chunk.text
    if (len(chunk_text) > 0):
        # Clean any control characters which break the language detection for other parts of the reader.
        re_bad_chars = regex.compile(r"[\p{Cc}\p{Cs}]+")
        chunk_text = re_bad_chars.sub("", chunk_text)

        # Remove PDF citations text
        chunk_text = re.sub("\\(cid:\\d+\\)", "", chunk_text)  # matches (cid:###)
        # Clean whitespace and broken paragraphs
        # chunk_text = clean_extra_whitespace(chunk_text)
        # chunk_text = group_broken_paragraphs(chunk_text)
        # Save cleaned text.
        pdf_chunk.text = chunk_text

    return pdf_chunk


def clean_abbreviations(pdf_chunks: list[GenericNode]) -> list[GenericNode]:
    """Remove any common abbreviations in the text which can confuse the sentence model.

    Args:
        pdf_chunks (List[GenericNode]): List of llama-index nodes.

    Returns:
        List[GenericNode]: The nodes with cleaned text, abbreviations replaced.
    """
    for pdf_chunk in pdf_chunks:
        text = getattr(pdf_chunk, "text", "")
        if (text == ""):
            continue
        # No. -> Number
        text = re.sub(r"\bNo\b\.\s", "Number", text, flags=re.IGNORECASE)
        # Fig. -> Figure
        text = re.sub(r"\bFig\b\.", "Figure", text, flags=re.IGNORECASE)
        # Eq. -> Equation
        text = re.sub(r"\bEq\b\.", "Equation", text, flags=re.IGNORECASE)
        # Mr. -> Mr
        text = re.sub(r"\bMr\b\.", "Mr", text, flags=re.IGNORECASE)
        # Mrs. -> Mrs
        text = re.sub(r"\bMrs\b\.", "Mrs", text, flags=re.IGNORECASE)
        # Dr. -> Dr
        text = re.sub(r"\bDr\b\.", "Dr", text, flags=re.IGNORECASE)
        # Jr. -> Jr
        text = re.sub(r"\bJr\b\.", "Jr", text, flags=re.IGNORECASE)
        # etc. -> etc
        text = re.sub(r"\betc\b\.", "etc", text, flags=re.IGNORECASE)
        pdf_chunk.text = text

    return pdf_chunks


def _remove_chunk(
    pdf_chunks: list[GenericNode],
    chunk_index: int | None=None,
    chunk_id: str | None=None
) -> list[GenericNode]:
    """Given a list of chunks, remove the chunk at the given index or with the given id.

    Args:
        pdf_chunks (List[GenericNode]): The list of chunks.
        chunk_index (Optional[int]): The index of the chunk to remove.
        chunk_id (Optional[str]): The id of the chunk to remove.

    Returns:
        List[GenericNode]: The updated list of chunks, without the removed chunk.
    """
    if (chunk_index is None and chunk_id is None):
        msg = "_remove_chunk: Either chunk_index or chunk_id must be set."
        raise ValueError(msg)

    # Convert chunk_id to chunk_index
    elif (chunk_index is None):
        chunk = next((c for c in pdf_chunks if c.node_id == chunk_id), None)
        if chunk is not None:
            chunk_index = pdf_chunks.index(chunk)
        else:
            msg = f"_remove_chunk: No chunk found with id {chunk_id}."
            raise ValueError(msg)
    elif (chunk_index < 0 or chunk_index >= len(pdf_chunks)):
        msg = f"_remove_chunk: Chunk {chunk_index} is out of range. Maximum index is {len(pdf_chunks) - 1}."
        raise ValueError(msg)

    # Update the previous-next node relationships around that index
    def _node_rel_prev_next(prev_node: GenericNode, next_node: GenericNode) -> tuple[GenericNode, GenericNode]:
        """Update pre-next node relationships between two nodes."""
        prev_node.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(
            node_id=next_node.node_id,
            metadata={"filename": next_node.metadata["filename"]}
        )
        next_node.relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(
            node_id=prev_node.node_id,
            metadata={"filename": prev_node.metadata["filename"]}
        )
        return (prev_node, next_node)

    if (chunk_index > 0 and chunk_index < len(pdf_chunks) - 1):
        pdf_chunks[chunk_index - 1], pdf_chunks[chunk_index + 1] = _node_rel_prev_next(prev_node=pdf_chunks[chunk_index - 1], next_node=pdf_chunks[chunk_index + 1])

    popped_chunk = pdf_chunks.pop(chunk_index)
    chunk_id = chunk_id or popped_chunk.node_id

    # Remove any references to the removed chunk in node relationships or metadata
    for node in pdf_chunks:
        node.relationships = {k: v for k, v in node.relationships.items() if v.node_id != chunk_id}
        node.metadata = {k: v for k, v in node.metadata.items() if ((isinstance(v, list) and (chunk_id in v)) or (v != chunk_id))}
    return pdf_chunks


def _clean_overlap_text(
    text1: str,
    text2: str,
    combining_text: str=" ",
    min_length: int | None = 1,
    max_length: int | None = 50,
    overlap_threshold: float = 0.9
) -> str:
    r"""Remove any overlapping text between two strings.

    Args:
        text1 (str): The first string.
        text2 (str): The second string.
        combining_text (str, optional): The text to combine the two strings with. Defaults to space (' '). Can also be \n.
        min_length (int, optional): The minimum length of the overlap. Defaults to 1. None is no minimum.
        max_length (int, optional): The maximum length of the overlap. Defaults to 50. None is no maximum.
        overlap_threshold (float, optional): The threshold for being an overlap. Defaults to 0.8.

    Returns:
        str: The strings combined with the overlap removed.
    """
    for overlap_len in range(min(len(text1), len(text2), (max_length or len(text1))), ((min_length or 1)-1), -1):
        end_substring = text1[-overlap_len:]
        start_substring = text2[:overlap_len]
        similarity = difflib.SequenceMatcher(None, end_substring, start_substring).ratio()
        if (similarity >= overlap_threshold):
            return combining_text.join([text1[:-overlap_len], text2[overlap_len:]]).strip()

    return combining_text.join([text1, text2]).strip()


def _combine_chunks(c1: GenericNode, c2: GenericNode) -> GenericNode:
    """Combine two chunks into one.

    Args:
        c1 (GenericNode): The first chunk.
        c2 (GenericNode): The second chunk.

    Returns:
        GenericNode: The combined chunk.
    """
    # Metadata merging
    # Type merging
    text_types = ["NarrativeText", "ListItem", "Formula", "UncategorizedText", "Composite-TextOnly"]
    image_types = ["FigureCaption", "Image"]  # things that make Image nodes.

    def _combine_chunks_type(c1_type: str, c2_type: str) -> str:
        """Combine the types of two chunks.

        Args:
            c1_type (str): The type of the first chunk.
            c2_type (str): The type of the second chunk.

        Returns:
            str: The type of the combined chunk.
        """
        if (c1_type == c2_type):
            return c1_type
        elif (c1_type in text_types and c2_type in text_types):
            return "Composite-TextOnly"
        elif (c1_type in image_types and c2_type in image_types):
            return "Image"   # Add caption to image
        else:
            return "Composite"

    c1_type = c1.metadata["type"]
    c2_type = c2.metadata["type"]
    c1.metadata["type"] = _combine_chunks_type(c1_type, c2_type)

    # All other metadata merging
    for k, v in c2.metadata.items():
        if k not in c1.metadata:
            c1.metadata[k] = v
        # Merge lists
        elif k in ["page_number", 'page_name', 'languages', 'emphasized_text_contents', 'link_texts', 'link_urls']:
            if not isinstance(c1.metadata[k], list):
                c1.metadata[k] = list(c1.metadata[k])
            if (v not in c1.metadata[k]):
                # Add to list, dedupe
                c1.metadata[k].extend(v)
                c1.metadata[k] = sorted(set(c1.metadata[k]))

    # Text merging
    c1_text = getattr(c1, "text", "")
    c2_text = getattr(c2, "text", "")
    if (c1_text == c2_text):
        # No duplicates.
        return c1
    if (c1_text == "" or c2_text == ""):
        c1.text = c1_text + c2_text
        return c1

    # Check if a sentence has been split between two chunks
    # Option 1: letters
    c1_text_last = c1_text[-1]

    # Check if c1_text_last has a lowercase letter, digit, or punctuation that doesn't end a sentence
    if (re.search(r'[\da-z\[\]\(\)\{\}\<\>\%\^\&\"\'\:\;\,\/\-\_\+\= \t\n\r]', c1_text_last)):
        # We can probably combine these two texts as if they were on the same line.
        c1.text = _clean_overlap_text(c1_text, c2_text, combining_text=" ")
    else:
        # We'll treat these as if they were on separate lines.
        c1.text = _clean_overlap_text(c1_text, c2_text, combining_text="\n")

    # NOTE: Relationships merging is handled in other functions, because it requires looking back at prior prior chunks.
    return c1

def dedupe_title_chunks(pdf_chunks: list[GenericNode]) -> list[GenericNode]:
    """Given a list of chunks, return a list of chunks without any title duplicates.

    Args:
        pdf_chunks (List[BaseNode]): The list of chunks to have titles deduped.

    Returns:
        List[BaseNode]: The deduped list of chunks.
    """
    index = 0
    while (index < len(pdf_chunks)):
        if (
            (pdf_chunks[index].metadata["type"] in ("Title")) # is title
            and (index > 0) # is not first chunk
            and (pdf_chunks[index - 1].metadata["type"] in ("Title"))  # previous chunk is also title
        ):
            # if (getattr(pdf_chunks[index], 'text', None) != getattr(pdf_chunks[index - 1], 'text', '')):
                # pdf_chunks[index].text = getattr(pdf_chunks[index - 1], 'text', '') + '\n' + getattr(pdf_chunks[index], 'text', '')
            pdf_chunks[index] = _combine_chunks(pdf_chunks[index - 1], pdf_chunks[index])

            # NOTE: We'll remove the PRIOR title, since duplicates AND child relationships are built on the CURRENT title.
            # There shouldn't be any PARENT/CHILD relationships to the title that we are deleting, so this seems fine.
            pdf_chunks = _remove_chunk(pdf_chunks=pdf_chunks, chunk_index=index-1)
            # NOTE: don't need to shift index because we removed an element.
        else:
            # We don't care about any situations other than consecutive title chunks.
            index += 1

    return (pdf_chunks)


def combine_listitem_chunks(pdf_chunks: list[GenericNode]) -> list[GenericNode]:
    """Given a list of chunks, combine any adjacent chunks which are ListItems into one List.

    Args:
        pdf_chunks (List[GenericNode]): The list of chunks to combine.

    Returns:
        List[GenericNode]: The list of chunks with ListItems combined into one List chunk.
    """
    index = 0
    while (index < len(pdf_chunks)):
        if (
            (pdf_chunks[index].metadata["type"] == "ListItem") # is list item
            and (index > 0) # is not first chunk
            and (pdf_chunks[index - 1].metadata["type"] == "ListItem")  # previous chunk is also list item
        ):
            # Okay, we have a consecutive list item. Combine into one list.
            # NOTE: We'll remove the PRIOR list item, since duplicates AND child relationships are built on the CURRENT list item.
            # 1. Append prior list item's text to the current list item's text
            # pdf_chunks[index].text = getattr(pdf_chunks[index - 1], 'text', '') + '\n' + getattr(pdf_chunks[index], 'text', '')
            pdf_chunks[index] = _combine_chunks(pdf_chunks[index - 1], pdf_chunks[index])
            # 2. Remove PRIOR list item
            pdf_chunks.pop(index - 1)
            # 3. Replace NEXT relationship from PRIOR list item with the later list item node ID, if prior prior node exists.
            if (index - 2 >= 0):
                pdf_chunks[index - 2].relationships[NodeRelationship.NEXT] = RelatedNodeInfo(
                    node_id=pdf_chunks[index].node_id,
                    metadata={"filename": pdf_chunks[index].metadata["filename"]}
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
    return (pdf_chunks)


def remove_header_footer_repeated(
    pdf_chunks_input: list[GenericNode],
    window_size: int = 3,
    fuzz_threshold: int = 80
) -> list[GenericNode]:
    """Given a list of chunks, remove any header/footer chunks that are repeated across pages.

    Args:
        pdf_chunks (List[GenericNode]): The list of chunks to process.
        window_size (int): The number of chunks to consider at the beginning and end of each page.
        fuzz_threshold (int): The threshold for fuzzy matching of chunk texts.

    Returns:
        List[GenericNode]: The list of chunks with header/footer chunks removed.
    """
    nodes_to_remove = set()   # id's to remove.
    pdf_chunks = deepcopy(pdf_chunks_input)

    # Build a dictionary of chunks by page number
    chunks_by_page = defaultdict(list)
    for chunk in pdf_chunks:
        chunk_page_number = min(chunk.metadata["page_number"]) if isinstance(chunk.metadata["page_number"], list) else chunk.metadata["page_number"]
        chunks_by_page[chunk_page_number].append(chunk)

    # Get the first window_size and last window_size chunks on each page
    header_candidates = defaultdict(set)  # hashmap of chunk text, and set of chunk ids with that text.
    footer_candidates = defaultdict(set)  # hashmap of chunk text, and set of chunk ids with that text.
    page_number_regex = re.compile(r"(?:-|\( ?)?\b(?:page|p\.?(?:[pg](?:\b|\.)?)?)? ?(?:\d+|\b[ivxm]+\b)\.?(?: ?-|\))?\b", re.IGNORECASE)
    for chunks in chunks_by_page.values():
        header_chunks = chunks[:window_size]
        footer_chunks = chunks[-window_size:]

        for chunk in header_chunks:
            chunk_text = getattr(chunk, "text", "")
            if chunk.metadata["type"] == "Header" and len(chunk_text) > 0:
                chunk_text_is_pagenum_only = page_number_regex.match(chunk_text)
                if chunk_text_is_pagenum_only and (len(chunk_text_is_pagenum_only.group(0)) == len(chunk_text)):
                    # Full match!
                    chunk.text = "Page Number Only"
                    nodes_to_remove.add(chunk.node_id)
                elif chunk_text_is_pagenum_only and len(chunk_text_is_pagenum_only.group(0)) > 0:
                    # Remove the page number content from the chunk text for this exercise
                    chunk_text = page_number_regex.sub('', chunk_text)
                    chunk.text = chunk_text

            if chunk.metadata["type"] not in ("Image", "Table") and len(chunk_text) > 0:
                header_candidates[chunk_text].add(chunk.node_id)

        for chunk in footer_chunks:
            chunk_text = getattr(chunk, "text", "")
            if chunk.metadata["type"] == "Footer" and len(chunk_text) > 0:
                chunk_text_is_pagenum_only = page_number_regex.match(chunk_text)
                if chunk_text_is_pagenum_only and (len(chunk_text_is_pagenum_only.group(0)) == len(chunk_text)):
                    # Full match!
                    chunk.text = "Page Number Only"
                    nodes_to_remove.add(chunk.node_id)
                elif chunk_text_is_pagenum_only and len(chunk_text_is_pagenum_only.group(0)) > 0:
                    # Remove the page number content from the chunk text for this exercise
                    chunk_text = page_number_regex.sub('', chunk_text)
                    chunk.text = chunk_text

            if chunk.metadata["type"] not in ("Image", "Table") and len(chunk_text) > 0:
                footer_candidates[chunk_text].add(chunk.node_id)

    # Identify any texts which are too similar to other header texts.
    header_texts = list(header_candidates.keys())
    header_distance_matrix = rapidfuzz.process.cdist(header_texts, header_texts, scorer=rapidfuzz.fuzz.ratio, score_cutoff=fuzz_threshold)

    footer_texts = list(footer_candidates.keys())
    footer_distance_matrix = rapidfuzz.process.cdist(footer_texts, footer_texts, scorer=rapidfuzz.fuzz.ratio, score_cutoff=fuzz_threshold)
    # Combine header candidates which are too similar to each other in the distance matrix
    for i in range(len(header_distance_matrix)-1):
        for j in range(i+1, len(header_distance_matrix)):
            if i == j:
                continue
            if header_distance_matrix[i][j] >= fuzz_threshold:
                header_candidates[header_texts[i]].update(header_candidates[header_texts[j]])
                header_candidates[header_texts[j]].update(header_candidates[header_texts[i]])

    for i in range(len(footer_distance_matrix)-1):
        for j in range(i+1, len(footer_distance_matrix)):
            if i == j:
                continue
            if footer_distance_matrix[i][j] >= fuzz_threshold:
                footer_candidates[footer_texts[i]].update(footer_candidates[footer_texts[j]])
                footer_candidates[footer_texts[j]].update(footer_candidates[footer_texts[i]])

    headers_to_remove = set()
    for chunk_ids in header_candidates.values():
        if len(chunk_ids) > 1:
            headers_to_remove.update(chunk_ids)

    footers_to_remove = set()
    for chunk_ids in footer_candidates.values():
        if len(chunk_ids) > 1:
            footers_to_remove.update(chunk_ids)

    nodes_to_remove = nodes_to_remove.union(headers_to_remove.union(footers_to_remove))

    for node_id in nodes_to_remove:
        pdf_chunks = _remove_chunk(pdf_chunks=pdf_chunks, chunk_id=node_id)

    return pdf_chunks

def remove_overlap_images(pdf_chunks: list[GenericNode]) -> list[GenericNode]:
    # TODO(Jonathan Wang): Implement this function to remove images which are completely overlapping each other
    # OR... get a better dang reader!
    raise NotImplementedError


def chunk_by_header(
    pdf_chunks_in: list[GenericNode],
    combine_text_under_n_chars: int = 1024,
    multipage_sections: bool = True,
# ) -> Tuple[List[GenericNode], List[GenericNode]]:
) -> list[GenericNode]:
    """Combine chunks together that are part of the same header and have similar meaning.

    Args:
        pdf_chunks (List[GenericNode]): List of chunks to be combined.

    Returns:
        List[GenericNode]: List of combined chunks.
        List[GenericNode]: List of original chunks, with node references updated.
    """
    # TODO(Jonathan Wang): Handle semantic chunking between elements within a Header chunk.
    # TODO(Jonathan Wang): Handle splitting element chunks if they are over `max_characters` in length (does this ever really happen?)
    # TODO(Jonathan Wang): Handle relationships between nodes.

    pdf_chunks = deepcopy(pdf_chunks_in)
    output = []
    id_to_index = {}
    index = 0

    # Pass 1: Combine chunks together that are part of the same title chunk.
    while (index < len(pdf_chunks)):
        chunk = pdf_chunks[index]
        if (chunk.metadata["type"] in ["Header", "Footer", "Image", "Table"]):
            # These go immediately into the semantic title chunks and also reset the new node.

            # Let's add a newline to distinguish from any other content.
            if (chunk.metadata["type"] in ["Header", "Footer", "Table"]):
                chunk.text = getattr(chunk, "text", "") + "\n"

            output.append(chunk)
            index += 1
            continue

        # Make a new node if we have a new title (or if we don't have a title).
        if (
            chunk.metadata["type"] == "Title"
        ):
            # We're good, this node can stay as a TitleChunk.
            chunk.metadata['type'] = 'Composite'
            # if (not isinstance(chunk.metadata['page number'], list)):
                # chunk.metadata['page number'] = [chunk.metadata['page number']]

            # Let's add a newline to distinguish the title from the content.
            setattr(chunk, 'text', getattr(chunk, 'text', '') + "\n")

            output.append(chunk)
            id_to_index[chunk.id_] = len(output) - 1
            index += 1
            continue
        
        elif (chunk.metadata.get('parent_id', None) in id_to_index):
            # This chunk is part of the same title as a prior chunk.
            # Add this text into the prior title node.
            jndex = id_to_index[chunk.metadata['parent_id']]

            # if (not isinstance(output[jndex].metadata['page number'], list)):
                # output[jndex].metadata['page number'] = [chunk.metadata['page number']]
            
            output[jndex] = _combine_chunks(output[jndex], chunk)
            # output[jndex].text = getattr(output[jndex], 'text', '') + '\n' + getattr(chunk, 'text', '')
            # output[jndex].metadata['page number'] = list(set(output[jndex].metadata['page number'] + [chunk.metadata['page number']]))
            # output[jndex].metadata['languages'] = list(set(output[jndex].metadata['languages'] + chunk.metadata['languages']))
            
            pdf_chunks.remove(chunk)
            continue
        
        elif (
            (chunk.metadata.get('parent_id', None) is None) 
            and (
                len(getattr(chunk, 'text', '')) > combine_text_under_n_chars  # big enough text section to stand alone
                or (len(id_to_index.keys()) <= 0)  # no prior title
            )
        ):
            # Okay, so either we don't have a title, or it was interrupted by an image / table.
            # This chunk can stay as a TextChunk.
            chunk.metadata['type'] = 'Composite-TextOnly'
            # if (not isinstance(chunk.metadata['page number'], list)):
                # chunk.metadata['page number'] = [chunk.metadata['page number']]

            output.append(chunk)
            id_to_index[chunk.id_] = len(output) - 1
            index += 1
            continue

        else:
            # Add the text to the prior node that isn't a table or image.
            jndex = len(output) - 1
            while (
                (jndex >= 0) 
                and (output[jndex].metadata['type'] in ['Table', 'Image'])
            ):
                # for title_chunk in output:
                    # print(f'''{title_chunk.id_}: {title_chunk.metadata['type']}, text: {title_chunk.text}, parent: {title_chunk.metadata['parent_id']}''')
                jndex -= 1
            
            if (jndex < 0):
                raise Exception(f'''Prior title chunk not found: {index}, {chunk.metadata.get('parent_id', None)}''')
            
            # Add this text into the prior title node.
            # if (not isinstance(output[jndex].metadata['page number'], list)):
                # output[jndex].metadata['page number'] = [chunk.metadata['page number']]
            
            output[jndex] = _combine_chunks(output[jndex], chunk)
            # output[jndex].text = getattr(output[jndex], 'text', '') + ' ' + getattr(chunk, 'text', '')
            # output[jndex].metadata['page number'] = list(set(output[jndex].metadata['page number'] + [chunk.metadata['page number']]))
            # output[jndex].metadata['languages'] = list(set(output[jndex].metadata['languages'] + chunk.metadata['languages']))
            
            pdf_chunks.remove(chunk)
            # TODO: Update relationships between nodes.
            continue
    
    return (output)


### TODO:
# Merge images together that are substantially overlapping.
# Favour image with no confidence score. (these come straight from pdf).
# Favour the larger image over the smaller one.
# Favour the image with higher confidence score.
def merge_images() -> None:
    pass
