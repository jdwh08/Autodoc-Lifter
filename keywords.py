#####################################################
### DOCUMENT PROCESSOR [Keywords]
#####################################################
### Jonathan Wang

# ABOUT: 
# This creates an app to chat with PDFs.

# This is the Keywords
# Which creates keywords based on documents.
#####################################################
### TODO Board:
# TODO(Jonathan Wang): Add Maximum marginal relevance to the merger for better keywords.
# TODO(Jonathan Wang): create own version of Rake keywords

#####################################################
### PROGRAM SETTINGS


#####################################################
### PROGRAM IMPORTS
from __future__ import annotations

from typing import Any, Callable, Optional

# Keywords
# from multi_rake import Rake  # removing because of compile issues and lack of maintainence
import yake
from llama_index.core.bridge.pydantic import Field
from llama_index.core.schema import BaseNode

# Own Modules
from metadata_adder import MetadataAdder

#####################################################
### SCRIPT

def get_keywords(input_text: str) -> str:
    """
    Given a string, get its keywords using RAKE+YAKE w/ Distribution Based Fusion.

    Inputs:
        input_text (str): the input text to get keywords from
        # top_k (int): the number of keywords to get

    Returns:
        str: A list of the keywords, joined into a string.
    """
    # RAKE
    # kw_extractor = Rake()
    # keywords_rake = kw_extractor.apply(input_text)
    # keywords_rake = dict(keywords_rake)
    # YAKE
    kw_extractor = yake.KeywordExtractor(lan="en", dedupLim=0.9, n=3)
    keywords_yake = kw_extractor.extract_keywords(input_text)
    # reorder scores so that higher is better
    keywords_yake = {keyword[0].lower(): (1 - keyword[1]) for keyword in keywords_yake}
    keywords_yake = dict(
        sorted(keywords_yake.items(), key=lambda x: x[1], reverse=True)  # type hinting YAKE is miserable
        )

    # Merge RAKE and YAKE based on scores.
    # keywords_merged = _merge_on_scores(
    #     list(keywords_yake.keys()), 
    #     list(keywords_rake.keys()), 
    #     list(keywords_yake.values()), 
    #     list(keywords_rake.values()), 
    #     a_weight=0.5, 
    #     top_k=top_k
    # )

    # return (list(keywords_rake.keys())[:top_k], list(keywords_yake.keys())[:top_k], keywords_merged)
    return ", ".join(keywords_yake)  # kinda regretting forcing this into a string


class KeywordMetadataAdder(MetadataAdder):
    """Adds keyword metadata to a document.

    Args:
        metadata_name: The name of the metadata to add to the document. Defaults to 'keyword_metadata'.
        keywords_function: A function for keywords, given a source string and the number of keywords to get.
    """

    keywords_function: Callable[[str, int], str] = Field(
        description="The function to use to extract keywords from the text. Input is string and number of keywords to extract. Ouptut is string of keywords.",
        default=get_keywords,
    )
    num_keywords: int = Field(
        default=5,
        description="The number of keywords to extract from the text. Defaults to 5.",
    )

    def __init__(
        self,
        metadata_name: str = "keyword_metadata",
        keywords_function: Callable[[str], str] = get_keywords,
        num_keywords: int = 5,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        super().__init__(metadata_name=metadata_name, keywords_function=keywords_function, num_keywords=num_keywords, **kwargs)  # ah yes i love oop :)

    @classmethod
    def class_name(cls) -> str:
        return "KeywordMetadataAdder"

    def get_node_metadata(self, node: BaseNode) -> str | None:
        if not hasattr(node, "text") or node.text is None:
            return None
        return self.keywords_function(node.get_content(), self.num_keywords)
