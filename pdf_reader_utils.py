#####################################################
# LIFT DOCUMENT INTAKE PROCESSOR [PDF READER UTILS]
#####################################################
# Jonathan Wang (jwang15, jonathan.wang@discover.com)
# Greenhouse GenAI Modeling Team

# ABOUT: 
# This project automates the processing of legal documents 
# requesting information about our customers.
# These are handled by LIFT
# (Legal, Investigation, and Fraud Team)

# This is the PDF READER UTILITIES.
# It defines helper functions for the PDF reader,
# such as getting Keywords or finding Contact Info.
#####################################################
### TODO Board:
# Better Summarizer than T5, which has been stripped out?
# Better keywords than the RAKE+YAKE fusion we're currently using?
# Consider using GPE/GSP tagging with spacy to confirm mailing addresses?

#####################################################
### Imports
from typing import List

import os
import re

# Keywords
from multi_rake import Rake
import yake
from merger import _merge_on_scores

# Summarizer (NOT DONE!)

#####################################################
### Constants
# ah how beautiful the regex
# handy visualizer and checker: https://www.debuggex.com/, https://www.regexpr.com/

DATE_REGEX = re.compile('(?:(?<!\:)(?<!\:\d)[0-3]?\d(?:st|nd|rd|th)?\s+(?:of\s+)?(?:jan\.?|january|feb\.?|february|mar\.?|march|apr\.?|april|may|jun\.?|june|jul\.?|july|aug\.?|august|sep\.?|september|oct\.?|october|nov\.?|november|dec\.?|december)|(?:jan\.?|january|feb\.?|february|mar\.?|march|apr\.?|april|may|jun\.?|june|jul\.?|july|aug\.?|august|sep\.?|september|oct\.?|october|nov\.?|november|dec\.?|december)\s+(?<!\:)(?<!\:\d)[0-3]?\d(?:st|nd|rd|th)?)(?:\,)?\s*(?:\d{4})?|[0-3]?\d[-\./][0-3]?\d[-\./]\d{2,4}', re.IGNORECASE)
# TIME_REGEX = re.compile('\d{1,2}:\d{2} ?(?:[ap]\.?m\.?)?|\d[ap]\.?m\.?', re.IGNORECASE)
REL_DATE_REGEX = re.compile('\d{0,3} (?:day|week|month)')

EMAIL_REGEX = re.compile(r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)")
PHONE_REGEX = re.compile('((?:(?<![\d-])(?:\+?\d{1,3}[-.\s*]?)?(?:\(?\d{3}\)?[-.\s*]?)?\d{3}[-.\s*]?\d{4}(?![\d-]))|(?:(?<![\d-])(?:(?:\(\+?\d{2}\))|(?:\+?\d{2}))\s*\d{2}\s*\d{3}\s*\d{4}(?![\d-])))')
MAIL_ADDR_REGEX = re.compile('\d{1,4}.{1,10}[\w\s]{1,20}[\s]+(?:street|st|avenue|ave|road|rd|highway|hwy|square|sq|trail|trl|drive|dr|court|ct|parkway|pkwy|circle|cir|boulevard|blvd)\W?(?=\s|$)', re.IGNORECASE)

#####################################################
## FUNCTIONS
def has_email(input_text: str) -> bool:
    """
    Given a chunk of text, determine whether it has an email address or not.

    We're using the long complex email regex from https://emailregex.com/index.html
    but supposedly for LIFT there is a bijective relationship between @ and email address.
    i.e., @ only occurs for email, and all emails have @
    """
    return (EMAIL_REGEX.search(input_text) is not None)


def has_phone(input_text: str) -> bool:
    """
    Given a chunk of text, determine whether it has a phone number or not.
    """
    has_phone = PHONE_REGEX.search(input_text)
    return (has_phone is not None)


def has_mail_addr(input_text: str) -> bool:
    """
    Given a chunk of text, determine whether it has a mailing address or not.

    NOTE: This is difficult to do with regex.
        ... We could use spacy's English language NER model instead / as well:
        Assume that addresses will have a GSP (geospatial political) or GPE (geopolitical entity).
        DOCS SEE: https://www.nltk.org/book/ch07.html | https://spacy.io/usage/linguistic-features
    """
    has_addr = MAIL_ADDR_REGEX.search(input_text)
    return (has_addr is not None)


def has_date(input_text: str) -> bool:
    """
    Given a chunk of text, determine whether it has a date or not.
    NOTE: relative dates are stuff like "within 30 days"
    """
    has_date = DATE_REGEX.search(input_text)
    has_relative_date = REL_DATE_REGEX.search(input_text)
    return (has_date is not None and has_relative_date is not None)


def get_keywords(input_text: str, top_k: int=5) -> List[str]:
    """
    Given a string, get its keywords using RAKE+YAKE w/ Distribution Based Fusion.  
    Inputs:  
        input_text (str): the input text to get keywords from  
        top_k (int): the number of keywords to get  

    Returns:  
        List[str]: A list of the keywords
    """

    # RAKE
    kw_extractor = Rake()
    keywords_rake = kw_extractor.apply(input_text)
    keywords_rake = dict(keywords_rake)
    kws_rake = list(keywords_rake.keys())[:top_k]

    # YAKE
    kw_extractor = yake.KeywordExtractor(lan="en", dedupLim=0.9, n=3)
    keywords_yake = kw_extractor.extract_keywords(input_text)
    # reorder scores so that higher is better
    keywords_yake = {keyword[0].lower(): (1 - keyword[1]) for keyword in keywords_yake}
    keywords_yake = dict(sorted(keywords_yake.items(), key=lambda x: x[1], reverse=True))

    # Merge RAKE and YAKE based on scores.
    keywords_merged = _merge_on_scores(list(keywords_yake.keys()), list(keywords_rake.keys()), list(keywords_yake.values()), list(keywords_rake.values()), a_weight=0.5, top_k=top_k)

    # return (list(keywords_rake.keys())[:top_k], list(keywords_yake.keys())[:top_k], keywords_merged)
    return (keywords_merged)