#####################################################
### DOCUMENT PROCESSOR [PROMPTS]
#####################################################
# Jonathan Wang

# ABOUT: 
# This project creates an app to chat with PDFs.

# This is the prompts sent to the LLM.
#####################################################
## TODOS:
# Use the row names instead of .at indesx locators
# This is kinda dumb because we read the same .csv file over again
    # Should we structure this abstraction differently?

#####################################################
## IMPORTS:
import pandas as pd
from llama_index.core import PromptTemplate

#####################################################
## CODE:

# https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/prompts/default_prompts.py
QA_PROMPT = """Context information is below.\n
---------------------
{context_str}
---------------------
Given the context information, answer the query.
You must adhere to the following rules:
- Use the context information, not prior knowledge.
- End the answer with any brief quote(s) from the context that are the most essential in answering the question. 
    - If the context is not helpful in answering the question, do not include a quote.

Query: {query_str}
Answer: """

# https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/prompts/default_prompts.py
REFINE_PROMPT = """The original query is as follows: {query_str}
We have provided an existing answer: {existing_answer}
We have the opportunity to refine the existing answer (only if needed) with some more context below.
---------------------
{context_msg}
---------------------
Given the new context, refine the original answer to better answer the query.
You must adhere to the following rules:
- If the context isn't useful, return the original answer.
- End the answer with any brief quote(s) from the original answer or new context that are the most essential in answering the question. 
    - If the new context is not helpful in answering the question, leave the original answer unchanged.

Refined Answer: """

def get_qa_prompt(
    # prompt_file_path: str
) -> PromptTemplate:
    """Given a path to the prompts, get prompt for Question-Answering"""
    # prompts = pd.read_csv(prompt_file_path)
    # https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/prompts/default_prompts.py
    custom_qa_prompt = PromptTemplate(
        QA_PROMPT
    )
    return (custom_qa_prompt)


def get_refine_prompt(
    # prompt_file_path: str
) -> PromptTemplate:
    """Given a path to the prompts, get prompt to Refine answer after new info"""
    # prompts = pd.read_csv(prompt_file_path)
    # https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/prompts/default_prompts.py
    custom_refine_prompt = PromptTemplate(
        REFINE_PROMPT
    )
    return (custom_refine_prompt)


# def get_reqdoc_prompt(
#     prompt_file_path: str
# ) -> PromptTemplate:
#     """Given a path to the prompts, get prompt to identify requested info from document."""
#     prompts = pd.read_csv(prompt_file_path)
#     # https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/prompts/default_prompts.py
#     reqdoc_prompt = PromptTemplate(
#         prompts.at[2, 'Prompt']
#     )
#     return (reqdoc_prompt)