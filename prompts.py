#####################################################
# LIFT DOCUMENT INTAKE PROCESSOR [PROMPTS]
#####################################################
# Jonathan Wang (jwang15, jonathan.wang@discover.com)
# Greenhouse GenAI Modeling Team

# ABOUT: 
# This project automates the processing of legal documents 
# requesting information about our customers.
# These are handled by LIFT
# (Legal, Investigation, and Fraud Team)

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
def get_qa_prompt(
    prompt_file_path: str
) -> PromptTemplate:
    """Given a path to the prompts, get prompt for Question-Answering"""
    prompts = pd.read_csv(prompt_file_path)
    # https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/prompts/default_prompts.py
    custom_qa_prompt = PromptTemplate(
        prompts.at[0, 'Prompt']
    )
    return (custom_qa_prompt)


def get_refine_prompt(
    prompt_file_path: str
) -> PromptTemplate:
    """Given a path to the prompts, get prompt to Refine answer after new info"""
    prompts = pd.read_csv(prompt_file_path)
    # https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/prompts/default_prompts.py
    custom_refine_prompt = PromptTemplate(
        prompts.at[1, 'Prompt']
    )
    return (custom_refine_prompt)


def get_reqdoc_prompt(
    prompt_file_path: str
) -> PromptTemplate:
    """Given a path to the prompts, get prompt to identify requested info from document."""
    prompts = pd.read_csv(prompt_file_path)
    # https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/prompts/default_prompts.py
    reqdoc_prompt = PromptTemplate(
        prompts.at[2, 'Prompt']
    )
    return (reqdoc_prompt)