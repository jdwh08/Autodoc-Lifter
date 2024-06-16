#####################################################
# LIFT DOCUMENT INTAKE PROCESSOR [MODELS]
#####################################################
# Jonathan Wang (jwang15, jonathan.wang@discover.com)
# Greenhouse GenAI Modeling Team

# ABOUT: 
# This project automates the processing of legal documents 
# requesting information about our customers.
# These are handled by LIFT
# (Legal, Investigation, and Fraud Team)

# This is the LANGUAGE MODELS
# that are used in the document reader.
#####################################################
## TODOS:
# Add support for vLLM / AWQ / GPTQ models.

#####################################################
## IMPORTS:
import os
from typing import List, Optional

import config

import streamlit as st

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.llms.custom import CustomLLM

## LAZY LOADING of LLMS:
## LLamacpp:
# from llama_index.llms.llama_cpp import LlamaCPP
# from llama_index.llms.llama_cpp.llama_utils import (
#     messages_to_prompt,
#     completion_to_prompt
# )

## HF Transformers LLM:
# from transformers import AutoTokenizer, BitsAndBytesConfig
# from llama_index.llms.huggingface import HuggingFaceLLM

## GROQ
# from llama_index.llms.groq import Groq

#####################################################
## CODE:
@st.cache_resource
def get_embedder(
    model_path: str = "mixedbread-ai/mxbai-embed-large-v1",
) -> BaseEmbedding:
    """Given the path to an embedding model, load it."""

    # NOTE: okay we definitely could have not made this wrapper, but shrug
    embed_model = HuggingFaceEmbedding(
        model_path
    )
    return embed_model


@st.cache_resource
def get_reranker(
    model_path: str = "mixedbread-ai/mxbai-rerank-large-v1",
    top_n: int = 5,
    device: str = 'cpu',
) -> SentenceTransformerRerank:  # technically this is a BaseNodePostprocessor, but that seems too abstract.
    """Given the path to a reranking model, load it."""

    # NOTE: okay we definitely could have not made this wrapper, but shrug
    rerank_model = SentenceTransformerRerank(
        model=model_path, 
        top_n=top_n,
        device=device
    )
    return rerank_model


# def _get_llamacpp_llm(
#     model_path: str,
#     model_seed: int = 31415926,
#     model_temperature: float = 1e-64,  # ideally 0, but HF-type doesn't allow that. # a good dev might use sys.float_info()['min']
#     model_context_length: Optional[int] = 8192,
#     model_max_new_tokens: Optional[int] = 1024,
# ) -> CustomLLM:
#     """Load a LlamaCPP model using GPU and other sane defaults."""
#     # Lazy Loading
#     from llama_index.llms.llama_cpp import LlamaCPP
#     from llama_index.llms.llama_cpp.llama_utils import (
#         messages_to_prompt,
#         completion_to_prompt
#     )

#     # Arguments to Pass
#     llm = LlamaCPP(
#         model_path=model_path,
#         temperature=model_temperature,
#         max_new_tokens=model_max_new_tokens,
#         context_window=model_context_length,
#         # kwargs to pass to __call__()
#         generate_kwargs={'seed': model_seed}, # {'temperature': TEMPERATURE, 'top_p':0.7, 'min_p':0.1, 'seed': MODEL_SEED},
#         # kwargs to pass to __init__()
#         # set to at least 1 to use GPU
#         model_kwargs={'n_gpu_layers': -1, 'n_threads': os.cpu_count()-1}, #, 'rope_freq_scale': 0.83, 'rope_freq_base': 20000},
#         # transform inputs into model format
#         messages_to_prompt=messages_to_prompt,
#         completion_to_prompt=completion_to_prompt,
#         verbose=True,
#     )
#     return (llm)


# def _get_hf_llm(
#     model_path: str = "/home/jwang15/ace-shared/ml-models/hface/meta-llama_Meta-Llama-3-8B-Instruct",
#     model_seed: int = 31415926,
#     model_temperature: float = 0,  # ideally 0, but HF-type doesn't allow that. # a good dev might use sys.float_info()['min'] to confirm (?)
#     model_context_length: Optional[int] = 8192,
#     model_max_new_tokens: Optional[int] = 1024,
#     hf_quant_level: Optional[int] = 8,
# ) -> CustomLLM:
#     """Load a Huggingface-Transformers based model using sane defaults."""
#     # Lazy Loading
#     # from torch import bfloat16
#     from transformers import AutoTokenizer, BitsAndBytesConfig
#     from llama_index.llms.huggingface import HuggingFaceLLM

#     # Get Quantization with BitsandBytes
#     bnb_config = None  # NOTE: by default, no quantization.
#     if (hf_quant_level == 4):
#         bnb_config = BitsAndBytesConfig(
#             # load_in_8bit=True,
#             load_in_4bit=True,
#             # bnb_4bit_use_double_quant=True,
#             bnb_4bit_quant_type="nf4",
#             # bnb_4bit_compute_dtype=bfloat16,  # NOTE: Our Tesla T4 GPUs are too crappy for bfloat16
#             bnb_4bit_compute_dtype='float16'
#         )
#     elif (hf_quant_level == 8):
#         bnb_config = BitsAndBytesConfig(
#             load_in_8bit=True
#         )

#     # Get Stopping Tokens for Llama3 based models, because they're /special/ and added a new one.
#     tokenizer = AutoTokenizer.from_pretrained(
#         model_path
#     )
#     stopping_ids = [
#         tokenizer.eos_token_id,
#         tokenizer.convert_tokens_to_ids("<|eot_id|>"),
#     ]
#     llm = HuggingFaceLLM(
#         model_name=model_path,
#         tokenizer_name=model_path,
#         stopping_ids=stopping_ids,
#         tokenizer_kwargs={'trust_remote_code': True},
#         model_kwargs={'trust_remote_code': True, 'quantization_config': bnb_config},
#         generate_kwargs={
#             'max_new_tokens': model_max_new_tokens,
#             'do_sample': False if model_temperature > 0 else True, 
#             'temperature': model_temperature,
#         }
#     )
#     return (llm)


@st.cache_resource
def get_llm() -> CustomLLM:
    from llama_index.llms.groq import Groq

    llm = Groq(
        model='llama3-8b-8192',
        api_key=config.groq_api_key
    )
    return (llm)

# def get_llm(
#     model_path: str = "../../models/mistral-7b-instruct-v0.2.Q5_K_M.gguf",
#     model_seed: int = 31415926,
#     model_temperature: float = 0,  # ideally 0, but HF-type doesn't allow that. # a good dev might use sys.float_info()['min']
#     model_context_length: Optional[int] = 8192,
#     model_max_new_tokens: Optional[int] = 1024,

#     hf_quant_level: Optional[int] = None,  # 4-bit / 8-bit loading for HF models
# ) -> CustomLLM:
#     """
#     Given the path to a LLM, determine the type, load it in and convert it into a Llamaindex-compatable LLM.
#     NOTE: I chose to set some "sane" defaults, so it's probably not as flexible as some other dev would like
#     """
#     model_path_extension = model_path[model_path.rindex('.'): ].lower()
    
#     if (model_path_extension == ".gguf"):
#         ##### LLAMA.CPP
#         return(_get_llamacpp_llm(model_path, model_seed, model_temperature, model_context_length, model_max_new_tokens))

#     # TODO:
#     # vLLM support for AWQ/GPTQ models
#     # I guess reluctantly AutoAWQ and AutoGPTQ packages.
#     # Exllamav2 is kinda dead IMO.

#     else:
#         #### No extension or weird fake extension suggests a folder, i.e., the base model from HF
#         return(_get_hf_llm(model_path, model_seed, model_temperature, model_context_length, model_max_new_tokens, hf_quant_level))