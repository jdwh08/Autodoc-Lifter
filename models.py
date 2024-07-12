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
import logging
from typing import Any, Callable, Dict, List, Union, Optional, Sequence, Tuple, cast
from PIL import Image as PILImage

import config

import streamlit as st

import gc
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, StoppingCriteria, StoppingCriteriaList
from wtpsplit import SaT  # Sentence segmentation model. Yes. Really.

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.schema import ImageNode, ImageDocument
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    MessageRole,
    LLMMetadata
)
from llama_index.core.llms.callbacks import (
    llm_chat_callback,
    llm_completion_callback,
)

from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.types import BaseOutputParser, PydanticProgramMode, Thread

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import SentenceTransformerRerank

from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.multi_modal_llms import MultiModalLLM, MultiModalLLMMetadata
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core.callbacks import CallbackManager

from llama_index.core.bridge.pydantic import Field, PrivateAttr

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
logger = logging.getLogger(__name__)

@st.cache_resource
def get_sat_sentence_splitter(
    model_name: str = "sat-3l-sm"  # segment anything
) -> SaT:
    """Given the path to a sentence segmentation model, load it."""

    # NOTE: okay we definitely could have not made this wrapper, but shrug
    sat = SaT(model_name)
    return sat


@st.cache_resource
def get_embedder(
    model_path: str = "mixedbread-ai/mxbai-embed-large-v1",
    device: str = 'cuda', # 'cpu',
) -> BaseEmbedding:
    """Given the path to an embedding model, load it."""

    # NOTE: okay we definitely could have not made this wrapper, but shrug
    embed_model = HuggingFaceEmbedding(
        model_path,
        device=device
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


## LLM Options Below

def _get_llamacpp_llm(
    model_path: str,
    model_seed: int = 31415926,
    model_temperature: float = 1e-64,  # ideally 0, but HF-type doesn't allow that. # a good dev might use sys.float_info()['min']
    model_context_length: Optional[int] = 8192,
    model_max_new_tokens: Optional[int] = 1024,
) -> BaseLLM:
    """Load a LlamaCPP model using GPU and other sane defaults."""
    # Lazy Loading
    from llama_index.llms.llama_cpp import LlamaCPP
    from llama_index.llms.llama_cpp.llama_utils import (
        messages_to_prompt,
        completion_to_prompt
    )

    # Arguments to Pass
    llm = LlamaCPP(
        model_path=model_path,
        temperature=model_temperature,
        max_new_tokens=model_max_new_tokens,
        context_window=model_context_length,
        # kwargs to pass to __call__()
        generate_kwargs={'seed': model_seed}, # {'temperature': TEMPERATURE, 'top_p':0.7, 'min_p':0.1, 'seed': MODEL_SEED},
        # kwargs to pass to __init__()
        # set to at least 1 to use GPU
        model_kwargs={'n_gpu_layers': -1, 'n_threads': os.cpu_count()-1}, #, 'rope_freq_scale': 0.83, 'rope_freq_base': 20000},
        # transform inputs into model format
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        verbose=True,
    )
    return (llm)


def _get_hf_llm(
    model_path: str = "/home/jwang15/ace-shared/ml-models/hface/meta-llama_Meta-Llama-3-8B-Instruct",
    model_seed: int = 31415926,
    model_temperature: float = 0,  # ideally 0, but HF-type doesn't allow that. # a good dev might use sys.float_info()['min'] to confirm (?)
    model_context_length: Optional[int] = 8192,
    model_max_new_tokens: Optional[int] = 1024,
    hf_quant_level: Optional[int] = 8,
) -> BaseLLM:
    """Load a Huggingface-Transformers based model using sane defaults."""
    # Lazy Loading
    # from torch import bfloat16
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from llama_index.llms.huggingface import HuggingFaceLLM

    # Get Quantization with BitsandBytes
    bnb_config = None  # NOTE: by default, no quantization.
    if (hf_quant_level == 4):
        bnb_config = BitsAndBytesConfig(
            # load_in_8bit=True,
            load_in_4bit=True,
            # bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            # bnb_4bit_compute_dtype=bfloat16,  # NOTE: Our Tesla T4 GPUs are too crappy for bfloat16
            bnb_4bit_compute_dtype='float16'
        )
    elif (hf_quant_level == 8):
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True
        )

    # Get Stopping Tokens for Llama3 based models, because they're /special/ and added a new one.
    tokenizer = AutoTokenizer.from_pretrained(
        model_path
    )
    stopping_ids = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]
    llm = HuggingFaceLLM(
        model_name=model_path,
        tokenizer_name=model_path,
        stopping_ids=stopping_ids,
        tokenizer_kwargs={'trust_remote_code': True},
        model_kwargs={'trust_remote_code': True, 'quantization_config': bnb_config},
        generate_kwargs={
            'max_new_tokens': model_max_new_tokens,
            'do_sample': False if model_temperature > 0 else True, 
            'temperature': model_temperature,
        },
        is_chat_model=True,
    )
    return (llm)


# @st.cache_resource
# def get_llm(
#     model_path: str = "../../models/mistral-7b-instruct-v0.2.Q5_K_M.gguf",
#     model_seed: int = 31415926,
#     model_temperature: float = 0,  # ideally 0, but HF-type doesn't allow that. # a good dev might use sys.float_info()['min']
#     model_context_length: Optional[int] = 8192,
#     model_max_new_tokens: Optional[int] = 1024,

#     hf_quant_level: Optional[int] = None,  # 4-bit / 8-bit loading for HF models
# ) -> BaseLLM:
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
#          #### No extension or weird fake extension suggests a folder, i.e., the base model from HF
#          return(_get_hf_llm(model_path, model_seed, model_temperature, model_context_length, model_max_new_tokens, hf_quant_level))


@st.cache_resource
def get_llm() -> BaseLLM:
    from llama_index.llms.groq import Groq

    llm = Groq(
        model='llama3-8b-8192',
        api_key=config.groq_api_key
    )
    return (llm)


DEFAULT_HF_MULTIMODAL_LLM = 'microsoft/Phi-3-vision-128k-instruct'
DEFAULT_HF_MULTIMODAL_CONTEXT_WINDOW = 500
DEFAULT_HF_MULTIMODAL_MAX_NEW_TOKENS = 1000

class HuggingFaceMultiModalLLM(MultiModalLLM):
    model_name: str = Field(
        description='The multi-modal huggingface LLM to use. Currently only using Phi3.',
        default=DEFAULT_HF_MULTIMODAL_LLM
    )
    context_window: int = Field(
        default=DEFAULT_HF_MULTIMODAL_CONTEXT_WINDOW,
        description="The maximum number of tokens available for input.",
        gt=0,
    )
    max_new_tokens: int = Field(
        default=DEFAULT_HF_MULTIMODAL_MAX_NEW_TOKENS,
        description="The maximum number of tokens to generate.",
        gt=0,
    )
    system_prompt: str = Field(
        default="",
        description=(
            "The system prompt, containing any extra instructions or context. "
            "The model card on HuggingFace should specify if this is needed."
        ),
    )
    query_wrapper_prompt: PromptTemplate = Field(
        default=PromptTemplate("{query_str}"),
        description=(
            "The query wrapper prompt, containing the query placeholder. "
            "The model card on HuggingFace should specify if this is needed. "
            "Should contain a `{query_str}` placeholder."
        ),
    )
    processor_name: str = Field(
        default=DEFAULT_HF_MULTIMODAL_LLM,
        description=(
            "The name of the processor to use from HuggingFace. "
            "Unused if `processor` is passed in directly."
        ),
    )
    device_map: str = Field(
        default="auto", description="The device_map to use. Defaults to 'auto'."
    )
    stopping_ids: List[int] = Field(
        default_factory=list,
        description=(
            "The stopping ids to use. "
            "Generation stops when these token IDs are predicted."
        ),
    )
    tokenizer_outputs_to_remove: list = Field(
        default_factory=list,
        description=(
            "The outputs to remove from the tokenizer. "
            "Sometimes huggingface tokenizers return extra inputs that cause errors."
        ),
    )
    processor_kwargs: dict = Field(
        default_factory=dict, description="The kwargs to pass to the processor."
    )
    model_kwargs: dict = Field(
        default_factory=dict,
        description="The kwargs to pass to the model during initialization.",
    )
    generate_kwargs: dict = Field(
        default_factory=dict,
        description="The kwargs to pass to the model during generation.",
    )
    is_chat_model: bool = Field(
        default=False,
        description=(
            "Whether the model can have multiple messages passed at once, like the OpenAI chat API. This is almost certainly no."
            # LLMMetadata.__fields__["is_chat_model"].field_info.description
            # + " Be sure to verify that you either pass an appropriate tokenizer "
            # "that can convert prompts to properly formatted chat messages or a "
            # "`messages_to_prompt` that does so."
        ),
    )
    
    _model: AutoModelForCausalLM = PrivateAttr()
    _processor: AutoProcessor = PrivateAttr()
    _stopping_criteria: Any = PrivateAttr()

    def __init__(
        self, 
        context_window: int = DEFAULT_HF_MULTIMODAL_CONTEXT_WINDOW,
        max_new_tokens: int = DEFAULT_HF_MULTIMODAL_MAX_NEW_TOKENS,
        query_wrapper_prompt: Union[str, PromptTemplate] = "{query_str}",
        processor_name: str = DEFAULT_HF_MULTIMODAL_LLM,
        model_name: str = DEFAULT_HF_MULTIMODAL_LLM,
        model: Optional[AutoModelForCausalLM] = None,
        processor: Optional[AutoProcessor] = None,
        device_map: str = 'auto',
        stopping_ids: Optional[List[int]] = None,
        processor_kwargs: Optional[dict] = {},
        tokenizer_outputs_to_remove: Optional[list]= None,
        model_kwargs: Optional[dict] = {},
        generate_kwargs: Optional[dict] = {},
        is_chat_model: Optional[bool] = False,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: str = "",
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage], int], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        output_parser: Optional[BaseOutputParser] = None,
    ) -> None:
        self._model = model or AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            trust_remote_code=True,
            **model_kwargs
        )
        
        # check context_window
        config_dict = self._model.config.to_dict()
        model_context_window = int(
            config_dict.get("max_position_embeddings", context_window)
        )
        if ((model_context_window is not None) and (model_context_window < context_window)):
            logger.warning(
                f"Supplied context_window {context_window} is greater "
                f"than the model's max input size {model_context_window}. "
                "Disable this warning by setting a lower context_window."
            )
            context_window = model_context_window
        
        if ((processor_kwargs is None) or ("max_length" not in processor_kwargs)):
            processor_kwargs = processor_kwargs or {}
            processor_kwargs["max_length"] = context_window

        self._processor = processor or AutoProcessor.from_pretrained(
            processor_name or model_name,
            trust_remote_code=True,
            **processor_kwargs
        )

        # Processor-Model disagreement
        if (self._processor.tokenizer.name_or_path != model_name):
            logger.warning(
                f"The model `{model_name}` and processor `{self._processor.tokenizer.name_or_path}` "
                f"are different, please ensure that they are compatible."
            )
        
        # setup stopping criteria
        stopping_ids_list = stopping_ids or []

        class StopOnTokens(StoppingCriteria):
            def __call__(
                self,
                input_ids: torch.LongTensor,
                scores: torch.FloatTensor,
                **kwargs: Any,
            ) -> bool:
                for stop_id in stopping_ids_list:
                    if input_ids[0][-1] == stop_id:
                        return True
                return False

        self._stopping_criteria = StoppingCriteriaList([StopOnTokens()])

        if isinstance(query_wrapper_prompt, str):
            query_wrapper_prompt = PromptTemplate(query_wrapper_prompt)

        messages_to_prompt = messages_to_prompt or self._messages_to_prompt
        
        super().__init__(
            context_window=context_window,
            max_new_tokens=max_new_tokens,
            query_wrapper_prompt=query_wrapper_prompt,
            tokenizer_name=processor_name,
            model_name=model_name,
            device_map=device_map,
            stopping_ids=stopping_ids or [],
            tokenizer_kwargs=processor_kwargs or {},
            tokenizer_outputs_to_remove=tokenizer_outputs_to_remove or [],
            model_kwargs=model_kwargs or {},
            generate_kwargs=generate_kwargs or {},
            is_chat_model=is_chat_model,
            callback_manager=callback_manager,
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
        )
        return (None)

    @classmethod
    def class_name(cls) -> str:
        return "HuggingFace_MultiModal_LLM"

    @property
    def metadata(self) -> LLMMetadata:
        """LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.max_new_tokens,
            model_name=self.model_name,
            is_chat_model=self.is_chat_model,
        )
    
    def _message_to_prompt(self, message: str, num_images: int) -> str:
        ### TODO: Make this work generically, not just for Phi-3.
        """Converts a list of messages into a prompt for Phi-3, handling the image placeholder tags.
        NOTE: we assume for simplicity here that these images are related, and not the user bouncing between multiple different topics. Thus, we send them all at once.

        Args:
            messages (List[dict]): A list of the messages to convert, where each message is a dict containing the message role and content.
            num_images (int): The number of images the user is passing to the MultiModalLLM.
        Returns:
            str: The prompt for Phi-3.
        """
        # For simplicity, we will send all images the first time the user speaks in the conversation.
        message_with_images = ""
        for image_index in range(1, num_images+1):
            message_with_images += f"<|image_{image_index}|>\n"
        message_with_images += message
        messages_list = [
            {'role': 'user', 'content': message_with_images}
        ]
        prompt = self._processor.tokenizer.apply_chat_template(messages_list, tokenize=False, add_generation_prompt=True)  # type: ignore
        # need to remove last <|endoftext|> if it is there, which is used for training, not inference. For training, make sure to add <|endoftext|> in the end.
        if prompt.endswith("<|endoftext|>"):
            prompt = prompt.rstrip("<|endoftext|>")
        return (prompt)

    
    def _messages_to_prompt(self, messages: Sequence[ChatMessage], num_images: int) -> str:
        ### TODO: Make this work generically, not just for Phi-3.
        """Converts a list of messages into a prompt for Phi-3, handling the image placeholder tags.
        NOTE: we assume for simplicity here that these images are related, and not the user bouncing between multiple different topics. Thus, we send them all at once.

        Args:
            messages (List[dict]): A list of the messages to convert, where each message is a dict containing the message role and content.
            num_images (int): The number of images the user is passing to the MultiModalLLM.
        Returns:
            str: The prompt for Phi-3.
        """
        # For simplicity, we will send all images the first time the user speaks in the conversation.
        image_placeholders = ""
        for image_index in range(1, num_images+1):
            image_placeholders += f"<|image_{image_index}|>\n"

        messages_dict = [
            {"role": message.role.value, "content": message.content}
            for message in messages
        ]

        # NOTE: We add the image placeholders to the first user-provided chat message.
        def _get_first_user_message_index(messages_dict: List[Dict]) -> Optional[int]:
            """Get the index of the first user message in the message list"""
            for index, message in enumerate(messages):
                if (message['role'] == MessageRole.USER):
                    return (index)
            return (None)

        first_user_message_index = _get_first_user_message_index(messages_dict)
        messages_dict[first_user_message_index]['content'] = image_placeholders + messages_dict[first_user_message_index]['content']
        
        prompt = self._processor.tokenizer.apply_chat_template(messages_list, tokenize=False, add_generation_prompt=True)  # type: ignore
        # need to remove last <|endoftext|> if it is there, which is used for training, not inference. For training, make sure to add <|endoftext|> in the end.
        if prompt.endswith("<|endoftext|>"):
            prompt = prompt.rstrip("<|endoftext|>")
        return (prompt)

    
    @llm_completion_callback()
    def complete(
        self,
        prompt: str,
        image_documents: ImageNode | List[ImageNode],  # this also takes ImageDocument which inherits from ImageNode.
        formatted: bool = False,
        **kwargs: Any
    ) -> CompletionResponse:
        """Given a prompt and image node(s), get the Phi-3 Vision prompt"""
        
        # Handle images input
        image_list = []
        if (not isinstance(image_documents, list)):
            image_documents = [image_documents]
        
        # Convert input images into PIL images for the model.
        for image in image_documents:
            # NOTE: ImageDocument inherets from ImageNode. We'll go extract the image.
            image_io = image.resolve_image()
            image_pil = PILImage.open(image_io)
            image_list.append(image_pil)
        
        num_images = len(image_list)
        
        # Get the prompt
        prompt = self._message_to_prompt(prompt, num_images)

        full_prompt = prompt
        if not formatted:
            if self.query_wrapper_prompt:
                full_prompt = self.query_wrapper_prompt.format(query_str=prompt)
            if self.system_prompt:
                full_prompt = f"{self.system_prompt} {full_prompt}"

        # Get the model input
        model_inputs = self._processor(prompt, image_list, return_tensors="pt").to(self._model.device)
        gc.collect()
        torch.cuda.empty_cache()

        # remove keys from the tokenizer if needed, to avoid HF errors
        for key in self.tokenizer_outputs_to_remove:
            if key in model_inputs:
                model_inputs.pop(key, None)

        # Get output
        tokens = self._model.generate(
            **model_inputs, 
            max_new_tokens=self.max_new_tokens,
            stopping_criteria=self._stopping_criteria,
            eos_token_id=self._processor.tokenizer.eos_token_id,
            **self.generate_kwargs
        )
        gc.collect()
        torch.cuda.empty_cache()

        completion_tokens = tokens[:, model_inputs['input_ids'].shape[1]:]
        completion = self._processor.batch_decode(
            completion_tokens, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        gc.collect()
        torch.cuda.empty_cache()

        output = CompletionResponse(text=completion, raw={'model_output': tokens})

        # Clean stuff up
        del model_inputs, tokens, completion_tokens, completion
        gc.collect()
        torch.cuda.empty_cache()

        # Return the completion
        return (output)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        raise NotImplementedError

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        raise NotImplementedError

    @llm_chat_callback()
    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseGen:
        raise NotImplementedError

    @llm_completion_callback()
    async def acomplete(
        self,
        prompt: str,
        images: ImageNode | List[ImageNode],  # this also takes ImageDocument which inherits from ImageNode.
        formatted: bool = False,
        **kwargs: Any
    ) -> CompletionResponse:
        raise NotImplementedError

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        raise NotImplementedError

    @llm_chat_callback()
    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        raise NotImplementedError

    @llm_chat_callback()
    async def astream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseGen:
        raise NotImplementedError


@st.cache_resource()
def get_multimodal_llm(**kwargs) -> MultiModalLLM:
    vision_llm = OpenAIMultiModal(
        model='gpt-4o',
        temperature=0,
        max_new_tokens=512,
        image_detail='auto'
    )
    return (vision_llm)
# def get_multimodal_llm(
#     model_name: str = DEFAULT_HF_MULTIMODAL_LLM,
#     processor_name: Optional[str] = None,
#     device_map: str = 'auto',
#     processor_kwargs: Optional[dict] = {},
#     model_kwargs: Optional[dict] = {'torch_dtype': 'auto'},
#     generate_kwargs: Optional[dict] = {}
# ) -> HuggingFaceMultiModalLLM:
#     vision_llm = HuggingFaceMultiModalLLM(
#         model_name=model_name,
#         processor_name=processor_name or model_name,
#         device_map=device_map,
#         processor_kwargs=processor_kwargs,
#         model_kwargs=model_kwargs,
#         generate_kwargs=generate_kwargs
#     )
#     return (vision_llm)