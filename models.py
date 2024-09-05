#####################################################
### DOCUMENT PROCESSOR [MODELS]
#####################################################
# Jonathan Wang

# ABOUT: 
# This project creates an app to chat with PDFs.

# This is the LANGUAGE MODELS
# that are used in the document reader.
#####################################################
## TODOS:
# <!> Add support for vLLM / AWQ / GPTQ models. (probably not going to be done due to lack of attention scores)

# Add KTransformers backend?
# https://github.com/kvcache-ai/ktransformers

# https://github.com/Tada-AI/pdf_parser

#####################################################
## IMPORTS:
from __future__ import annotations

import gc
import logging
import sys
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Union,
    cast,
    runtime_checkable,
)

import streamlit as st
import torch
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.base.llms.generic_utils import (
    messages_to_prompt as generic_messages_to_prompt,
)
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
    MessageRole,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.constants import DEFAULT_CONTEXT_WINDOW, DEFAULT_NUM_OUTPUTS
from llama_index.core.llms.callbacks import (
    llm_chat_callback,
    llm_completion_callback,
)
from llama_index.core.multi_modal_llms import MultiModalLLM
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.schema import ImageDocument, ImageNode
from llama_index.core.types import BaseOutputParser, PydanticProgramMode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from PIL import Image as PILImage
from transformers import (
    AutoImageProcessor,
    AutoModelForVision2Seq,
    AutoTokenizer,
    LogitsProcessor,
    QuantoConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)
from typing_extensions import Annotated

# from wtpsplit import SaT  # Sentence segmentation model. Dropping this. Requires adapters=0.2.1->Transformers=4.39.3 | Phi3 Vision requires Transformers 4.40.2

## NOTE: Proposal for LAZY LOADING packages for running LLMS:
# Currently not done because empahsis is on local inference w/ ability to get Attention Scores, which is not yet supported in non-HF Transformers methods.

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
### SETTINGS:
DEFAULT_HF_MULTIMODAL_LLM = "Salesforce/xgen-mm-phi3-mini-instruct-interleave-r-v1.5"
DEFAULT_HF_MULTIMODAL_CONTEXT_WINDOW = 1024
DEFAULT_HF_MULTIMODAL_MAX_NEW_TOKENS = 1024

#####################################################
### CODE:
logger = logging.getLogger(__name__)

@st.cache_resource
def get_embedder(
    model_path: str = "mixedbread-ai/mxbai-embed-large-v1",
    device: str = "cuda",  # 'cpu' is unbearably slow
) -> BaseEmbedding:
    """Given the path to an embedding model, load it."""
    # NOTE: okay we definitely could have not made this wrapper, but shrug
    return HuggingFaceEmbedding(
        model_path,
        device=device
    )


@st.cache_resource
def get_reranker(
    model_path: str = "mixedbread-ai/mxbai-rerank-large-v1",
    top_n: int = 3,
    device: str = "cpu",  # 'cuda' if we were rich
) -> SentenceTransformerRerank:  # technically this is a BaseNodePostprocessor, but that seems too abstract.
    """Given the path to a reranking model, load it."""
    # NOTE: okay we definitely could have not made this wrapper, but shrug
    return SentenceTransformerRerank(
        model=model_path,
        top_n=top_n,
        device=device
    )


## LLM Options Below
# def _get_llamacpp_llm(
#     model_path: str,
#     model_seed: int = 31415926,
#     model_temperature: float = 1e-64,  # ideally 0, but HF-type doesn't allow that. # a good dev might use sys.float_info()['min']
#     model_context_length: Optional[int] = 8192,
#     model_max_new_tokens: Optional[int] = 1024,
# ) -> BaseLLM:
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


@st.cache_resource
def _get_hf_llm(
    model_path: str,
    model_temperature: float = sys.float_info.min,  # ideally 0, but HF-type doesn't allow that. # a good dev might use sys.float_info()['min'] to confirm (?)
    model_context_length: int | None = 16384,
    model_max_new_tokens: int | None = 2048,
    hf_quant_level: int | None = 8,
) -> BaseLLM:
    """Load a Huggingface-Transformers based model using sane defaults."""
    # Fix temperature if needed; HF implementation complains about it being zero
    model_temperature = max(sys.float_info.min, model_temperature)

    # Get Quantization with BitsandBytes
    quanto_config = None  # NOTE: by default, no quantization.
    if (hf_quant_level == 4):
        # bnb_config = BitsAndBytesConfig(
        #     # load_in_8bit=True,
        #     load_in_4bit=True,
        #     # bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_compute_dtype='bfloat16',  # NOTE: Tesla T4 GPUs are too crappy for bfloat16
        #     # bnb_4bit_compute_dtype='float16'
        # )
        quanto_config = QuantoConfig(
            weights="int4"  # there's also 'int2' if you're crazy...
        )
    elif (hf_quant_level == 8):
        # bnb_config = BitsAndBytesConfig(
        #     load_in_8bit=True
        # )
        quanto_config = QuantoConfig(
            weights="int8"
        )

    # Get Stopping Tokens for Llama3 based models, because they're /special/ and added a new one.
    tokenizer = AutoTokenizer.from_pretrained(
        model_path
    )
    stopping_ids = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]
    return HuggingFaceLLM(
        model_name=model_path,
        tokenizer_name=model_path,
        stopping_ids=stopping_ids,
        max_new_tokens=model_max_new_tokens or DEFAULT_NUM_OUTPUTS,
        context_window=model_context_length or DEFAULT_CONTEXT_WINDOW,
        tokenizer_kwargs={"trust_remote_code": True},
        model_kwargs={"trust_remote_code": True, "quantization_config": quanto_config},
        generate_kwargs={
            "do_sample": not model_temperature > sys.float_info.min,
            "temperature": model_temperature,
        },
        is_chat_model=True,
    )


@st.cache_resource
def get_llm(
    model_path: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    model_temperature: float = 0,  # ideally 0, but HF-type doesn't allow that. # a good dev might use sys.float_info()['min']
    model_context_length: int | None = 8192,
    model_max_new_tokens: int | None = 1024,

    hf_quant_level: int | None = 8,  # 4-bit / 8-bit loading for HF models
) -> BaseLLM:
    """
    Given the path to a LLM, determine the type, load it in and convert it into a Llamaindex-compatable LLM.

    NOTE: I chose to set some "sane" defaults, so it's probably not as flexible as some other dev would like.
    """
    # if (model_path_extension == ".gguf"):
    #     ##### LLAMA.CPP
    #     return(_get_llamacpp_llm(model_path, model_seed, model_temperature, model_context_length, model_max_new_tokens))

    # TODO(Jonathan Wang): Consider non-HF-Transformers backends
    # vLLM support for AWQ/GPTQ models
    # I guess reluctantly AutoAWQ and AutoGPTQ packages.
    # Exllamav2 is kinda dead IMO.

    # else:
        #### No extension or weird fake extension suggests a folder, i.e., the base model from HF
    return(_get_hf_llm(model_path=model_path, model_temperature=model_temperature, model_context_length=model_context_length, model_max_new_tokens=model_max_new_tokens, hf_quant_level=hf_quant_level))


# @st.cache_resource
# def get_llm() -> BaseLLM:
#     from llama_index.llms.groq import Groq

#     llm = Groq(
#         model='llama-3.1-8b-instant',  # old: 'llama3-8b-8192'
#         api_key=os.environ.get('GROQ_API_KEY'),
#     )
#     return (llm)


class EosLogitProcessor(LogitsProcessor):
    """Special snowflake processor for Salesforce Vision Model."""
    def __init__(self, eos_token_id: int, end_token_id: int):
        super().__init__()
        self.eos_token_id = eos_token_id
        self.end_token_id = end_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.size(1) > 1: # Expect at least 1 output token.
            forced_eos = torch.full((scores.size(1),), -float("inf"), device=input_ids.device)
            forced_eos[self.eos_token_id] = 0

            # Force generation of EOS after the <|end|> token.
            scores[input_ids[:, -1] == self.end_token_id] = forced_eos
        return scores

# NOTE: These two protocols are needed to appease mypy
# https://github.com/run-llama/llama_index/blob/5238b04c183119b3035b84e2663db115e63dcfda/llama-index-core/llama_index/core/llms/llm.py#L89
@runtime_checkable
class MessagesImagesToPromptType(Protocol):
    def __call__(self, messages: Sequence[ChatMessage], images: Sequence[ImageDocument], **kwargs: Any) -> str:
        pass

MessagesImagesToPromptCallable = Annotated[
    Optional[MessagesImagesToPromptType],
    WithJsonSchema({"type": "string"}),
]


# https://huggingface.co/Salesforce/xgen-mm-phi3-mini-instruct-interleave-r-v1.5/blob/main/batch_inference.ipynb

class HuggingFaceMultiModalLLM(MultiModalLLM):
    """Supposed to be a wrapper around HuggingFace's Vision LLMS.
    Currently only supports one model type: Salesforce/xgen-mm-phi3-mini-instruct-interleave-r-v1.5
    """

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
    tokenizer_name: str = Field(
        default=DEFAULT_HF_MULTIMODAL_LLM,
        description=(
            "The name of the tokenizer to use from HuggingFace. "
            "Unused if `tokenizer` is passed in directly."
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
    stopping_ids: list[int] = Field(
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
    tokenizer_kwargs: dict = Field(
        default_factory=dict, description="The kwargs to pass to the tokenizer."
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
            "Whether the model can have multiple messages passed at once, like the OpenAI chat API."
            # LLMMetadata.__fields__["is_chat_model"].field_info.description
            # + " Be sure to verify that you either pass an appropriate tokenizer "
            # "that can convert prompts to properly formatted chat messages or a "
            # "`messages_to_prompt` that does so."
        ),
    )
    messages_images_to_prompt: MessagesImagesToPromptCallable = Field(
        default=generic_messages_to_prompt,
        description="A function that takes in a list of messages and images and returns a prompt string.",
    )

    _model: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()
    # TODO(Jonathan Wang): We need to add a separate field for AutoProcessor as opposed to ImageProcessors.
    _processor: Any = PrivateAttr()
    _stopping_criteria: Any = PrivateAttr()

    def __init__(
        self, 
        context_window: int = DEFAULT_HF_MULTIMODAL_CONTEXT_WINDOW,
        max_new_tokens: int = DEFAULT_HF_MULTIMODAL_MAX_NEW_TOKENS,
        query_wrapper_prompt: Union[str, PromptTemplate] = "{query_str}",
        tokenizer_name: str = DEFAULT_HF_MULTIMODAL_LLM,
        processor_name: str = DEFAULT_HF_MULTIMODAL_LLM,
        model_name: str = DEFAULT_HF_MULTIMODAL_LLM,
        model: Any | None = None,
        tokenizer: Any | None = None,
        processor: Any | None = None,
        device_map: str = "auto",
        stopping_ids: list[int] | None = None,
        tokenizer_kwargs: dict[str, Any] | None = None,
        processor_kwargs: dict[str, Any] | None = None,
        tokenizer_outputs_to_remove: list[str] | None = None,
        model_kwargs: dict[str, Any] | None = None,
        generate_kwargs: dict[str, Any] | None = None,
        is_chat_model: bool = False,
        callback_manager: CallbackManager | None = None,
        system_prompt: str = "",
        messages_images_to_prompt: Callable[[Sequence[ChatMessage], Sequence[ImageDocument]], str] | None = None,
        # completion_to_prompt: Callable[[str], str] | None = None,
        # pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        # output_parser: BaseOutputParser | None = None,
    ) -> None:

        logger.info(f"CUDA Memory Pre-AutoModelForVision2Seq: {torch.cuda.mem_get_info()}")
        # Salesforce one is a AutoModelForVision2Seq, but not AutoCausalLM which is more common.
        model = model or AutoModelForVision2Seq.from_pretrained(
            model_name,
            device_map=device_map,
            trust_remote_code=True,
            **(model_kwargs or {})
        )
        logger.info(f"CUDA Memory Post-AutoModelForVision2Seq: {torch.cuda.mem_get_info()}")

        # check context_window
        config_dict = model.config.to_dict()
        model_context_window = int(
            config_dict.get("max_position_embeddings", context_window)
        )
        if model_context_window < context_window:
            logger.warning(
                f"Supplied context_window {context_window} is greater "
                f"than the model's max input size {model_context_window}. "
                "Disable this warning by setting a lower context_window."
            )
            context_window = model_context_window

        processor_kwargs = processor_kwargs or {}
        if "max_length" not in processor_kwargs:
            processor_kwargs["max_length"] = context_window

        # NOTE: Sometimes models (phi-3) will use AutoProcessor and include the tokenizer within it.
        logger.info(f"CUDA Memory Pre-Processor: {torch.cuda.mem_get_info()}")
        processor = processor or AutoImageProcessor.from_pretrained(
            processor_name or model_name,
            trust_remote_code=True,
            **processor_kwargs
        )
        logger.info(f"CUDA Memory Post-Processor: {torch.cuda.mem_get_info()}")

        tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            tokenizer_name or model_name,
            trust_remote_code=True,
            **(tokenizer_kwargs or {})
        )
        logger.info(f"CUDA Memory Post-Tokenizer: {torch.cuda.mem_get_info()}")

        # Tokenizer-Model disagreement
        if (hasattr(tokenizer, "name_or_path") and tokenizer.name_or_path != model_name):  # type: ignore (checked for attribute)
            logger.warning(
                f"The model `{model_name}` and processor `{getattr(tokenizer, 'name_or_path', None)}` "
                f"are different, please ensure that they are compatible."
            )
        # Processor-Model disagreement
        if (hasattr(processor, "name_or_path") and getattr(processor, "name_or_path", None) != model_name):
            logger.warning(
                f"The model `{model_name}` and processor `{getattr(processor, 'name_or_path', None)}` "
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
                return any(input_ids[0][-1] == stop_id for stop_id in stopping_ids_list)

        stopping_criteria = StoppingCriteriaList([StopOnTokens()])

        if isinstance(query_wrapper_prompt, str):
            query_wrapper_prompt = PromptTemplate(query_wrapper_prompt)

        messages_images_to_prompt = messages_images_to_prompt or self._processor_messages_to_prompt

        # Initiate standard LLM
        super().__init__(
            callback_manager=callback_manager or CallbackManager([]),
        )
        logger.info(f"CUDA Memory Post-SuperInit: {torch.cuda.mem_get_info()}")

        # Initiate remaining fields
        self._model = model
        self._tokenizer = tokenizer
        self._processor = processor
        logger.info(f"CUDA Memory Post-Init: {torch.cuda.mem_get_info()}")
        self._stopping_criteria = stopping_criteria
        self.model_name = model_name
        self.context_window=context_window
        self.max_new_tokens=max_new_tokens
        self.system_prompt=system_prompt
        self.query_wrapper_prompt=query_wrapper_prompt
        self.tokenizer_name=tokenizer_name
        self.processor_name=processor_name
        self.model_name=model_name
        self.device_map=device_map
        self.stopping_ids=stopping_ids or []
        self.tokenizer_outputs_to_remove=tokenizer_outputs_to_remove or []
        self.tokenizer_kwargs=tokenizer_kwargs or {}
        self.processor_kwargs=processor_kwargs or {}
        self.model_kwargs=model_kwargs or {}
        self.generate_kwargs=generate_kwargs or {}
        self.is_chat_model=is_chat_model
        self.messages_images_to_prompt=messages_images_to_prompt
        # self.completion_to_prompt=completion_to_prompt,
        # self.pydantic_program_mode=pydantic_program_mode,
        # self.output_parser=output_parser,

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

    def _processor_messages_to_prompt(self, messages: Sequence[ChatMessage], images: Sequence[ImageDocument]) -> str:
        ### TODO(Jonathan Wang): Make this work generically. Currently we're building for `Salesforce/xgen-mm-phi3-mini-instruct-interleave-r-v1.5`
        """Converts a list of messages into a prompt for the multimodal LLM.
        NOTE: we assume for simplicity here that these images are related, and not the user bouncing between multiple different topics. Thus, we send them all at once.

        Args:
            messages (Sequence[ChatMessage]): A list of the messages to convert, where each message is a dict containing the message role and content.
            images (Sequence[ImageDocument]): The number of images the user is passing to the MultiModalLLM.
        Returns:
            str: The prompt.
        """
        # NOTE: For `Salesforce/xgen-mm-phi3-mini-instruct-interleave-r-v1.5`, we actually ignore the `images`; no plaaceholders.

        """Use the tokenizer to convert messages to prompt. Fallback to generic."""
        if hasattr(self._tokenizer, "apply_chat_template"):
            messages_dict = [
                {"role": message.role.value, "content": message.content}
                for message in messages
            ]
            return self._tokenizer.apply_chat_template(
                messages_dict, tokenize=False, add_generation_prompt=True
            )

        return generic_messages_to_prompt(messages)

    @llm_completion_callback()
    def complete(
        self,
        prompt: str,
        image_documents: ImageNode | List[ImageNode] | ImageDocument | List[ImageDocument],  # this also takes ImageDocument which inherits from ImageNode.
        formatted: bool = False,
        **kwargs: Any
    ) -> CompletionResponse:
        """Given a prompt and image node(s), get the Phi-3 Vision prompt"""
        # Handle images input
        # https://huggingface.co/Salesforce/xgen-mm-phi3-mini-instruct-interleave-r-v1.5/blob/main/demo.ipynb
        batch_image_list = []
        batch_image_sizes = []
        batch_prompt = []

        # Fix image_documents input typing
        if (not isinstance(image_documents, list)):
            image_documents = [image_documents]
        image_documents = [cast(ImageDocument, image) for image in image_documents]  # we probably won't be using the Document features, so I think this is fine.

        # Convert input images into PIL images for the model.
        image_list = []
        image_sizes = []
        for image in image_documents:
            # NOTE: ImageDocument inherets from ImageNode. We'll go extract the image.
            image_io = image.resolve_image()
            image_pil = PILImage.open(image_io)
            image_list.append(self._processor([image_pil], image_aspect_ratio='anyres')['pixel_values'].to(self._model.device))
            image_sizes.append(image_pil.size)

        batch_image_list.append(image_list)
        batch_image_sizes.append(image_sizes)
        batch_prompt.append(prompt)  # only one question per image

        # Get the prompt
        if not formatted and self.query_wrapper_prompt:
            prompt = self.query_wrapper_prompt.format(query_str=prompt)

        prompt_sequence = []
        if self.system_prompt:
            prompt_sequence.append(ChatMessage(role=MessageRole.SYSTEM, content=self.system_prompt))
        prompt_sequence.append(ChatMessage(role=MessageRole.USER, content=prompt))

        prompt = self.messages_images_to_prompt(messages=prompt_sequence, images=image_documents)

        # Get the model input
        batch_inputs = {
            "pixel_values": batch_image_list
        }
        language_inputs = self._tokenizer(
            [prompt], 
            return_tensors="pt",
            padding='longest',  # probably not needed.
            max_length=self._tokenizer.model_max_length,
            truncation=True
        ).to(self._model.device)  
        # TODO: why does the example cookbook have this weird conversion to Cuda instead of .to(device)?
        # language_inputs = {name: tensor.cuda() for name, tensor in language_inputs.items()}
        batch_inputs.update(language_inputs)
        
        gc.collect()
        torch.cuda.empty_cache()

        # remove keys from the tokenizer if needed, to avoid HF errors
        # TODO: this probably is broken and wouldn't work.
        for key in self.tokenizer_outputs_to_remove:
            if key in batch_inputs:
                batch_inputs.pop(key, None)

        # Get output
        tokens = self._model.generate(
            **batch_inputs, 
            image_sizes=batch_image_sizes,
            pad_token_id=self._tokenizer.pad_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
            max_new_tokens=self.max_new_tokens,
            stopping_criteria=self._stopping_criteria,
            # NOTE: Special snowflake processor for Salesforce XGEN Phi3 Mini.
            logits_processor=[EosLogitProcessor(eos_token_id=self._tokenizer.eos_token_id, end_token_id=32007)],
            **self.generate_kwargs
        )
        gc.collect()
        torch.cuda.empty_cache()

        # completion_tokens = tokens[:, batch_inputs['input_ids'].shape[1]:]
        completion = self._tokenizer.batch_decode(
            tokens, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        gc.collect()
        torch.cuda.empty_cache()

        output = CompletionResponse(text=completion, raw={'model_output': tokens})

        # Clean stuff up
        del batch_image_list, batch_image_sizes, batch_inputs, tokens, completion
        gc.collect()
        torch.cuda.empty_cache()

        # Return the completion
        return output

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


# @st.cache_resource()
# def get_multimodal_llm(**kwargs) -> MultiModalLLM:
#     vision_llm = OpenAIMultiModal(
#         model='gpt-4o-mini',
#         temperature=0,
#         max_new_tokens=512,
#         image_detail='auto'
#     )
#     return (vision_llm)

@st.cache_resource
def get_multimodal_llm(
    model_name: str = DEFAULT_HF_MULTIMODAL_LLM,
    device_map: str = "cuda",  # does not support 'auto'
    processor_kwargs: dict[str, Any] | None = None,
    model_kwargs: dict[str, Any] | None = None, # {'torch_dtype': torch.bfloat16}, # {'torch_dtype': torch.float8_e5m2}
    generate_kwargs: dict[str, Any] | None = None,  # from the example cookbook

    hf_quant_level: int | None = 8,
) -> HuggingFaceMultiModalLLM:

    # Get default generate kwargs
    if model_kwargs is None:
        model_kwargs = {}
    if processor_kwargs is None:
        processor_kwargs = {}
    if generate_kwargs is None:
        generate_kwargs = {
            "temperature": sys.float_info.min,
            "top_p": None,
            "num_beams": 1
            # NOTE: we hack in EOSLogitProcessor in the HuggingFaceMultiModalLLM because it allows us to get the tokenizer.eos_token_id
        }

    # Get Quantization with Quanto
    quanto_config = None  # NOTE: by default, no quantization.
    if (hf_quant_level == 4):
        # bnb_config = BitsAndBytesConfig(
        #     # load_in_8bit=True,
        #     load_in_4bit=True,
        #     # bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_compute_dtype='bfloat16',  # NOTE: Tesla T4 GPUs are too crappy for bfloat16
        #     # bnb_4bit_compute_dtype='float16'
        # )
        quanto_config = QuantoConfig(
            weights="int4"  # there's also 'int2' if you're crazy...
        )
    elif (hf_quant_level == 8):
        # bnb_config = BitsAndBytesConfig(
        #     load_in_8bit=True
        # )
        quanto_config = QuantoConfig(
            weights="int8"
        )

    if (quanto_config is not None):
        model_kwargs["quantization_config"] = quanto_config

    return HuggingFaceMultiModalLLM(
        model_name=model_name,
        device_map=device_map,
        processor_kwargs=processor_kwargs,
        model_kwargs=model_kwargs,
        generate_kwargs=generate_kwargs,

        max_new_tokens=1024  # from the example cookbook
    )
