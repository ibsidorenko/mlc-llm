"""
Required interfaces for the actual inference capability in InferenceEngine.
"""
from dataclasses import dataclass
from typing import Optional, Protocol, Union, List, Sequence, Any
import torch

from .base import (
    ChatMessage,
    MLCServeEngineConfig,
    RawLogprobsInfo,
    RequestId,
    RequestState,
    SequenceId,
)
from ..model.base import ModelArtifactConfig
from .sampling_params import SamplingParams


@dataclass
class PrefillRequest:
    request_id: RequestId
    # `token_ids` contains prompt token ids
    token_ids: List[int]
    prompt_mask: Optional[torch.Tensor]
    # Number of sequences to generate
    num_sequence: int
    sampling_params: SamplingParams

    # Extension for multi-modal model
    # class ImagePayload:
    #   pre_image_tokens: List[int]
    #   image: bytes
    #   post_image_tokens: List[int]


@dataclass
class DecodeRequest:
    sequence_id: SequenceId
    prompt_token_counts: int
    prompt_mask: Optional[torch.Tensor]
    # Decoded tokens for this sequence
    token_ids: List[int]
    sampling_params: SamplingParams


@dataclass
class DraftTokens:
    token_ids: List[int]

    @property
    def num_tokens(self):
        return len(self.token_ids)


@dataclass
class EvictedTokens:
    token_ids: List[int]

    @property
    def num_tokens(self):
        return len(self.token_ids)


@dataclass
class EvalMultiQueryRequest:
    sequence_id: SequenceId
    num_past_tokens: int
    prompt_mask: Optional[torch.Tensor]
    queries: Union[DraftTokens, EvictedTokens]
    sampling_params: SamplingParams


@dataclass
class FailedRequest:
    request_id: RequestId
    error: Exception


RequestType = Union[PrefillRequest, DecodeRequest, EvalMultiQueryRequest]


@dataclass
class TextGenerationResult:
    """
    Represent the result of sequence generation.
    """

    sequence_id: SequenceId
    # for most cases, there should be only one token returned
    # making this a list of token ids to leave room for speculative decoding
    generated_tokens: List[int]
    error: Optional[str]
    logprob_info: Optional[List[Optional[RawLogprobsInfo]]]


class KVCache(Protocol):
    """
    Opaque object representing the KVCache.

    It should be passed to Executor.generate_text.
    """


class KVCacheManager(Protocol):
    def get_cache(self) -> KVCache:
        """
        Get an opaque object that represents the KVCache.

        The returned value should be passed to Executor.generate_text.
        """

    def allocate(self, request_id: RequestId, num_tokens: int, num_sequences: int):
        """
        Allocate cache space for request, raise error if there is no space.
        """

    def extend(self, sequence_id: SequenceId, new_tokens: int):
        """
        Extend cache space for a sequence, raise error if there is no space.
        """

    def free(self, sequence_id: SequenceId):
        """
        Free cache space for a sequence in a request.
        """

    def free_request(self, state: RequestState):
        """
        Free cache space for all sequences in a request.
        """

    def get_kv_cache_size(self) -> int:
        """
        Return the size of the cache, in number of tokens.
        """

    def get_free_space(self) -> int:
        """
        Get available space of the cache.
        Return number of tokens that can be allocated for a new request.

        For paged KV cache, this ignores the remaining tokens in pages allocated
        for existing sequences, since they cannot be used for the new request.
        """

    def get_max_new_tokens(self) -> int:
        """
        Get the maximum number of new tokens that can be extended for
        all sequences in the cache.

        For example, if the cache size is 16 tokens, with page size 1, and
        there are 3 sequences in the cache, each of them have 3 tokens cached,
        this method should return 2.

        It should return the result of `get_kv_cache_size` if there is
        no requests in the cache.
        """


class TextGenerator(Protocol):
    """
    TextGenerator provides an abstract interface to perform the actual model inference
    """

    def generate(
        self,
        requests: Sequence[RequestType],
        kv_cache,
    ) -> List[TextGenerationResult]:
        """
        A unified entrypoint for text generation.

        `requests` could contain both new requests to prefill and existing request to decode.
        It's up to the ModelExecutor to decide how the actual inference is performed.
        It could do one prefill pass and one decode pass, or just do a single pass.
        """


class Tokenizer(Protocol):
    _tokenizer: Any
    eos_token_id: int
    skip_special_tokens: bool
    all_special_ids: List[int]
    is_fast: bool

    def encode(self, text: str) -> List[int]:
        ...

    def decode(self, tokens: List[int]) -> str:
        ...

    def convert_ids_to_tokens(self, token_ids: List[int]) -> List[str]:
        ...

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        ...


class ConversationTemplate(Protocol):
    def apply(self, messages: List[ChatMessage]) -> str:
        pass


class TextTokenGeneratorModule(Protocol):
    """
    A module that provides components for the token generation process.
    """

    @property
    def text_generator(self) -> TextGenerator:
        ...

    @property
    def cache_manager(self) -> KVCacheManager:
        ...


class TokenizerModule(Protocol):
    """
    A module that provides components for the tokenization process.
    """

    @property
    def tokenizer(self) -> Tokenizer:
        ...

    @property
    def conversation_template(self) -> ConversationTemplate:
        ...


class ModelModule(TextTokenGeneratorModule, TokenizerModule, Protocol):
    model_artifact_config: ModelArtifactConfig
    engine_config: MLCServeEngineConfig
