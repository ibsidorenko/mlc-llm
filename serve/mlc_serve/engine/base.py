from __future__ import annotations
import structlog
import torch
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

from typing import List, Callable, Any, Optional, Dict
import inspect

from .sampling_params import SamplingParams, SamplingType
from ..openai_logprob_protocol import LogprobsContent
from ..model.base import ModelArtifactConfig

LOG = structlog.stdlib.get_logger(__name__)
RequestId = str


@dataclass
class RawLogprobsInfo:
    current_token_id: int
    current_logprob: float
    top_token_ids: Optional[torch.Tensor]
    top_logprobs: Optional[torch.Tensor]


# TODO(@sunggg): consider transition to something like Pydantic
@dataclass
class MLCServeEngineConfig:
    # The maximum number of tokens in the batch.
    # TODO(@sunggg): figure out better defaults
    use_staging_engine: bool = True
    max_num_batched_tokens: int = 4096
    max_num_seq: int = 2048
    max_num_seq_per_request: Optional[int] = None # default to `max_num_seq / 4`
    min_decode_steps: int = 32
    max_decode_steps: int = 48
    init_timeout: int = 120
    model_type: str = "tvm"  # "tvm", "torch"
    num_shards: Optional[int] = None  # Need to be specified for if model_type is "torch"
    gpu_memory_utilization: float = 0.9

    @classmethod
    def _from_json(config_cls, json_obj: Dict[Any, Any]):
        return config_cls(
            **{
                k: v
                for k, v in json_obj.items()
                if k in inspect.signature(config_cls).parameters
            }
        )


def get_engine_config(dict_config):
    engine_config = MLCServeEngineConfig._from_json(dict_config)
    # Checks to make sure engine configs are set correctly
    # since engine config is critical to the performance
    assert isinstance(engine_config.use_staging_engine, bool)
    assert isinstance(engine_config.max_num_batched_tokens, int)
    assert isinstance(engine_config.max_decode_steps, int)
    assert isinstance(engine_config.min_decode_steps, int)
    assert (engine_config.min_decode_steps > 0) and (engine_config.max_decode_steps > 0)
    assert engine_config.max_decode_steps > engine_config.min_decode_steps

    if engine_config.model_type == "torch":
        assert (
            engine_config.num_shards is not None
        ), "num_shards in MLCServeEngineConfig needs to be provided for PT models."

    return engine_config


@dataclass
class StoppingCriteria:
    """
    Parameters about when to stop text generation.
    """

    max_tokens: Optional[int] = None
    stop_sequences: Optional[list[str]] = None


@dataclass
class ChatMessage:
    role: str
    content: Optional[str]


@dataclass
class DebugOptions:
    ignore_eos: bool = False
    # Override messages with a single prompt, skipping conversation template
    prompt: Optional[str] = None
    # Overrides prompts, skipping tokenization
    prompt_token_ids: Optional[list[int]] = None


class FinishReason(Enum):
    Stop = "stop"
    Length = "length"
    Cancelled = "cancelled"


# A single token.
Token = int


@dataclass
class ValidationError:
    msg: str


# The type signature of the token validation callback.
ValidateTokensCallback = Callable[["Request", List[Token]], ValidationError]


@dataclass
class Request:
    request_id: RequestId
    messages: List[ChatMessage]
    # Number of sequences to generate
    num_sequences: int = 1
    # TODO: should `best_of` be handled in the serving layer?
    best_of: Optional[int] = None
    # Options for sampling.
    sampling_params: SamplingParams = field(default_factory=SamplingParams)
    # Options for stopping.
    stopping_criteria: StoppingCriteria = field(
        default_factory=lambda: StoppingCriteria()
    )
    # Options for debugging.
    debug_options: DebugOptions = field(default_factory=DebugOptions)
    # Perform request validation post-tokenization, used by the HTTP layer to control validation.
    validate_tokens: Optional[ValidateTokensCallback] = None
    # Context variables to attach to logging.
    contextvars: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.best_of is None:
            self.best_of = self.num_sequences
        if self.num_sequences < 1:
            raise ValueError(
                f"num_sequences must be at least 1, got {self.num_sequences}."
            )
        if self.best_of < self.num_sequences:
            raise ValueError(
                f"best_of must be greater than or equal to num_sequences, "
                f"got n={self.num_sequences} and best_of={self.best_of}."
            )
        if (
            self.best_of > 1
            and self.sampling_params.sampling_type == SamplingType.GREEDY
        ):
            raise ValueError(
                "best_of must be 1 when using greedy sampling." f"Got {self.best_of}."
            )

        if self.debug_options.prompt_token_ids is not None:
            LOG.warn(
                f"`debug_options.prompt_token_ids` is provided. This will be used directly and the prompts will be ignored if provided."
            )
            if not isinstance(self.debug_options.prompt_token_ids, list):
                raise ValueError("`prompt_token_ids` needs to be list.")
        else:
            if self.debug_options.prompt is not None:
                LOG.warn(
                    f"`debug_options.prompt` is provided. It will be used instead of `messages`. Conversation template will be skipped."
                )


@dataclass
class SequenceOutput:
    index: int
    delta: Optional[str] = None
    # If finish_reason is not None, delta should be None.
    finish_reason: Optional[FinishReason] = None
    # Number of generated tokens so far
    num_generated_tokens: int = 0
    logprob_info: List[Optional[LogprobsContent]] = field(default_factory=list)

    @property
    def is_finished(self) -> bool:
        return self.finish_reason is not None


@dataclass
class RequestOutput:
    request_id: RequestId
    sequences: List[SequenceOutput]
    # TODO: reconsider the place to put this number
    # Only set for outputs with valid sequence outputs
    num_prompt_tokens: Optional[int] = None
    # TODO(@jroesch): We should generalize the type here so we are allowed to return more structured information
    # for logging/user output.
    #
    # Right now I am abusing dynamic typing by putting the ValidationError in here.
    # I would prefer to unblock ourselves then figure this one out right now
    error: Optional[str] = None
    # Context variables to attach to logging.
    contextvars: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_finished(self) -> bool:
        return self.error is not None or all(seq.is_finished for seq in self.sequences)


@dataclass
class InferenceStepResult:
    outputs: list[RequestOutput]


class InferenceEngine(ABC):
    """
    Expose the model config to the high-level APIs.
    """

    model_artifact_config: ModelArtifactConfig

    @abstractmethod
    def add(self, requests: list[Request]) -> None:
        """
        Add requests to the InferenceEngine.

        Requests will be processed when `step` is called, if there is capacity.
        Requests will be handled on a first-in, first-out basis.
        """
        ...

    @abstractmethod
    def cancel(self, request_id: RequestId) -> None:
        """
        Cancel the generation of a request.

        The next call to `step` will return TextGenerationOutput for cancelled requests.
        The output will contain empty delta and finish reason `cancelled`.
        """
        ...

    @abstractmethod
    def has_pending_requests(self) -> bool:
        """
        Check if there is pending requests in the engine.
        """
        ...

    @abstractmethod
    def wait_for_request(self, timeout_seconds=None) -> bool:
        """
        Block until there is request to process.
        It can also be given a timeout_seconds parameter, allowing it to return even if
        no requests are coming in. The return value is a boolean that indicates whether
        there are requests when it's returned.
        """
        ...

    @abstractmethod
    def step(self) -> InferenceStepResult:
        """
        Perform a single inference step. In general it will generates one token for each
        requests that are being processed, but it could generate more if speculative decoding
        is used.

        If the engine has no requests in the queue, `step` will return immediately with
        an empty `InferenceStepResult.outputs`.
        """
        ...


class ScopedInferenceEngine(InferenceEngine):
    @abstractmethod
    def start(self) -> None:
        ...

    @abstractmethod
    def stop(self) -> None:
        ...


@dataclass(frozen=True)
class SequenceId:
    """
    SequenceId identified a unique sequence to be generated.

    Each request will have `n` unique SequenceIds, where `n` is
    the `n` from SamplingParams.
    """

    request_id: RequestId
    sequence_index: int


# IMPORTANT!!!
#
# To support parallel sampling, we use the following convention to encode prompt
# and decode sequences.
#
# Prompt sequences: SequenceID(request_id, PROMPT_SEQEUNCE_INDEX)
# Decode sequences: SequenceID(request_id, idx) for idx in [0, num_sequences]
#
# This way, we can immediately get the corresponding prompt sequence from a decode sequence.

PROMPT_SEQEUNCE_INDEX = -1


def get_prompt_sequence_id(request_id: RequestId) -> SequenceId:
    return SequenceId(request_id, PROMPT_SEQEUNCE_INDEX)


@dataclass
class GenerationSequence:
    seq_id: SequenceId
    generated_token_ids: list[int]
    next_start_position: int
    output_text: str
    prefix_begin_offset: int = 0
    prefix_end_offset: int = 0
    prev_tokens: Optional[List[str]] = None
    is_finished: bool = False


@dataclass
class RequestState:
    """
    The internal state of request in the InferenceEngine.
    """

    request_id: RequestId
    prompt_token_ids: list[int]
    prompt_mask: Optional[torch.Tensor]
    sampling_params: SamplingParams
    generation_sequences: list[GenerationSequence]
    stopping_criteria: StoppingCriteria
    debug_options: DebugOptions
    arrival_timestamp: float
    validation_err: Optional[ValidationError] = None
    # Context variables to attach to logging.
    contextvars: Dict[str, Any] = field(default_factory=dict)
    is_prefilled: bool = False

    @property
    def is_finished(self) -> bool:
        return all(seq.is_finished for seq in self.generation_sequences)

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_sequences(self) -> int:
        return len(self.generation_sequences)

    @property
    def num_total_tokens(self) -> int:
        return self.prompt_len + sum(
            len(gen_seq.generated_token_ids) for gen_seq in self.generation_sequences
        )
