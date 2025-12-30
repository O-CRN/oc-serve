"""OC-Serve API Models Module."""
import time
from typing import Union, List, Literal, Optional

from starlette.requests import Request
from starlette.responses import Response, StreamingResponse, JSONResponse
from fastapi import Form
from pydantic import Field, ConfigDict
from openai._types import NOT_GIVEN
from vllm.entrypoints.openai.protocol import (
    OpenAIBaseModel,
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    CompletionResponse,
    DetokenizeRequest,
    DetokenizeResponse,
    ErrorResponse,
    TokenizeRequest,
    TokenizeResponse,
    ScoreRequest,
    ScoreResponse,
    PoolingRequest,
    PoolingResponse,
    TranscriptionRequest,
)
from vllm.utils import random_uuid


class UsageInfoTranscriptionModels(OpenAIBaseModel):
    """Usage information for transcription models."""
    transcription_tokens: int = 0
    input_audio_duration: float = 0


class TranscribeResponseData(OpenAIBaseModel):
    """Response data for transcription models."""
    index: int
    object: str = "text"
    text: str  
    seek: Optional[float] = None
    start: Optional[float] = None
    end: Optional[float] = None
    tokens: Optional[List[int]] = None
    temperature: Optional[float] = None
    avg_logprob: Optional[float] = None
    compression_ratio: Optional[float] = None
    no_speech_prob: Optional[float] = None


class UsageInfoSpeechModels(OpenAIBaseModel):
    """Usage information for speech synthesis models."""
    prompt_tokens: int = 0
    synthesis_duration: float = 0


class SpeechResponseData(OpenAIBaseModel):
    """Response data for speech synthesis models."""
    model_config = ConfigDict(extra="ignore")

    index: int
    type: str = Literal["audio", "url"]
    audio: str = None
    url: str = None


class SpeechResponse(OpenAIBaseModel):
    """Response model for speech synthesis models."""
    id: str = Field(default_factory=lambda: f"aud-{random_uuid()}")
    object: str = "list"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    data: Union[List[SpeechResponseData], List[TranscribeResponseData]]
    usage: Union[UsageInfoSpeechModels, UsageInfoTranscriptionModels] = NOT_GIVEN
