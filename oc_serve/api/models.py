"""OC-Serve API Models Module."""
import time
from typing import Union, List, Literal, Optional

from http import HTTPStatus
from starlette.requests import Request as OCRequest
from starlette.responses import (
    Response as OCResponse,
    StreamingResponse as OCStreamingResponse,
    JSONResponse as OCJSONResponse,
)
from fastapi import Form
from fastapi.responses import ORJSONResponse as OCORJSONResponse
from pydantic import Field, ConfigDict
from openai._types import NOT_GIVEN
from vllm.entrypoints.openai.protocol import (
    OpenAIBaseModel as OCOpenAIBaseModel,
    ChatCompletionRequest as OCChatCompletionRequest,
    ChatCompletionResponse as OCChatCompletionResponse,
    CompletionRequest as OCCompletionRequest,
    CompletionResponse as OCCompletionResponse,
    DetokenizeRequest as OCDetokenizeRequest,
    DetokenizeResponse as OCDetokenizeResponse,
    ErrorResponse as OCErrorResponse,
    TokenizeRequest as OCTokenizeRequest,
    TokenizeResponse as OCTokenizeResponse,
    PoolingRequest as OCPoolingRequest,
    PoolingResponse as OCPoolingResponse,
    TranscriptionRequest as OCTranscriptionRequest,
    EmbeddingRequest as OCEmbeddingRequest,
    EmbeddingResponse as OCEmbeddingResponse,
    ScoreRequest as VLLMScoreRequest,
    ScoreResponse as VLLMScoreResponse,
    RerankRequest as VLLMRerankRequest,
    RerankResponse as VLLMRerankResponse,
)
from vllm.utils import random_uuid
from sglang.srt.entrypoints.openai.protocol import (
    ModelCard as OCModelCard,
    ModelList as OCModelList,
    ScoringRequest as SGLangScoringRequest,
    ScoringResponse as SGLangScoringResponse,
    V1RerankReqInput as SGLangRerankRequest,
    RerankResponse as SGLangRerankResponse,
)

OCRerankRequest = Union[VLLMRerankRequest, SGLangRerankRequest]
OCRerankResponse = Union[VLLMRerankResponse, SGLangRerankResponse]
OCScoringRequest = Union[VLLMScoreRequest, SGLangScoringRequest]
OCScoringResponse = Union[VLLMScoreResponse, SGLangScoringResponse]

class OCUsageInfoTranscriptionModels(OCOpenAIBaseModel):
    """Usage information for transcription models."""
    transcription_tokens: int = 0
    input_audio_duration: float = 0


class OCTranscriptionResponseData(OCOpenAIBaseModel):
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


class OCUsageInfoSpeechModels(OCOpenAIBaseModel):
    """Usage information for speech synthesis models."""
    prompt_tokens: int = 0
    synthesis_duration: float = 0


class OCSpeechResponseData(OCOpenAIBaseModel):
    """Response data for speech synthesis models."""
    model_config = ConfigDict(extra="ignore")

    index: int
    type: str = Literal["audio", "url"]
    audio: str = None
    url: str = None


class OCSpeechResponse(OCOpenAIBaseModel):
    """Response model for speech synthesis models."""
    id: str = Field(default_factory=lambda: f"aud-{random_uuid()}")
    object: str = "list"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    data: Union[List[OCSpeechResponseData], List[OCTranscriptionResponseData]]
    usage: Union[OCUsageInfoSpeechModels, OCUsageInfoTranscriptionModels] = NOT_GIVEN

class OCUsageInfoEmbed(OCOpenAIBaseModel):
    """Usage information for embedding models."""
    prompt_tokens: Optional[int] = 0
    total_tokens: Optional[int] = 0
    embedding_dimension_size: int = 0
    total_inputs: Optional[int] = 0
    audio_length: Optional[float] = 0

def error_response(message: str,
                          err_type: str = "BadRequestError",
                          status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,) -> OCORJSONResponse:
    """Helper function to create an error response."""
    error = OCErrorResponse(message=message, type=err_type, code=status_code.value)
    return OCORJSONResponse(content=error.model_dump(), status_code=error.code)

