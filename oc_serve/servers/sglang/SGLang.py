"""SGLang Server implementation for OC-Serve."""
import asyncio
from typing import Annotated

from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.entrypoints.openai.serving_completions import OpenAIServingCompletion
from sglang.srt.entrypoints.openai.serving_embedding import OpenAIServingEmbedding
from sglang.srt.entrypoints.openai.serving_rerank import OpenAIServingRerank
from sglang.srt.entrypoints.openai.serving_score import OpenAIServingScore
from sglang.srt.entrypoints.engine import _launch_subprocesses as sglang_launch_subprocesses
from sglang.utils import get_exception_traceback
from prometheus_client.openmetrics.exposition import CONTENT_TYPE_LATEST, generate_latest

from oc_serve.servers import Server
from oc_serve.utils import (
    oc_logger,
    get_metrics_registry,
)
from oc_serve.api.models import (
    Form,
    OCRequest,
    OCResponse,
    OCChatCompletionRequest,
    OCCompletionRequest,
    OCEmbeddingRequest,
    OCTranscriptionRequest,
    OCModelCard,
    OCModelList,
    OCScoringRequest,
    OCRerankRequest,
    OCTokenizeRequest,
    OCDetokenizeRequest,
    OCJSONResponse,
    OCORJSONResponse,
    OCTokenizeResponse,
    OCDetokenizeResponse,
    OCPoolingRequest,
    HTTPStatus,
    error_response,
)
from configs import ServerConfigs
from configs.servers_configs.SGLangServerArgs import _GlobalState

@Server.register("sglang")
class SGLang(Server):
    """SGLang Server implementation."""
    def __init__(self, server_configs: ServerConfigs):
        """Initialize the SGLang server."""
        self.logger = oc_logger.get_logger("sglang_server")
        self.server_args = server_configs
        tokenizer_manager, template_manager, scheduler_info, _ = sglang_launch_subprocesses(server_configs)
        self.global_state = _GlobalState(tokenizer_manager=tokenizer_manager,
                                         template_manager=template_manager,
                                         scheduler_info=scheduler_info,)
        self.openai_serving_completion = OpenAIServingCompletion(self.global_state.tokenizer_manager,
                                                                 self.global_state.template_manager)
        self.openai_serving_chat = OpenAIServingChat(self.global_state.tokenizer_manager,
                                                     self.global_state.template_manager)
        self.openai_serving_embedding = OpenAIServingEmbedding(self.global_state.tokenizer_manager,
                                                               self.global_state.template_manager)
        self.openai_serving_score = OpenAIServingScore(self.global_state.tokenizer_manager)
        self.openai_serving_rerank = OpenAIServingRerank(self.global_state.tokenizer_manager)
        self.semaphore = asyncio.Semaphore(int(getattr(server_configs.extra_args, 'max_concurrent_calls', 1)))
        self.metrics_registry = get_metrics_registry()

    async def check_model_health(self, raw_request: OCRequest = None) -> OCResponse:
        """Check the health of the model"""
        try:
            async with self.semaphore:
                if self.global_state.tokenizer_manager is None:
                    return error_response(
                        message="Model is not healthy: tokenizer_manager is None",
                        err_type="ModelHealthError",
                        status_code=HTTPStatus.SERVICE_UNAVAILABLE,
                    )
                _ = self.global_state.tokenizer_manager.tokenizer
                return OCResponse(status_code=200,
                                content="Model is Healthy!")
        except Exception as e:
            self.logger.error(f"Health check failed: {get_exception_traceback()}")
            return error_response(
                        message=f"Model is not healthy: {str(e)}",
                        err_type="ModelHealthError",
                        status_code=HTTPStatus.SERVICE_UNAVAILABLE,
                    )

    async def get_model_info(self, raw_request: OCRequest = None) -> OCResponse:
        """Get model information such as supported model names and their max context lengths."""
        async with self.semaphore:
            served_model_names = [self.global_state.tokenizer_manager.served_model_name]
            model_cards = []
            for served_model_name in served_model_names:
                model_card = OCModelCard(
                        id=served_model_name,
                        root=served_model_name,
                        max_model_len=self.global_state.tokenizer_manager.model_config.context_len,
                    )
                model_cards.append(model_card)
            model_list = OCModelList(data=model_cards).model_dump()
            return OCJSONResponse(content=model_list)

    async def instruct(self, request: OCChatCompletionRequest, raw_request: OCRequest) -> OCResponse:
        """Handle chat completion requests."""
        async with self.semaphore:
            return await self.openai_serving_chat.handle_request(request, raw_request)

    async def complete(self, request: OCCompletionRequest, raw_request: OCRequest) -> OCResponse:
        """Handle completion requests."""
        async with self.semaphore:
            return await self.openai_serving_completion.handle_request(request, raw_request)

    async def transcribe(self, request: Annotated[OCTranscriptionRequest, Form()],
                       raw_request: OCRequest) -> OCResponse:
        """Handle transcription requests."""
        return error_response(
            message="Transcribe endpoint is not implemented in SGLang server.",
            err_type="NotImplementedError",
            status_code=HTTPStatus.NOT_IMPLEMENTED,
        )

    async def tokenize(self, request: OCTokenizeRequest, raw_request: OCRequest) -> OCResponse:
        """Handle tokenization requests."""
        async with self.semaphore:
            try:
                tokenizer = self.global_state.tokenizer_manager.tokenizer
                max_model_len = getattr(tokenizer, "model_max_length", -1)
                if isinstance(request.prompt, str):
                    token_ids = tokenizer.encode(
                        request.prompt,
                        add_special_tokens=request.add_special_tokens,
                    )
                    count = len(token_ids)
                    tokens_to_return = token_ids
                elif isinstance(request.prompt, list):
                    token_ids_list = [
                        tokenizer.encode(
                            text,
                            add_special_tokens=request.add_special_tokens,
                        )
                        for text in request.prompt
                    ]
                    count = [len(ids) for ids in token_ids_list]
                    tokens_to_return = token_ids_list
                else:
                    return error_response(
                        message=f"Invalid input: 'prompt' must be str or List[str]. \
                            Found type: {type(request.prompt)}. Expected str or List[str].",
                        err_type="InvalidRequest",
                        status_code=HTTPStatus.BAD_REQUEST
                        )
                return OCORJSONResponse(OCTokenizeResponse(tokens=tokens_to_return,
                                                    count=count,
                                                    max_model_len=max_model_len).model_dump())
            except Exception as e:
                self.logger.error(f"Error during tokenization: {get_exception_traceback()}")
                return error_response(
                    message=f"Internal server error during tokenization: {e}",
                    err_type="InternalServerError",
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR
                    )

    async def detokenize(self, request: OCDetokenizeRequest, raw_request: OCRequest) -> OCResponse:
        """Handle detokenization requests."""
        async with self.semaphore:
            try:
                tokenizer = self.global_state.tokenizer_manager.tokenizer
                if (
                    isinstance(request.tokens, list)
                    and request.tokens
                    and isinstance(request.tokens[0], int)
                ):
                    if not all(isinstance(t, int) for t in request.tokens):
                        return error_response(
                            message=f"Invalid input: 'tokens' list must contain only integers. \
                                Found: {request.tokens}",
                            err_type="InvalidRequest",
                            status_code=HTTPStatus.BAD_REQUEST)
                    tokens_to_decode = [int(t) for t in request.tokens]
                    text = tokenizer.decode(tokens_to_decode,
                                            skip_special_tokens=request.skip_special_tokens)
                    text_to_return = text
                elif (
                    isinstance(request.tokens, list)
                    and request.tokens
                    and isinstance(request.tokens[0], list)
                ):
                    texts = []
                    for token_list in request.tokens:
                        if not all(isinstance(t, int) for t in token_list):
                            return error_response(message=f"Invalid input: 'tokens' \
                                                sublist must contain only integers. Found: {token_list}",
                                                err_type="InvalidRequest",
                                                status_code=HTTPStatus.BAD_REQUEST)
                        tokens_to_decode = [int(t) for t in token_list]
                        decoded_text = tokenizer.decode(tokens_to_decode,
                                                        skip_special_tokens=request.skip_special_tokens)
                        texts.append(decoded_text)
                    text_to_return = texts
                elif isinstance(request.tokens, list) and not request.tokens:
                    text_to_return = ""
                else:
                    return error_response(
                        message=f"Invalid input: 'tokens' must be List[int] \
                            or List[List[int]]. Found type: {type(request.tokens)}",
                        err_type="InvalidRequest",
                        status_code=HTTPStatus.BAD_REQUEST)
                return OCORJSONResponse(OCDetokenizeResponse(text=text_to_return).model_dump())
            except Exception as e:
                self.logger.error(f"Error during detokenization: {get_exception_traceback()}")
                if "decode" in str(e).lower():
                    self.logger.warning(
                        f"Detokenization decode warning/error: {e}. Input tokens shape/type: {type(request.tokens)}"
                    )
                    return error_response(
                        message=f"Error decoding tokens: {e}. \
                            Input tokens might be invalid for the model.",
                        err_type="DecodeError",
                        status_code=HTTPStatus.BAD_REQUEST,
                    )
                return error_response(
                    message=f"Internal server error during detokenization: {e}",
                    err_type="InternalServerError",
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                )

    async def score(self, request: OCScoringRequest, raw_request: OCRequest) -> OCResponse:
        """Handle scoring requests."""
        async with self.semaphore:
            return await self.openai_serving_score.handle_request(request, raw_request)

    async def rerank(self, request: OCRerankRequest, raw_request: OCRequest) -> OCResponse:
        """Handle reranking requests."""
        async with self.semaphore:
            return await self.openai_serving_rerank.handle_request(request, raw_request)

    async def pool(self, request: OCPoolingRequest, raw_request: OCRequest) -> OCResponse:
        """Handle pooling requests."""
        return error_response(
            message="Pooling endpoint is not implemented in SGLang server.",
            err_type="NotImplementedError",
            status_code=HTTPStatus.NOT_IMPLEMENTED,
        )

    async def embed(self, request: OCEmbeddingRequest, raw_request: OCRequest) -> OCResponse:
        """Handle embedding requests."""
        async with self.semaphore:
            return await self.openai_serving_embedding.handle_request(request, raw_request)

    async def get_metrics(self, request: OCRequest = None) -> OCResponse:
        """Handle metrics requests."""
        return OCResponse(generate_latest(self.metrics_registry),
                        headers={"Content-Type": CONTENT_TYPE_LATEST})
