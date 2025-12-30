"""VLLM Server implementation."""
import os
from typing import Annotated

import asyncio
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.serving_pooling import OpenAIServingPooling
from vllm.entrypoints.openai.serving_score import ServingScores
from vllm.entrypoints.openai.serving_tokenization import OpenAIServingTokenization
from vllm.entrypoints.openai.serving_transcription import OpenAIServingTranscription
from vllm.entrypoints.openai.serving_models import OpenAIServingModels, BaseModelPath
from prometheus_client.openmetrics.exposition import CONTENT_TYPE_LATEST, generate_latest

from oc_serve.servers import Server
from oc_serve.utils import (
    oc_logger,
    get_metrics_registry,
    split_audio_by_time,
)
from oc_serve.api.models import (
    Form,
    Request,
    Response,
    StreamingResponse,
    JSONResponse,
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
    TranscribeResponseData,
    UsageInfoTranscriptionModels,
    SpeechResponse,
)
from configs import ServerConfigs

@Server.register("vllm")
class VLLM(Server):
    """
    VLLM Server implementation.
    """
    def __init__(self, server_configs: ServerConfigs):
        self.logger = oc_logger.get_logger("vllm")
        self.engine_args = server_configs
        self.logger.info("Starting AsyncLLM Engine with args: %s", self.engine_args)
        del os.environ['CUDA_VISIBLE_DEVICES']
        if int(self.engine_args.extra_args.vllm_use_v1):
            self.engine = AsyncLLM.from_engine_args(self.engine_args)
            model_config = self.engine.model_config
        else:
            self.engine = AsyncLLMEngine.from_engine_args(self.engine_args)
            model_config = self.engine.engine.get_model_config()
        base_model_paths = [BaseModelPath(
            name=self.engine_args.served_model_name \
                if self.engine_args.served_model_name is not None \
                else self.engine_args.model,
            model_path=self.engine_args.model
        )]
        self.openai_models = OpenAIServingModels(
            self.engine,
            model_config,
            base_model_paths=base_model_paths,
            lora_modules=self.engine_args.extra_args.lora_modules
        )
        self.instruction_server = OpenAIServingChat(
            self.engine,
            model_config,
            models=self.openai_models,
            response_role=self.engine_args.extra_args.response_role,
            chat_template=self.engine_args.extra_args.chat_template,
            request_logger=None,
            return_tokens_as_token_ids=self.engine_args.extra_args.return_tokens_as_token_ids,
            enable_auto_tools=self.engine_args.extra_args.enable_auto_tools,
            tool_parser=self.engine_args.extra_args.tool_parser,
            chat_template_content_format=self.engine_args.extra_args.chat_template_content_format
        )
        self.completion_server = OpenAIServingCompletion(
            self.engine,
            model_config,
            models=self.openai_models,
            request_logger=None,
            return_tokens_as_token_ids=self.engine_args.extra_args.return_tokens_as_token_ids
        )
        self.tokenization_server = OpenAIServingTokenization(
            self.engine,
            model_config,
            models=self.openai_models,
            request_logger=None,
            chat_template=self.engine_args.extra_args.chat_template,
            chat_template_content_format=self.engine_args.extra_args.chat_template_content_format
        )
        if int(self.engine_args.extra_args.vllm_enable_scoring):
            self.scoring_server = ServingScores(
                self.engine,
                model_config,
                models=self.openai_models,
                request_logger=None
            )
            self.logger.info("Scoring endpoint is ENABLED")
        else:
            self.logger.info("Scoring endpoint is DISABLED")
        self.pooling_server = None
        if int(self.engine_args.extra_args.vllm_enable_pooling):
            self.pooling_server = OpenAIServingPooling(
                self.engine,
                model_config,
                models=self.openai_models,
                request_logger=None,
                chat_template=self.engine_args.extra_args.chat_template,
                chat_template_content_format=self.engine_args.extra_args.chat_template_content_format
            )
            self.logger.info("Pooling/embeddings endpoint is ENABLED")
        else:
            self.logger.info("Pooling/embeddings endpoint is DISABLED")

        if int(self.engine_args.extra_args.use_transcribe_server):
            self.transcription_server = OpenAIServingTranscription(
                self.engine,
                model_config,
                models=self.openai_models,
                request_logger=None,
                return_tokens_as_token_ids=self.engine_args.extra_args.return_tokens_as_token_ids
            )
        self.skips = int(self.engine_args.extra_args.skips)
        self.semaphore = asyncio.Semaphore(int(self.engine_args.extra_args.max_concurrent_calls))
        self.metrics_registry = get_metrics_registry()


    async def check_model_health(self, raw_request: Request = None):
        async with self.semaphore:
            await self.engine.check_health()
            return Response(status_code=200,
                            content="Model is Healthy!")


    async def get_model_info(self, raw_request: Request = None):
        async with self.semaphore:
            models = await self.openai_models.show_available_models()
            return JSONResponse(content=models.model_dump())


    async def instruct(self, request: ChatCompletionRequest, raw_request: Request):
        async with self.semaphore:
            self.logger.info("Instruct Request")
            generator = await self.instruction_server.create_chat_completion(request,
                                                                             raw_request)
            if isinstance(generator, ErrorResponse):
                return JSONResponse(content=generator.model_dump(),
                                    status_code=generator.code)
            if request.stream:
                return StreamingResponse(content=generator,
                                         media_type="text/event-stream")

            assert isinstance(generator, ChatCompletionResponse)
            return JSONResponse(content=generator.model_dump())


    async def complete(self, request: CompletionRequest, raw_request: Request):
        async with self.semaphore:
            self.logger.info("Complete Request")
            generator = await self.completion_server.create_completion(request,
                                                                       raw_request)
            if isinstance(generator, ErrorResponse):
                return JSONResponse(content=generator.model_dump(),
                                    status_code=generator.code)
            if request.stream:
                return StreamingResponse(content=generator,
                                         media_type="text/event-stream")

            assert isinstance(generator, CompletionResponse)
            return JSONResponse(content=generator.model_dump())


    async def transcribe(self,
                         request: Annotated[TranscriptionRequest, Form()],
                         raw_request: Request):
        """Transcription endpoint handling audio transcription requests."""
        self.logger.info("Request Transcribe")
        if not bool(self.engine_args.extra_args.use_transcribe_server):
            return JSONResponse(content={"error": {"message": "It seems "
                                         "this model does not support transcription, "
                                         "or transcription is disabled on this server.",
                                         "type": "disabled_feature"}},
                                status_code=404)

        audio_data = await request.file.read()
        chunks, input_audio_duration = split_audio_by_time(audio_data)
        self.logger.debug("split into %d audio chunks", len(chunks))

        texts = []
        for i, chunk in enumerate(chunks):
            self.logger.debug("processing audio chunk %d", i)
            generator = await self.transcription_server.create_transcription(
                chunk, request, raw_request
            )
            if isinstance(generator, ErrorResponse):
                return JSONResponse(content=generator.model_dump(),
                                    status_code=generator.code)

            data = generator.model_dump()
            texts.append(data["text"])
            response = SpeechResponse(
                model=self.engine_args.model,
                data=[TranscribeResponseData(index=1, text=" ".join(texts))],
                usage=UsageInfoTranscriptionModels(transcription_tokens=0,
                                                   input_audio_duration=input_audio_duration),
            )

        return JSONResponse(content=response.model_dump(exclude_none=True))


    async def metrics(self, request: Request = None) -> Response:
        return Response(generate_latest(self.metrics_registry),
                        headers={"Content-Type": CONTENT_TYPE_LATEST})


    async def tokenize(self, request: TokenizeRequest, raw_request: Request):
        async with self.semaphore:
            self.logger.info("Tokenize Request")
            generator = await self.tokenization_server.create_tokenize(request,
                                                                       raw_request)
            if isinstance(generator, ErrorResponse):
                return JSONResponse(content=generator.model_dump(),
                                    status_code=generator.code)
            assert isinstance(generator, TokenizeResponse)
            return JSONResponse(content=generator.model_dump())


    async def scoring(self, request: ScoreRequest, raw_request: Request):
        if not int(self.engine_args.extra_args.vllm_enable_scoring) or self.scoring_server is None:
            return JSONResponse(content={"error": {"message": "Scoring is disabled on this server.",
                                   "type": "disabled_feature"}},
                                status_code=404)
        async with self.semaphore:
            self.logger.info("Scoring Request")
            generator = await self.scoring_server.create_score(request,
                                                               raw_request)
            if isinstance(generator, ErrorResponse):
                return JSONResponse(content=generator.model_dump(),
                                    status_code=generator.code)
            assert isinstance(generator, ScoreResponse)
            return JSONResponse(content=generator.model_dump())


    async def pooling(self, request: PoolingRequest, raw_request: Request):
        if not int(self.engine_args.extra_args.vllm_enable_pooling) or self.pooling_server is None:
            return JSONResponse(content={"error": {"message": "Pooling is disabled on this server.",
                                                   "type": "disabled_feature"}},
                                status_code=404)
        async with self.semaphore:
            self.logger.info("Pooling Request")
            generator = await self.pooling_server.create_pooling(request,
                                                                 raw_request)
            if isinstance(generator, ErrorResponse):
                return JSONResponse(content=generator.model_dump(),
                                    status_code=generator.code)
            assert isinstance(generator, PoolingResponse)
            return JSONResponse(content=generator.model_dump())


    async def detokenize(self, request: DetokenizeRequest, raw_request: Request):
        async with self.semaphore:
            self.logger.info("Detokenize Request")
            generator = await self.tokenization_server.create_detokenize(request,
                                                                         raw_request)
            if isinstance(generator, ErrorResponse):
                return JSONResponse(content=generator.model_dump(),
                                    status_code=generator.code)
            assert isinstance(generator, DetokenizeResponse)
            return JSONResponse(content=generator.model_dump())
