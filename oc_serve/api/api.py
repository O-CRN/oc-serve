"""Root API Application for OC-Serve"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any


class RootAPI(FastAPI):
    """Root API Application for OC-Serve."""

    def __init__(self, api_kwargs: Dict[str, Any] | None = None):
        default_kwargs: Dict[str, Any] = {
            "title": "OC-Serve",
            "summary": "OC-Serve RESTful APIs",
            "description": (
                "OC-Serve is an open-source model serving framework designed "
                "to deploy and manage large language models (LLMs) efficiently "
                "and scalably."
            ),
            "version": "0.0.1-beta",
            "swagger_ui_parameters": {"defaultModelsExpandDepth": -1},
        }
        super().__init__(**{**default_kwargs, **(api_kwargs or {})})
        self._configure_middlewares()

    def _configure_middlewares(self) -> None:
        self.add_middleware(CORSMiddleware,
                            allow_origins=["*"],
                            allow_methods=["*"],
                            allow_headers=["*"],
                            allow_credentials=True)
