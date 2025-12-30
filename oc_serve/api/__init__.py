"""OC-Serve API Package"""
from .api import RootAPI
from .models import *

root_api_app = RootAPI(
    api_kwargs={
        "title": "OC-Serve",
        "summary": "OC-Serve RESTful APIs",
        "description": (
            "OC-Serve is an open-source model serving framework designed "
            "to deploy and manage large language models (LLMs) efficiently and scalably."
        ),
        "version": "0.0.1-beta",
        "swagger_ui_parameters": {"defaultModelsExpandDepth": -1},
    }
)
