# OC-Serve

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

**OC-Serve** is an open-architecture, enterprise-grade model serving framework for deploying and managing AI models easily, efficiently, and scalably in production. It supports multiple AI model types including Large Language Models (LLMs), Large Vision Models (LVMs), Speech-to-Text (STT), and Text-to-Speech (TTS).

## Key Features

- **Multiple Serving Engines**: Support for vLLM, SGLang, TensorRT-LLM, and more
- **Cloud-Native**: Declarative deployment and management through simple YAML configuration
- **OpenAI-Compatible APIs**: Drop-in replacement with familiar API interfaces
- **Robust Orchestration**: Built on Ray for distributed computing and resource management
- **Auto-scaling**: Dynamic scaling based on workload demands
- **Flexible Configuration**: Easy-to-use YAML-based configuration system
- **Multi-Model Support**: Deploy and manage multiple models simultaneously

## Prerequisites for infrastructure

Before deploying OC-Serve, ensure you have the following components installed in your Kubernetes cluster:

- **Kubernetes**: Version 1.215 or higher
- **KubeRay**: If Ray will be used for distributed orchestration
- **GPU Operator**: For GPU resource management

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/O-CRN/oc-serve.git
cd oc-serve
```

2. Install using Helm:
```bash
helm install oc-serve ./helm -f your-values.yaml
```

### Configuration

Create a custom values file to define your AI models and resource requirements. Here's a basic example:

#### Example: Deploying Whisper STT Model

Create a file named `my-values.yaml`:

```yaml
applications:
  - name:  whisper
    configs:
      import_path: app:application
      route_prefix: /whisper
      deployments:
        - name: Ray
          num_replicas: 1
          max_ongoing_requests: 10
          max_queued_requests: 5
          user_config: {}
          ray_actor_options:
            num_cpus: 4
            num_gpus: 1
            accelerator_type: A4000
            runtime_env:
              env_vars:
                VLLM_MODEL: openai/whisper-tiny
                VLLM_EXTRA_VLLM_USE_V1: 1
                VLLM_EXTRA_USE_TRANSCRIBE_SERVER: 1
                VLLM_TENSOR_PARALLEL_SIZE: 1
                VLLM_GPU_MEMORY_UTILIZATION: "0.90"
                VLLM_DTYPE: bfloat16
                RAY_BACKEND_SERVER_TYPE: vllm
                OC_SERVE_ORCHESTRATOR_TYPE: ray
                HUGGING_FACE_HUB_TOKEN: <YOUR-HF-TOKEN>
                RAY_NAME: whisper

orchestrator:
  type: ray
  ray:
    workerGroups:
      - groupName: worker-group-01
        replicas: 1
        minReplicas: 1
        maxReplicas: 1
        nodeSelector: {}
        containers:
          - name: ctr-worker-01
            resources:
              limits:
                cpu: '4'
                memory: 8Gi
                gpu: 1
              requests:
                cpu: '2'
                memory: 4Gi
                gpu: 1
            volumeMounts: []
            env:
              - name: RAY_GCS_RPC_TIMEOUT_S
                value: '100000000000'
              - name: RAY_BACKEND_SERVER_TYPE
                value: vllm
              - name: OC_SERVE_ORCHESTRATOR_TYPE
                value: ray
```

### Deploy Your Application

```bash
helm install my-whisper-service ./helm -f my-values.yaml
```

## Configuration Guide

### Application Configuration

Each application in OC-Serve is defined with the following structure:

- **name**: Unique identifier for your application
- **configs**: Application-specific configurations
  - **import_path**: Python module path for the deployment
  - **route_prefix**: API endpoint prefix
  - **deployments**: List of deployment configurations
    - **num_replicas**: Number of replica instances
    - **max_ongoing_requests**: Maximum concurrent requests
    - **max_queued_requests**: Maximum queued requests
    - **ray_actor_options**: Ray-specific resource allocation

### Ray Orchestrator Configuration

Configure the Ray cluster based on your resource requirements:

- **workerGroups**: Define worker node groups with specific resources
- **replicas**: Number of worker nodes
- **resources**: CPU, memory, and GPU allocations
- **env**: Environment variables for runtime configuration

### Environment Variables

Key environment variables for model configuration:

#### Core Configuration
- `OC_SERVE_ORCHESTRATOR_TYPE`: Orchestrator type (ray)

#### vLLM Configuration
- `VLLM_MODEL`: Model identifier (HuggingFace model name or local path)
- `VLLM_EXTRA_VLLM_USE_V1`: Enable vLLM V1 engine (0 or 1)
- `VLLM_EXTRA_USE_TRANSCRIBE_SERVER`: Enable transcription server for STT models (0 or 1)
- `VLLM_TENSOR_PARALLEL_SIZE`: Number of GPUs for tensor parallelism
- `VLLM_GPU_MEMORY_UTILIZATION`: GPU memory utilization ratio (0.0-1.0)
- `VLLM_DTYPE`: Data type for inference (float16, bfloat16, float32, etc.)

#### Authentication & Deployment
- `HUGGING_FACE_HUB_TOKEN`: HuggingFace token for accessing gated models
- `RAY_NAME`: Ray deployment name identifier
- `RAY_BACKEND_SERVER_TYPE`: Backend serving engine (vllm, sglang, etc.)
- `RAY_GCS_RPC_TIMEOUT_S`: Ray GCS RPC timeout in seconds (default: large value for long-running tasks)

## Architecture

OC-Serve is built with a modular, extensible architecture:

```
┌───────────────────────────────────────────────────────────────┐
│                    OpenAI-Compatible API Layer                │
├───────────────────────────────────────────────────────────────┤
│                         OC-Serve Core                         │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │              Orchestrator Layer (Pluggable)             │  │
│  │                                                         │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │  │
│  │  │ Orchestrator │  │ Orchestrator │  │ Orchestrator │   │  │
│  │  │   (Ray)      │  │  (Custom)    │  │   (Future)   │   │  │
│  │  │              │  │              │  │              │   │  │
│  │  │ ┌──────────┐ │  │ ┌──────────┐ │  │ ┌──────────┐ │   │  │
│  │  │ │  Server  │ │  │ │  Server  │ │  │ │  Server  │ │   │  │
│  │  │ │  (vLLM)  │ │  │ │ (SGLang) │ │  │ │  (TRT)   │ │   │  │
│  │  │ └──────────┘ │  │ └──────────┘ │  │ └──────────┘ │   │  │
│  │  │ ┌──────────┐ │  │ ┌──────────┐ │  │              │   │  │
│  │  │ │  Server  │ │  │ │  Server  │ │  │              │   │  │
│  │  │ │ (Custom) │ │  │ │ (Custom) │ │  │              │   │  │
│  │  │ └──────────┘ │  │ └──────────┘ │  │              │   │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘   │  │
│  └─────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────┐
│              Kubernetes Infrastructure Layer                  │
└───────────────────────────────────────────────────────────────┘
```

**Key Components:**
- **API Layer**: OpenAI-compatible REST API endpoints
- **OC-Serve Core**: Central framework managing orchestrators and servers
- **Orchestrator Layer**: Pluggable orchestration engines (Ray, custom implementations)
- **Server Layer**: Multiple model serving backends per orchestrator (vLLM, SGLang, TensorRT-LLM, etc.)
- **Infrastructure**: Kubernetes-native deployment with auto-scaling and resource management

## API Usage

OC-Serve provides OpenAI-compatible APIs. Example usage:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://your-oc-serve-endpoint/whisper/v1",
    api_key="not-needed"
)

# Use the API as you would with OpenAI
response = client.audio.transcriptions.create(
    model="openai/whisper-large-v2",
    file=open("audio.mp3", "rb")
)
```

## Examples

### Deploying an LLM Model

```yaml
applications:
  - name: llama3
    configs:
      import_path: deployments.llama:llama_deployment
      route_prefix: /v1
      deployments:
        - name: llama-deployment
          num_replicas: 2
          ray_actor_options:
            num_cpus: 8
            num_gpus: 2
            accelerator_type: A100
            runtime_env:
              env_vars:
                VLLM_MODEL: meta-llama/Llama-3-8B
                VLLM_TENSOR_PARALLEL_SIZE: 2
                VLLM_GPU_MEMORY_UTILIZATION: '0.95'
```

### Project Structure

```
oc-serve/
├── configs/              # Configuration management
├── helm/                 # Kubernetes Helm charts
├── oc_serve/            # Core application code
│   ├── api/             # API endpoints and models
│   ├── orchestrators/   # Ray orchestration
│   ├── servers/         # Model serving backends
│   └── utils/           # Utilities and helpers
└── requirements.txt     # Python dependencies
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Support

For issues, questions, or contributions, please visit our [GitHub repository](https://github.com/O-CRN/oc-serve).

## Acknowledgments

OC-Serve leverages the following open-source projects:

- [Ray](https://github.com/ray-project/ray) - Distributed computing framework
- [vLLM](https://github.com/vllm-project/vllm) - High-performance LLM inference
- [KubeRay](https://github.com/ray-project/kuberay) - Ray on Kubernetes
