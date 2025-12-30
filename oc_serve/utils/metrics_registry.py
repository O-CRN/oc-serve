"""Utility to get the Prometheus metrics registry, supporting multiprocess mode if configured."""
import os

from prometheus_client import REGISTRY, CollectorRegistry, multiprocess


def get_metrics_registry():
    """Get the Prometheus metrics registry, supporting multiprocess mode if configured."""
    registry = REGISTRY
    prometheus_multiproc_dir_path = os.getenv("PROMETHEUS_MULTIPROC_DIR", None)
    if prometheus_multiproc_dir_path is not None:
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
    return registry
