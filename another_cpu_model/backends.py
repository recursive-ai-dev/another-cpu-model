from __future__ import annotations

import contextlib
import importlib.util
import io

import numpy as np

from .types import BackendAvailability


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def describe_numpy_backend() -> tuple[str, list[dict[str, object]]]:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        try:
            np.__config__.show()
        except Exception:
            pass
    blas = buffer.getvalue().strip() or "NumPy backend details unavailable"

    try:
        from threadpoolctl import threadpool_info

        pools = threadpool_info()
    except Exception:
        pools = []

    return blas, pools


def detect_backends() -> BackendAvailability:
    blas, pools = describe_numpy_backend()
    return BackendAvailability(
        has_hnswlib=_has_module("hnswlib"),
        has_faiss=_has_module("faiss"),
        has_hdbscan=_has_module("hdbscan"),
        blas=blas,
        threadpools=pools,
    )
