from __future__ import annotations

import os
import numpy as np
import streamlit as st
from typing import Any, Optional


def try_import_faiss():
    try:
        import faiss  # type: ignore

        return faiss
    except Exception:  # noqa: BLE001
        return None


def is_faiss_available() -> bool:
    return try_import_faiss() is not None


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms


def _get_gpu_ids_from_env() -> Optional[str]:
    v = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if v is None:
        return None
    v = str(v).strip()
    return v or None


def build_index_ip(
    feats: np.ndarray,
    *,
    use_gpu_prefer: bool = True,
    gpu_ids: Optional[str] = None,
    use_float16: bool = True,
    shard: bool = True,
) -> Any:
    """
    Build Faiss IP index for cosine similarity (expects L2-normalized feats).
    Uses GPU if available and requested; otherwise CPU.
    """
    faiss = try_import_faiss()
    if faiss is None:
        raise RuntimeError("Faiss 未安装：请安装 faiss-cpu 或 faiss-gpu 后重试。")

    feats = feats.astype(np.float32, copy=False)
    dim = int(feats.shape[1])

    # GPU path
    if use_gpu_prefer:
        try:
            num_gpus = faiss.get_num_gpus()
        except Exception:  # noqa: BLE001
            num_gpus = 0

        if gpu_ids is None:
            gpu_ids = _get_gpu_ids_from_env()

        if gpu_ids:
            # If env restricts GPUs, treat it as "some GPU is intended"
            # but we still rely on faiss.get_num_gpus() to check availability.
            pass

        if num_gpus > 0:
            if num_gpus == 1:
                res = faiss.StandardGpuResources()
                cfg = faiss.GpuIndexFlatConfig()
                cfg.device = 0
                cfg.useFloat16 = bool(use_float16)
                return faiss.GpuIndexFlatIP(res, dim, cfg)

            # Multi GPU
            cpu_index = faiss.IndexFlatIP(dim)
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = bool(use_float16)
            co.shard = bool(shard)
            return faiss.index_cpu_to_all_gpus(cpu_index, co=co)

    # CPU path
    return faiss.IndexFlatIP(dim)


@st.cache_resource(show_spinner=False)
def build_faiss_index_cached(
    feats: np.ndarray,
    *,
    use_gpu_prefer: bool = True,
    use_float16: bool = True,
) -> Any:
    index = build_index_ip(feats, use_gpu_prefer=use_gpu_prefer, use_float16=use_float16)
    index.add(feats)
    return index


def search_in_batches(
    index: Any,
    queries: np.ndarray,
    *,
    k: int,
    batch_size: int = 2048,
    progress_callback=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (sims, nbrs) for all queries."""
    queries = queries.astype(np.float32, copy=False)
    n = int(queries.shape[0])
    sims_all = np.empty((n, k), dtype=np.float32)
    nbrs_all = np.empty((n, k), dtype=np.int64)
    total_batches = int(np.ceil(n / batch_size))
    bi = 0
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        sims, nbrs = index.search(queries[start:end], k)
        sims_all[start:end] = sims
        nbrs_all[start:end] = nbrs
        bi += 1
        if progress_callback and (bi % 5 == 0 or bi == total_batches):
            progress_callback(bi / max(1, total_batches), f"Faiss search: {bi}/{total_batches}")
    return sims_all, nbrs_all

