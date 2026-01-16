from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Any, Callable, Optional
from sklearn.cluster import KMeans, MiniBatchKMeans
from src import faiss_utils


@dataclass(frozen=True)
class ClusterResult:
    labels: np.ndarray
    meta: dict[str, Any]


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [1] * n

    def find(self, u: int) -> int:
        while self.parent[u] != u:
            self.parent[u] = self.parent[self.parent[u]]
            u = self.parent[u]
        return u

    def union(self, u: int, v: int) -> None:
        ru = self.find(u)
        rv = self.find(v)
        if ru == rv:
            return
        if self.rank[ru] > self.rank[rv]:
            self.parent[rv] = ru
        elif self.rank[ru] < self.rank[rv]:
            self.parent[ru] = rv
        else:
            self.parent[rv] = ru
            self.rank[ru] += 1


def hac_cluster(
    feats: np.ndarray,
    *,
    cluster_th: float,
    topk: int = 256,
    init_labels: Optional[np.ndarray] = None,
    use_gpu_prefer: bool = True,
    batch_size: int = 20000,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> ClusterResult:
    """
    HAC-like clustering: union neighbors with similarity > cluster_th.
    Similarity is cosine (IP on L2-normalized vectors).
    """
    if feats.ndim != 2:
        raise ValueError("feats 必须是二维矩阵 (N,D)。")
    n = int(feats.shape[0])
    if n == 0:
        return ClusterResult(labels=np.empty((0,), dtype=np.int32), meta={})

    # Normalize in-place (cosine via IP)
    feats = feats.astype(np.float32, copy=False)
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    feats /= norms
    index = faiss_utils.build_index_ip(feats, use_gpu_prefer=use_gpu_prefer)
    index.add(feats)

    uf = _UnionFind(n)

    if init_labels is not None:
        # Union nodes sharing the same init label (excluding NaN)
        label_to_nodes: dict[Any, list[int]] = {}
        for i, lb in enumerate(init_labels.tolist()):
            if lb is None:
                continue
            if isinstance(lb, float) and np.isnan(lb):
                continue
            label_to_nodes.setdefault(lb, []).append(i)
        for nodes in label_to_nodes.values():
            if len(nodes) <= 1:
                continue
            rep = nodes[0]
            for j in nodes[1:]:
                uf.union(rep, j)

    k = int(topk) + 1  # include self then remove
    total_batches = int(np.ceil(n / batch_size))
    batch_i = 0
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        sims, nbrs = index.search(feats[start:end], k)
        sims = sims[:, 1:]
        nbrs = nbrs[:, 1:]
        for i in range(sims.shape[0]):
            u = start + i
            for j in range(sims.shape[1]):
                if sims[i, j] > cluster_th:
                    uf.union(u, int(nbrs[i, j]))
                else:
                    # faiss returns sorted sims; can break early
                    break
        batch_i += 1
        if progress_callback:
            progress_callback(batch_i / max(1, total_batches), f"HAC 搜索: {batch_i}/{total_batches}")

    # Compress roots to contiguous labels
    root_to_label: dict[int, int] = {}
    labels = np.empty((n,), dtype=np.int32)
    next_lb = 0
    for i in range(n):
        r = uf.find(i)
        if r not in root_to_label:
            root_to_label[r] = next_lb
            next_lb += 1
        labels[i] = root_to_label[r]

    return ClusterResult(
        labels=labels,
        meta={
            "method": "hac",
            "cluster_th": float(cluster_th),
            "topk": int(topk),
            "num_clusters": int(next_lb),
        },
    )


def infomap_cluster(
    feats: np.ndarray,
    *,
    cluster_th: float,
    topk: int = 256,
    sparsification: bool = True,
    node_link: int = 20,
    use_gpu_prefer: bool = True,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> ClusterResult:
    """
    Infomap clustering on KNN graph edges with sim > cluster_th.
    Requires `infomap` package.
    """
    try:
        import infomap  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("未安装 infomap：请 `pip install infomap` 后重试。") from e

    feats = feats.astype(np.float32, copy=False)
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    feats /= norms
    index = faiss_utils.build_index_ip(feats, use_gpu_prefer=use_gpu_prefer)
    index.add(feats)
    if progress_callback:
        progress_callback(0.05, "Infomap: Faiss search ...")
    def _search_cb(frac: float, text: str) -> None:
        if progress_callback:
            progress_callback(0.05 + 0.20 * float(frac), text)

    sims, nbrs = faiss_utils.search_in_batches(
        index,
        feats,
        k=int(topk) + 1,
        progress_callback=_search_cb if progress_callback else None,
    )
    sims = sims[:, 1:]
    nbrs = nbrs[:, 1:]

    info = infomap.Infomap("--two-level --silent --seed 42 --num-trials 3", flow_model="undirected")

    singles: list[int] = []
    links = 0
    if sparsification:
        sims_count = np.sum(sims > cluster_th, axis=1)
        n_nodes = int(nbrs.shape[0])
        for k in range(n_nodes):
            count = int(sims_count[k])
            if count == 0:
                singles.append(k)
                continue
            num_edges_to_add = min(count, int(node_link))
            indices_to_add = np.linspace(0, count - 1, num=num_edges_to_add, dtype=int)
            for j in indices_to_add:
                nbr = int(nbrs[k, j])
                if k < nbr:
                    info.addLink(k, nbr, float(sims[k, j]))
                    links += 1
            if progress_callback and (k % 10000 == 0 or k == n_nodes - 1):
                progress_callback(0.25 + 0.50 * ((k + 1) / max(1, n_nodes)), f"Infomap: 构建图 {k+1}/{n_nodes}")
    else:
        n_nodes = int(nbrs.shape[0])
        for i in range(n_nodes):
            added = 0
            for j, nbr in enumerate(nbrs[i]):
                if sims[i, j] > cluster_th:
                    info.addLink(int(i), int(nbr), float(sims[i, j]))
                    links += 1
                    added += 1
                else:
                    break
            if added == 0:
                singles.append(int(i))
            if progress_callback and (i % 10000 == 0 or i == n_nodes - 1):
                progress_callback(0.25 + 0.50 * ((i + 1) / max(1, n_nodes)), f"Infomap: 构建图 {i+1}/{n_nodes}")

    if links > 0:
        if progress_callback:
            progress_callback(0.75, "Infomap: 运行算法 ...")
        info.run()
    else:
        if progress_callback:
            progress_callback(0.75, "Infomap: 无有效边，跳过算法运行 ...")

    lb2nodes: dict[int, list[int]] = {}
    for node in info.iterTree():
        lb2nodes.setdefault(int(node.moduleIndex()), []).append(int(node.physicalId))

    idx2lb: dict[int, int] = {}
    # infomap's tree has a special root module; keep behavior similar to appendix
    for lb, nodes in lb2nodes.items():
        if lb == 0:
            for u in nodes[2:]:
                idx2lb[u] = lb
        else:
            for u in nodes[1:]:
                idx2lb[u] = lb

    lb_len = len(lb2nodes)
    for k in singles:
        if k not in idx2lb:
            idx2lb[k] = lb_len
            lb_len += 1

    labels = np.array([idx2lb.get(i, -1) for i in range(nbrs.shape[0])], dtype=np.int32)
    missing = np.where(labels == -1)[0]
    if len(missing) > 0:
        for idx in missing:
            labels[idx] = lb_len
            lb_len += 1

    return ClusterResult(
        labels=labels,
        meta={
            "method": "infomap",
            "cluster_th": float(cluster_th),
            "topk": int(topk),
            "sparsification": bool(sparsification),
            "node_link": int(node_link),
            "num_clusters": int(len(np.unique(labels))),
            "links": int(links),
            "singles": int(len(singles)),
        },
    )


def kmeans_cluster(
    feats: np.ndarray,
    *,
    n_clusters: int,
    metric: str = "cosine",
    minibatch: bool = True,
    batch_size: int = 2048,
    random_state: int = 42,
    max_iter: int = 300,
    use_gpu_prefer: bool = True,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> ClusterResult:
    """
    KMeans clustering.
    - If faiss is available, use Faiss Kmeans (fast, gpu support, spherical).
    - Else fallback to sklearn (slow).
    
    - metric="cosine": normalize and run spherical kmeans (if faiss) or normalized euclidean (sklearn).
    - metric="l2": raw euclidean kmeans.
    """
    if metric not in {"cosine", "l2"}:
        raise ValueError("metric 必须是 cosine 或 l2。")
        
    x = feats.astype(np.float32, copy=False)
    d = x.shape[1]
    
    if metric == "cosine":
        x = faiss_utils.l2_normalize(x)

    # Try using Faiss first
    faiss_lib = faiss_utils.try_import_faiss()
    if faiss_lib is not None:
        if progress_callback:
            progress_callback(0.05, "KMeans: 初始化 Faiss ...")
            
        use_gpu = False
        if use_gpu_prefer:
            try:
                num_gpus = faiss_lib.get_num_gpus()
                use_gpu = (num_gpus > 0)
            except Exception:
                pass
        
        spherical = (metric == "cosine")
        if progress_callback:
            progress_callback(0.10, f"KMeans: Training (k={n_clusters}, gpu={use_gpu}, spherical={spherical}) ...")
            
        kmeans = faiss_lib.Kmeans(
            d, 
            int(n_clusters), 
            niter=int(max_iter), 
            verbose=False, 
            spherical=spherical, 
            gpu=use_gpu,
            seed=int(random_state),
            min_points_per_centroid=1,  # Allow small clusters for high-K scenarios to avoid warnings/errors
            max_points_per_centroid=10000000 # No upper limit
        )
        
        kmeans.train(x)
        
        if progress_callback:
            progress_callback(0.80, "KMeans: Assigning labels ...")
            
        # Assign labels: find nearest centroid
        # kmeans.index is automatically populated with centroids
        _, I = kmeans.index.search(x, 1)
        labels = I.flatten().astype(np.int32)
        
        inertia = float(kmeans.obj[-1]) if len(kmeans.obj) > 0 else np.nan
        
        return ClusterResult(
            labels=labels,
            meta={
                "method": "faiss_kmeans",
                "n_clusters": int(n_clusters),
                "metric": metric,
                "spherical": spherical,
                "gpu": use_gpu,
                "inertia": inertia,
            },
        )

    # Fallback to Sklearn
    if progress_callback:
        progress_callback(0.10, "KMeans: Faiss not found, using Sklearn (slow) ...")
        
    if minibatch:
        model = MiniBatchKMeans(
            n_clusters=int(n_clusters),
            batch_size=int(batch_size),
            random_state=int(random_state),
            max_iter=int(max_iter),
            n_init="auto",
        )
    else:
        model = KMeans(
            n_clusters=int(n_clusters),
            random_state=int(random_state),
            max_iter=int(max_iter),
            n_init="auto",
        )

    labels = model.fit_predict(x).astype(np.int32)
    return ClusterResult(
        labels=labels,
        meta={
            "method": "sklearn_kmeans",
            "n_clusters": int(n_clusters),
            "metric": metric,
            "minibatch": bool(minibatch),
            "inertia": float(getattr(model, "inertia_", np.nan)),
        },
    )

