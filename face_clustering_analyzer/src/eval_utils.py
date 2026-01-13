from __future__ import annotations

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics.cluster import contingency_matrix


def pairwise_f1(gt_labels: np.ndarray, pred_labels: np.ndarray, sparse: bool = True) -> dict[str, float]:
    """
    Calculate Pairwise Precision, Recall, and F1.
    Matches the logic:
      tk = dot(c, c) - n
      pk = sum(col_sum^2) - n
      qk = sum(row_sum^2) - n
    """
    n_samples = gt_labels.shape[0]
    # contingency_matrix(labels_true, labels_pred)
    # Rows: True, Cols: Pred
    c = contingency_matrix(gt_labels, pred_labels, sparse=sparse)
    
    # tk: sum(n_ij^2) - N  (number of pairs in same cluster in BOTH true and pred)
    if sparse:
        # dot product of sparse matrix with itself (element-wise square sum)
        tk = np.dot(c.data, c.data) - n_samples
    else:
        tk = np.sum(c ** 2) - n_samples
        
    # pk: sum( (sum_i n_ij)^2 ) - N = sum(pred_cluster_size^2) - N
    # col sums (pred cluster sizes)
    pred_sizes = np.asarray(c.sum(axis=0)).ravel()
    pk = np.sum(pred_sizes ** 2) - n_samples
    
    # qk: sum( (sum_j n_ij)^2 ) - N = sum(true_cluster_size^2) - N
    # row sums (true cluster sizes)
    true_sizes = np.asarray(c.sum(axis=1)).ravel()
    qk = np.sum(true_sizes ** 2) - n_samples
    
    if pk == 0:
        avg_pre = 0.0
    else:
        avg_pre = tk / pk
        
    if qk == 0:
        avg_rec = 0.0
    else:
        avg_rec = tk / qk
        
    if (avg_pre + avg_rec) == 0:
        fscore = 0.0
    else:
        fscore = 2. * avg_pre * avg_rec / (avg_pre + avg_rec)
        
    return {
        "precision": float(avg_pre),
        "recall": float(avg_rec),
        "f1": float(fscore)
    }


def bcubed_f1(gt_labels: np.ndarray, pred_labels: np.ndarray) -> dict[str, float]:
    """
    Optimized BCubed Precision, Recall, F1.
    Reference logic:
      Precision for item i = |C(i) \cap L(i)| / |C(i)|
      Recall for item i    = |C(i) \cap L(i)| / |L(i)|
      where C(i) is pred cluster of i, L(i) is true cluster of i.
    
    Optimized calculation using contingency matrix:
      For each cell n_ij (true label i, pred label j):
        There are n_ij items in this intersection.
        For each of these n_ij items:
          Precision contribution = n_ij / |Pred_j|
          Recall contribution    = n_ij / |True_i|
      Total Precision = sum_{i,j} (n_ij * (n_ij / |Pred_j|)) / N
                      = sum_{i,j} (n_ij^2 / |Pred_j|) / N
      Total Recall    = sum_{i,j} (n_ij * (n_ij / |True_i|)) / N
                      = sum_{i,j} (n_ij^2 / |True_i|) / N
    """
    n_samples = gt_labels.shape[0]
    if n_samples == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    c = contingency_matrix(gt_labels, pred_labels, sparse=True)
    # c is sparse matrix (rows=true, cols=pred)
    # c.data contains n_ij
    # c.row/c.col indices (if COO) or computed from CSR
    
    # Convert to COO for easy iteration/vectorization if not already
    c = c.tocoo()
    
    n_ij = c.data.astype(np.float64)
    # n_ij_sq = n_ij ** 2
    
    # Calculate row sums (True sizes) and col sums (Pred sizes)
    # sum(axis=1) returns matrix (n_true, 1)
    true_sizes = np.array(c.sum(axis=1)).flatten()
    pred_sizes = np.array(c.sum(axis=0)).flatten()
    
    # Map each n_ij to its corresponding true_size and pred_size
    # c.row is index into true_sizes, c.col is index into pred_sizes
    row_sizes_aligned = true_sizes[c.row]
    col_sizes_aligned = pred_sizes[c.col]
    
    # Weighted average
    # Precision part: sum(n_ij^2 / pred_size_j) / N
    # Recall part:    sum(n_ij^2 / true_size_i) / N
    
    # n_ij * (n_ij / size)
    term_p = (n_ij * n_ij) / col_sizes_aligned
    term_r = (n_ij * n_ij) / row_sizes_aligned
    
    avg_pre = term_p.sum() / n_samples
    avg_rec = term_r.sum() / n_samples
    
    if (avg_pre + avg_rec) == 0:
        fscore = 0.0
    else:
        fscore = 2. * avg_pre * avg_rec / (avg_pre + avg_rec)
        
    return {
        "precision": float(avg_pre),
        "recall": float(avg_rec),
        "f1": float(fscore)
    }


def custom_eval(gt_labels: np.ndarray, pred_labels: np.ndarray) -> dict[str, float]:
    n_samples = len(gt_labels)
    if n_samples == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    df = pd.DataFrame({"gt": gt_labels, "pred": pred_labels})
    df["idx"] = np.arange(n_samples)
    
    pred_first_idx_map = df.groupby("pred")["idx"].first().to_dict()
    pred_counts_map = df["pred"].value_counts().to_dict()
    
    # Group by GT
    gt_groups = df.groupby("gt")
    
    TP = 0
    M = 0
    W = 0
    
    for gt_lb, group in gt_groups:
        nodes_id = set(group["idx"])
        W += len(group)
        
        if len(group) <= 1:
            continue

        preds_in_gt = group["pred"].values
        pred_counter = Counter(preds_in_gt)
        
        tp_list = []
        m_list = []
        
        for pred_c_id, num in pred_counter.most_common():
            pred_total_len = pred_counts_map.get(pred_c_id, 0)
            
            if pred_total_len == 1:
                continue # Ignore singletons

            cover_id = pred_first_idx_map.get(pred_c_id)
            if cover_id in nodes_id:
                tp_list.append(num)
                m_list.append(pred_total_len)
        
        if len(tp_list) > 0:
            tp = tp_list[0]
            main_c_count = 0
            for x in tp_list:
                if x == tp:
                    main_c_count += 1
                else:
                    break
            
            min_m = min(m_list[:main_c_count])
            TP += tp
            M += min_m
            
    if M == 0:
        P, R, F1 = 0.0, 0.0, 0.0
    else:
        P = TP / M
        R = TP / W
        F1 = (2 * P * R) / (P + R) if (P + R) > 0 else 0.0
        
    return {"precision": float(P), "recall": float(R), "f1": float(F1)}


def compute_basic_stats(labels: np.ndarray) -> dict[str, int]:
    # Drop NaNs before counting
    # Assuming labels handling NaN happens before calling this if needed, 
    # but unique() handles NaN as a unique value.
    u, counts = np.unique(labels, return_counts=True)
    return {
        "n_clusters": int(len(u)),
        "n_singletons": int(np.sum(counts == 1)),
        "max_size": int(np.max(counts)) if len(counts) > 0 else 0,
        "avg_size": int(np.mean(counts)) if len(counts) > 0 else 0,
    }
