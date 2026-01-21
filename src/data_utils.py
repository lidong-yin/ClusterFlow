from __future__ import annotations

import os
import numpy as np
import pandas as pd
import streamlit as st
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional

CORE_REQUIRED_COLUMNS = ("obj_id", "img_url", "gt_person_id")


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    errors: list[str]
    warnings: list[str]


def _file_ext(path: str) -> str:
    return os.path.splitext(path)[1].lower().lstrip(".")


@st.cache_data(show_spinner=False)
def load_dataframe(path: str) -> tuple[pd.DataFrame, str]:
    """
    Load dataframe from server-side path.

    Returns:
        (df, fmt) where fmt in {"csv","parquet","pkl"}
    """
    if not path or not isinstance(path, str):
        raise ValueError("数据路径为空或非法。请输入服务器端文件路径。")
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")
    if os.path.isdir(path):
        raise IsADirectoryError(f"路径是目录而不是文件: {path}")

    ext = _file_ext(path)
    if ext in {"pkl", "pickle"}:
        obj = pd.read_pickle(path)
        if not isinstance(obj, pd.DataFrame):
            try:
                obj = pd.DataFrame(obj)
            except Exception as e:
                raise TypeError(f"PKL 加载成功，但内容不是 DataFrame 且无法转换: {type(obj)}") from e
        
        # Enforce string type for critical columns to avoid PyArrow mixed type inference issues
        for c in ["obj_id", "img_url", "gt_person_id"]:
            if c in obj.columns:
                obj[c] = obj[c].astype(str)
        return obj, "pkl"

    if ext in {"parquet"}:
        obj = pd.read_parquet(path)
        if not isinstance(obj, pd.DataFrame):
            try:
                obj = pd.DataFrame(obj)
            except Exception as e:
                raise TypeError(f"PARQUET 加载成功，但内容不是 DataFrame 且无法转换: {type(obj)}") from e
        
        for c in ["obj_id", "img_url", "gt_person_id"]:
            if c in obj.columns:
                obj[c] = obj[c].astype(str)
        return obj, "parquet"

    if ext in {"csv"}:
        obj = pd.read_csv(path)
        if not isinstance(obj, pd.DataFrame):
            try:
                obj = pd.DataFrame(obj)
            except Exception as e:
                raise TypeError(f"CSV 加载成功，但内容不是 DataFrame 且无法转换: {type(obj)}") from e
        
        for c in ["obj_id", "img_url", "gt_person_id"]:
            if c in obj.columns:
                obj[c] = obj[c].astype(str)
        return obj, "csv"

    raise ValueError(f"不支持的文件格式: .{ext} (仅支持 pkl/pickle/parquet/csv)")


def validate_dataframe(df: pd.DataFrame) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    if df is None or not isinstance(df, pd.DataFrame):
        return ValidationResult(False, ["输入不是 pandas.DataFrame。"], [])

    # Required columns
    missing = [c for c in CORE_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"缺少必需字段: {missing}. 必须包含 {list(CORE_REQUIRED_COLUMNS)}")

    # Basic type sanity checks (warn-only; do not mutate)
    if "obj_id" in df.columns:
        if df["obj_id"].isna().any():
            warnings.append("obj_id 存在缺失值（NaN），可能导致搜索/标注定位不稳定。")
    if "img_url" in df.columns:
        if df["img_url"].isna().any():
            warnings.append("img_url 存在缺失值（NaN），对应样本图片可能无法展示。")
    if "gt_person_id" in df.columns:
        if df["gt_person_id"].isna().any():
            warnings.append("gt_person_id 存在缺失值（NaN），会影响按 GT 分组统计。")

    # Feature column checks
    if "feature" in df.columns:
        # Avoid full column copy/scan if possible
        first_valid_idx = df["feature"].first_valid_index()
        if first_valid_idx is None:
            warnings.append("feature 列存在但全为空；方差/散度/相似度功能将不可用。")
        else:
            sample = df.at[first_valid_idx, "feature"]
            if not isinstance(sample, (list, tuple, np.ndarray)):
                warnings.append(
                    f"feature 列的元素类型看起来不是 list/ndarray (示例: {type(sample)}). "
                    "后续特征解析可能失败。"
                )
    else:
        warnings.append("未检测到 feature 列；方差/散度/相似度/聚类功能将受限。")

    return ValidationResult(len(errors) == 0, errors, warnings)


def detect_cluster_label_columns(df: pd.DataFrame) -> list[str]:
    """Detect columns like cluster_id* (excluding gt_person_id)."""
    if df is None or not isinstance(df, pd.DataFrame):
        return []
    cols: list[str] = []
    for c in df.columns:
        if c == "gt_person_id":
            continue
        if c.startswith("cluster_id"):
            cols.append(c)
    return cols


def detect_ok_column(df: pd.DataFrame) -> Optional[str]:
    if df is None or not isinstance(df, pd.DataFrame):
        return None
    return "ok" if "ok" in df.columns else None


def _parse_feature_vector(x: Any) -> Optional[np.ndarray]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if isinstance(x, np.ndarray):
        if x.ndim == 1:
            return x
        return x.reshape(-1)
    if isinstance(x, (list, tuple)):
        return np.asarray(x)
    return None


def extract_feature_matrix(
    df: pd.DataFrame,
    *,
    feature_col: str = "feature",
    ok_col: Optional[str] = "ok",
    ok_only: bool = True,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract float32 feature matrix (N,D) and row indices into df.index.

    - Filters rows where feature is missing/un-parseable
    - If ok_only and ok_col exists, keeps ok == True
    - Does not modify df
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("未加载数据。")
    if feature_col not in df.columns:
        raise ValueError(f"缺少特征列: {feature_col}")

    mask = pd.Series(True, index=df.index)
    if ok_only and ok_col and ok_col in df.columns:
        mask &= df[ok_col].fillna(False).astype(bool)

    rows = df.loc[mask, feature_col]
    feats_list: list[np.ndarray] = []
    row_idx: list[Any] = []
    bad = 0
    total = int(len(rows))
    for ridx, val in rows.items():
        vec = _parse_feature_vector(val)
        if vec is None:
            bad += 1
            continue
        if vec.ndim != 1:
            vec = vec.reshape(-1)
        feats_list.append(vec.astype(np.float32, copy=False))
        row_idx.append(ridx)
        if progress_callback and (len(feats_list) % 20000 == 0):
            progress_callback(min(0.99, len(feats_list) / max(1, total)), f"解析特征: {len(feats_list):,}/{total:,}")

    if len(feats_list) == 0:
        raise ValueError(
            "无法构建特征矩阵：有效 feature 为 0。"
            + (f"（过滤掉 {bad} 条无效 feature）" if bad else "")
        )

    # Ensure consistent dimensionality
    dims = {v.shape[0] for v in feats_list}
    if len(dims) != 1:
        raise ValueError(f"feature 维度不一致，检测到多个维度: {sorted(dims)}")

    feats = np.stack(feats_list, axis=0).astype(np.float32, copy=False)
    return feats, np.asarray(row_idx)


def save_dataframe(df: pd.DataFrame, path: str) -> None:
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("df 为空，无法保存。")
    if not path or not isinstance(path, str):
        raise ValueError("保存路径为空或非法。")

    ext = _file_ext(path)
    out_dir = os.path.dirname(path) or "."
    os.makedirs(out_dir, exist_ok=True)

    if ext in {"pkl", "pickle"}:
        df.to_pickle(path)
        return
    if ext == "parquet":
        df.to_parquet(path, index=False)
        return
    if ext == "csv":
        df.to_csv(path, index=False)
        return
    raise ValueError(f"不支持的保存格式: .{ext} (仅支持 pkl/pickle/parquet/csv)")

