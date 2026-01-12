from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import streamlit as st


@dataclass(frozen=True)
class SessionKeys:
    df: str = "df"
    df_rev: str = "df_rev"
    data_path: str = "data_path"
    data_format: str = "data_format"
    load_warnings: str = "load_warnings"
    load_error: str = "load_error"

    # Feature / Faiss related (built from current df)
    feature_col: str = "feature_col"
    ok_col: str = "ok_col"
    faiss_index: str = "faiss_index"
    faiss_row_indices: str = "faiss_row_indices"  # index_pos -> df.index
    faiss_features: str = "faiss_features"  # aligned to index_pos
    faiss_meta: str = "faiss_meta"  # dict: df_rev/feature_col/ok_only/use_gpu_prefer

    # UI preferences
    img_cols: str = "img_cols"
    group_range_start: str = "group_range_start"
    group_range_end: str = "group_range_end"
    analysis_cache: str = "analysis_cache"


KEYS = SessionKeys()


def ensure_state() -> None:
    """Initialize Streamlit session_state keys safely (idempotent)."""
    defaults: dict[str, Any] = {
        KEYS.df: None,
        KEYS.df_rev: 0,
        KEYS.data_path: "",
        KEYS.data_format: "",
        KEYS.load_warnings: [],
        KEYS.load_error: "",
        KEYS.feature_col: "feature",
        KEYS.ok_col: "ok",
        KEYS.faiss_index: None,
        KEYS.faiss_row_indices: None,
        KEYS.faiss_features: None,
        KEYS.faiss_meta: None,
        KEYS.img_cols: 12,
        KEYS.group_range_start: 1,
        KEYS.group_range_end: 50,
        KEYS.analysis_cache: {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def get_df():
    return st.session_state.get(KEYS.df)


def set_df(df, *, data_path: str, data_format: str, warnings: list[str]) -> None:
    st.session_state[KEYS.df] = df
    st.session_state[KEYS.data_path] = data_path
    st.session_state[KEYS.data_format] = data_format
    st.session_state[KEYS.load_warnings] = warnings
    st.session_state[KEYS.load_error] = ""
    st.session_state[KEYS.df_rev] = int(st.session_state.get(KEYS.df_rev, 0)) + 1

    # Invalidate derived resources when df changes
    st.session_state[KEYS.faiss_index] = None
    st.session_state[KEYS.faiss_row_indices] = None
    st.session_state[KEYS.faiss_features] = None
    st.session_state[KEYS.faiss_meta] = None
    st.session_state[KEYS.analysis_cache] = {}


def set_load_error(msg: str) -> None:
    st.session_state[KEYS.load_error] = msg


def get_feature_col() -> str:
    return str(st.session_state.get(KEYS.feature_col, "feature"))


def set_feature_col(col: str) -> None:
    st.session_state[KEYS.feature_col] = col
    # Invalidate index if feature col changes
    st.session_state[KEYS.faiss_index] = None
    st.session_state[KEYS.faiss_row_indices] = None
    st.session_state[KEYS.faiss_features] = None
    st.session_state[KEYS.faiss_meta] = None


def get_ok_col() -> str:
    return str(st.session_state.get(KEYS.ok_col, "ok"))


def set_faiss(index: Any, features, row_indices) -> None:
    st.session_state[KEYS.faiss_index] = index
    st.session_state[KEYS.faiss_features] = features
    st.session_state[KEYS.faiss_row_indices] = row_indices


def get_faiss() -> tuple[Optional[Any], Any, Any]:
    return (
        st.session_state.get(KEYS.faiss_index),
        st.session_state.get(KEYS.faiss_features),
        st.session_state.get(KEYS.faiss_row_indices),
    )


def bump_df_rev() -> int:
    st.session_state[KEYS.df_rev] = int(st.session_state.get(KEYS.df_rev, 0)) + 1
    # Any df mutation should invalidate caches
    st.session_state[KEYS.analysis_cache] = {}
    st.session_state[KEYS.faiss_meta] = None
    return int(st.session_state[KEYS.df_rev])


def get_df_rev() -> int:
    return int(st.session_state.get(KEYS.df_rev, 0))

