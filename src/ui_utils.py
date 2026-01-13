from __future__ import annotations

import os
import pandas as pd
import streamlit as st
from pathlib import Path
from typing import Any, Iterable, Optional


def load_css(path: str) -> None:
    if not path or not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except Exception:  # noqa: BLE001
        return


def load_app_style() -> None:
    """
    Load global CSS under repo root `assets/style.css`.

    Important: Call this AFTER `st.set_page_config(...)` in each page.
    """
    try:
        root = Path(__file__).resolve().parents[1]
        css_path = root / "assets" / "style.css"
        load_css(str(css_path))
    except Exception:  # noqa: BLE001
        return


def render_warnings(warnings: list[str]) -> None:
    for w in warnings:
        st.warning(w)


def render_errors(errors: list[str]) -> None:
    for e in errors:
        st.error(e)


def render_image_grid(
    df: pd.DataFrame,
    *,
    img_col: str = "img_url",
    obj_id_col: str = "obj_id",
    metadata_cols: Optional[list[str]] = None,
    ncols: int = 12,
    max_images: int = 120,
) -> None:
    if df is None or len(df) == 0:
        st.info("该组没有可展示的样本。")
        return
    
    # Defensive fix for PyArrow mixed-type inference errors (e.g. numeric obj_id vs string URL)
    # We force object columns to string to ensure safe serialization.
    # Note: We work on a copy to avoid side effects, though st.dataframe/images usually copy anyway.
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str)

    if img_col not in df.columns or obj_id_col not in df.columns:
        st.error(f"图片展示需要列: {img_col}, {obj_id_col}")
        return

    metadata_cols = metadata_cols or []
    ncols = int(max(1, ncols))
    max_images = int(max(1, max_images))
    sub = df.head(max_images)
    records = sub.to_dict(orient="records")

    for i in range(0, len(records), ncols):
        row = records[i : i + ncols]
        cols = st.columns(ncols)
        for c, rec in zip(cols, row):
            obj_id = rec.get(obj_id_col, "")
            img = rec.get(img_col, "")
            caption_lines = [f"{obj_id}"]
            for m in metadata_cols:
                if m in rec:
                    caption_lines.append(f"{m}={rec.get(m)}")
            caption = "\n".join(caption_lines)
            with c:
                try:
                    st.image(img, caption=caption, use_container_width=True)
                except Exception:  # noqa: BLE001
                    st.write(caption)
                    st.error("图片加载失败（路径/URL不可用或权限问题）。")


def columns_except(df: pd.DataFrame, exclude: Iterable[str]) -> list[str]:
    ex = set(exclude)
    return [c for c in df.columns if c not in ex]

