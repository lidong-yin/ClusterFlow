from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Annotation - Face Clustering Analyzer", layout="wide", initial_sidebar_state="expanded")

from src import ui_utils
from src.state import ensure_state, get_df

ui_utils.load_app_style()


def main() -> None:
    ensure_state()
    st.title("标注（TODO）")

    df = get_df()
    if df is None:
        st.info("请先在 Home 加载数据。")
        return

    st.info("V2.0 文档中标注模块尚未实现（占位符）。后续会在此页面加入：CSV 标注读写、拆分/合并、应用标注等。")


if __name__ == "__main__":
    main()

