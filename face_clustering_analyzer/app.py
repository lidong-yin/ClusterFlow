from __future__ import annotations

import streamlit as st

from src import ui_utils
from src.state import KEYS, ensure_state


def main() -> None:
    st.set_page_config(page_title="Face Clustering Analyzer", layout="wide", initial_sidebar_state="expanded")
    ensure_state()
    ui_utils.load_app_style()

    st.title("Face Clustering Analyzer")
    st.markdown(
        "一个用于 **人脸聚类结果分析/评估** 的 Streamlit 工具。请从左侧页面进入 Home 加载数据。"
    )

    df = st.session_state.get(KEYS.df)
    if df is None:
        st.info("尚未加载数据：请进入 **Home** 页面输入数据文件路径并加载。")
        return

    st.success(f"当前已加载数据：{len(df):,} rows | path = `{st.session_state.get(KEYS.data_path,'')}`")


if __name__ == "__main__":
    main()

