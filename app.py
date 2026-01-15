from __future__ import annotations

import streamlit as st
from src import ui_utils
from src.state import KEYS, ensure_state


def main() -> None:
    st.set_page_config(page_title="Face Clustering Analyzer", layout="wide", initial_sidebar_state="expanded")
    ensure_state()
    ui_utils.load_app_style()

    st.title("Face Clustering Analyzer")
    
    st.markdown("### 🚀 功能导航")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**01 Home (主页)**\n\n数据加载、字段校验、全局统计概览、簇大小分布可视化")
        st.info("**02 Clustering (聚类)**\n\n执行 HAC / Infomap / KMeans 聚类算法，生成新标签列")
    with col2:
        st.info("**03 Analysis (分析)**\n\n多维度（大小/方差/散度）排序查看簇；支持 1v1 比对与 TopK 检索")
        st.info("**04 Annotation (标注)**\n\n基于分析结果进行人工标注：拆分不纯簇、合并相似簇")
    with col3:
        st.info("**05 Evaluation (评估)**\n\n计算 Pairwise F1, BCubed F1 等指标，评估聚类质量")

    st.divider()

    st.markdown("### ⏳ 当前状态")
    df = st.session_state.get(KEYS.df)
    if df is None:
        st.warning("🔴 尚未加载数据：请进入 **01_Home** 页面输入数据文件路径并加载。")
    else:
        st.success(f"🟢 当前已加载数据：{len(df):,} rows | path = `{st.session_state.get(KEYS.data_path,'')}`")

    st.divider()

    st.markdown("### 🔗 项目信息")
    c_info1, c_info2 = st.columns(2)
    with c_info1:
        st.markdown(
            """
            - **GitHub**: [ClusterFlow Repository](https://github.com/lidong-yin/ClusterFlow)
            - **文档**: 查看 `README.md` 获取详细说明
            """
        )
    with c_info2:
        st.markdown(
            """
            - **Version**: 2.0.0
            - **Email**: yld321@qq.com
            """
        )


if __name__ == "__main__":
    main()

