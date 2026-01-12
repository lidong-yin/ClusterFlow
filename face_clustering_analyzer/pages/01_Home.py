from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Home - Face Clustering Analyzer", layout="wide", initial_sidebar_state="expanded")

from src import data_utils, plot_utils, ui_utils
from src.state import KEYS, ensure_state, set_df, set_load_error

ui_utils.load_app_style()


def _render_load_panel() -> None:
    st.subheader("数据加载")
    with st.sidebar:
        st.header("Data")
        path = st.text_input(
            "数据文件路径 (pkl/pickle/parquet/csv)",
            value=st.session_state.get(KEYS.data_path, ""),
            placeholder="/path/to/data.parquet",
        )
        col1, col2 = st.columns([1, 1])
        with col1:
            do_load = st.button("加载数据", type="primary", use_container_width=True)
        with col2:
            clear = st.button("清空数据", use_container_width=True)

    if clear:
        st.session_state[KEYS.df] = None
        st.session_state[KEYS.df_rev] = 0
        st.session_state[KEYS.data_path] = ""
        st.session_state[KEYS.data_format] = ""
        st.session_state[KEYS.load_warnings] = []
        st.session_state[KEYS.load_error] = ""
        st.session_state[KEYS.faiss_index] = None
        st.session_state[KEYS.faiss_row_indices] = None
        st.session_state[KEYS.faiss_features] = None
        st.session_state[KEYS.faiss_meta] = None
        st.session_state[KEYS.analysis_cache] = {}
        st.rerun()

    if not do_load:
        return

    try:
        with st.spinner("正在加载数据，请稍候..."):
            prog_bar = st.progress(0, text="开始加载...")
            # We can't easily get fine-grained progress from pd.read_* 
            # but we can show status updates.
            prog_bar.progress(20, text="读取文件中...")
            df, fmt = data_utils.load_dataframe(path)
            
            prog_bar.progress(60, text="校验数据字段...")
            vr = data_utils.validate_dataframe(df)
            if not vr.ok:
                set_load_error("\n".join(vr.errors))
                st.error("数据校验失败：请修复后重试。")
                ui_utils.render_errors(vr.errors)
                if vr.warnings:
                    ui_utils.render_warnings(vr.warnings)
                return
            
            prog_bar.progress(90, text="初始化全局状态...")
            set_df(df, data_path=path, data_format=fmt, warnings=vr.warnings)
            prog_bar.progress(100, text="完成！")
            
        st.success(f"加载成功：{len(df):,} rows | format={fmt}")
        if vr.warnings:
            ui_utils.render_warnings(vr.warnings)
        st.rerun()
    except Exception as e:  # noqa: BLE001
        set_load_error(str(e))
        st.error(f"加载失败：{e}")


def _render_overview() -> None:
    df = st.session_state.get(KEYS.df)
    if df is None:
        st.info("请先在侧边栏输入文件路径并点击“加载数据”。")
        err = st.session_state.get(KEYS.load_error, "")
        if err:
            st.error(err)
        return

    st.subheader("全局统计")
    ok_col = data_utils.detect_ok_column(df)
    total = len(df)
    if ok_col:
        ok_cnt = int(df[ok_col].fillna(False).astype(bool).sum())
        bad_cnt = int(total - ok_cnt)
    else:
        ok_cnt = total
        bad_cnt = 0

    gt_clusters = int(df["gt_person_id"].nunique(dropna=False)) if "gt_person_id" in df.columns else 0
    cluster_cols = data_utils.detect_cluster_label_columns(df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("总节点数", f"{total:,}")
    c2.metric("有效节点数", f"{ok_cnt:,}")
    c3.metric("无效节点数", f"{bad_cnt:,}")
    c4.metric("GT 簇数量", f"{gt_clusters:,}")

    if cluster_cols:
        with st.expander("cluster_id* 簇数量统计", expanded=False):
            for c in cluster_cols:
                st.write(f"- `{c}`: {int(df[c].nunique(dropna=False)):,} clusters")
    else:
        st.info("未检测到 cluster_id* 列（这是可选字段）。")

    st.subheader("簇大小分布")
    
    c_chart_left, c_chart_right = st.columns(2)
    
    with c_chart_left:
        # 1. Custom Title Row
        st.markdown("**基准标签 (Ground Truth)**")
        # 2. Control Row (Collapsed label to ensure equal height)
        st.text_input("##gt_label_hidden", value="gt_person_id", disabled=True, label_visibility="collapsed")
        # 3. Chart
        if "gt_person_id" in df.columns:
            fig = plot_utils.plot_cluster_size_distribution(df["gt_person_id"], title="")
            st.plotly_chart(fig, use_container_width=True, key="chart_gt")
        else:
            st.warning("缺少 gt_person_id")

    with c_chart_right:
        sel_col = None
        # 1. Custom Title Row
        st.markdown("**选择对比标签 (Cluster ID)**")
        # 2. Control Row (Collapsed label)
        if cluster_cols:
            sel_col = st.selectbox(
                "##cluster_label_hidden", 
                options=cluster_cols, 
                index=0,
                label_visibility="collapsed"
            )
            # 3. Chart
            fig = plot_utils.plot_cluster_size_distribution(df[sel_col], title="")
            st.plotly_chart(fig, use_container_width=True, key="chart_cluster")
        else:
            st.text_input("##no_cluster_hidden", value="(无可用列)", disabled=True, label_visibility="collapsed")
            st.info("未检测到 cluster_id* 列")

    with st.expander("数据预览 (head)", expanded=False):
        st.dataframe(df.head(50), use_container_width=True)


def main() -> None:
    ensure_state()
    st.title("主页：数据集概览")
    st.caption("在此页面加载数据，并查看全局统计信息与簇大小分布。")
    _render_load_panel()
    _render_overview()


if __name__ == "__main__":
    main()

