from __future__ import annotations

import os
import time
import numpy as np
import pandas as pd
import streamlit as st
from typing import Any
from src import analysis_utils, annotation_utils, clustering_utils, data_utils, faiss_utils, ui_utils
from src.state import KEYS, ensure_state, get_df, get_feature_col, get_df_rev, bump_df_rev


def _get_anno_path(data_path: str) -> str:
    return annotation_utils.get_default_annotation_path(data_path)


def _render_sidebar(df: pd.DataFrame):
    st.sidebar.header("标注设置")
    
    cluster_cols = data_utils.detect_cluster_label_columns(df)
    cols = []
    if "gt_person_id" in df.columns:
        cols.append("gt_person_id")
    cols.extend(cluster_cols)
    if not cols:
        cols = list(df.columns)
    default_ix = 0
    target_col = st.sidebar.selectbox("待修正标签列", options=cols, index=default_ix)
    
    data_path = st.session_state.get(KEYS.data_path, "")
    default_anno = _get_anno_path(data_path)
    anno_path = st.sidebar.text_input("标注文件路径 (.csv)", value=default_anno)
    
    st.sidebar.divider()
    
    st.sidebar.subheader("模式与筛选")
    anno_type = st.sidebar.radio("标注类型", ["拆分 (Split)", "合并 (Merge)"], index=0)
    
    if anno_type == "拆分 (Split)":
        st.sidebar.info("拆分模式默认按 Variance 排序")
        sort_by = st.sidebar.selectbox("排序依据", ["Variance", "Size"], index=0)
    else:
        sort_by = "Scatter (散度)"
        st.sidebar.info("合并模式默认按 Min Sim 排序")

    # Scatter params for Merge
    sim_th = 0.55
    scatter_topk = 100
    dedup_scatter = True
    cand_limit = 3
    if anno_type == "合并 (Merge)":
        sim_th = st.sidebar.number_input("min sim th", -1.0, 1.0, 0.55, 0.01)
        scatter_topk = st.sidebar.number_input("sim topk", 10, 256, 100)
        cand_limit = st.sidebar.slider("每组显示候选簇数", min_value=1, max_value=20, value=3, step=1)
        dedup_scatter = st.sidebar.checkbox("散度结果去重 (A-B vs B-A)", value=True)
        
    start = st.sidebar.number_input("起始组 (1-based)", min_value=1, value=1)
    end = st.sidebar.number_input("结束组", min_value=1, value=5)
    
    st.sidebar.subheader("展示设置")
    img_cols = st.sidebar.slider("图片列数", 6, 20, 12)
    per_group_images = st.sidebar.slider("每簇预览图片数", 10, 200, 24)
    page_rows = st.sidebar.slider("展开时每页行数", 5, 30, 10)

    run_btn = st.sidebar.button("开始加载/计算", type="primary", use_container_width=True)
    
    return {
        "target_col": target_col,
        "anno_path": anno_path,
        "anno_type": anno_type,
        "sort_by": sort_by,
        "start": int(start),
        "end": int(end),
        "img_cols": int(img_cols),
        "per_group_images": int(per_group_images),
        "page_rows": int(page_rows),
        "sim_th": float(sim_th),
        "scatter_topk": int(scatter_topk),
        "dedup_scatter": bool(dedup_scatter),
        "cand_limit": int(cand_limit),
        "run_btn": run_btn
    }


def _run_sub_clustering(feats: np.ndarray, method: str, param: float | int) -> np.ndarray:
    n = feats.shape[0]
    if n < 2: return np.zeros(n, dtype=int)
    use_gpu = False 
    if method == "HAC":
        res = clustering_utils.hac_cluster(feats, cluster_th=float(param), topk=min(256, n), use_gpu_prefer=use_gpu)
    elif method == "Infomap":
        res = clustering_utils.infomap_cluster(feats, cluster_th=float(param), topk=min(256, n), sparsification=True, use_gpu_prefer=use_gpu)
    else: # KMeans
        res = clustering_utils.kmeans_cluster(feats, n_clusters=int(param), metric="cosine", minibatch=False, use_gpu_prefer=use_gpu)
    return res.labels


def _toggle_key(*parts: Any) -> str:
    return "anno_toggle::" + "::".join([str(p) for p in parts])


def _expanded_key(*parts: Any) -> str:
    return "anno_expanded::" + "::".join([str(p) for p in parts])


def _page_key(*parts: Any) -> str:
    return "anno_page::" + "::".join([str(p) for p in parts])


def _get_expand_state(*, total: int, collapsed_n: int, cfg: dict, key_parts: list[Any]) -> tuple[bool, int, int]:
    total = int(total)
    collapsed_n = int(collapsed_n)
    paginate_threshold = 500
    per_page = max(1, int(cfg["img_cols"]) * int(cfg["page_rows"]))
    if total <= paginate_threshold:
        per_page = total

    if total <= collapsed_n:
        return False, 1, per_page

    expanded_state_key = _expanded_key(get_df_rev(), *key_parts)
    expanded = bool(st.session_state.get(expanded_state_key, False))
    if not expanded:
        return False, 1, per_page

    num_pages = int(np.ceil(total / per_page))
    page_key = _page_key(get_df_rev(), *key_parts)
    page = int(st.session_state.get(page_key, 1))
    page = max(1, min(page, num_pages))
    return True, page, per_page


def _render_expand_trigger(*, total: int, collapsed_n: int, cfg: dict, key_parts: list[Any]) -> None:
    total = int(total)
    collapsed_n = int(collapsed_n)
    if total <= collapsed_n:
        return

    expanded_state_key = _expanded_key(get_df_rev(), *key_parts)
    expanded = bool(st.session_state.get(expanded_state_key, False))
    label = "收起" if expanded else f"查看全部 ({total:,})"

    c1, _ = st.columns([1, 5])
    with c1:
        if st.button(label, key=_toggle_key(get_df_rev(), *key_parts), use_container_width=True):
            st.session_state[expanded_state_key] = not expanded
            st.session_state[_page_key(get_df_rev(), *key_parts)] = 1
            st.rerun()

    if expanded and total > 500:
        per_page = max(1, int(cfg["img_cols"]) * int(cfg["page_rows"]))
        num_pages = int(np.ceil(total / per_page))
        if num_pages > 1:
            cur = int(st.session_state.get(_page_key(get_df_rev(), *key_parts), 1))
            st.slider(
                "页码",
                min_value=1,
                max_value=num_pages,
                value=cur,
                step=1,
                key=_page_key(get_df_rev(), *key_parts),
            )


def _slice_indices(indices: Any, start: int, end: int):
    return indices[start:end]


def _render_merge_candidate_block(df, c_gid, main_gid, cand_idx, group_idx, cfg):
    """
    Render a single candidate cluster with checkboxes and a dedicated merge button.
    """
    c_df = df[df[cfg["target_col"]] == c_gid]
    count = len(c_df)
    
    # Keys
    select_all_key = f"merge_sel_all_{group_idx}_{cand_idx}_{c_gid}"
    select_all_prev_key = f"{select_all_key}_prev"
    
    if select_all_key not in st.session_state:
        st.session_state[select_all_key] = True
    
    c1, c2 = st.columns([2, 8])
    with c1:
        select_all = st.checkbox("全选/全不选", key=select_all_key)
    with c2:
        st.markdown(f"**🟠 候选簇: {c_gid}** (包含 {count} 张图片)")
    
    # If select_all toggled, apply to all items (visible + hidden)
    if st.session_state.get(select_all_prev_key) is None:
        st.session_state[select_all_prev_key] = select_all
    if st.session_state.get(select_all_prev_key) != select_all:
        for oid in c_df["obj_id"].astype(str).tolist():
            item_key = f"merge_item_{group_idx}_{cand_idx}_{c_gid}_{oid}"
            st.session_state[item_key] = select_all
        st.session_state[select_all_prev_key] = select_all
    
    limit = int(cfg["per_group_images"])
    ncols = int(cfg["img_cols"])
    indices = c_df.index.to_list()
    
    expanded, page, per_page = _get_expand_state(
        total=count,
        collapsed_n=limit,
        cfg=cfg,
        key_parts=["merge_cand", group_idx, c_gid],
    )
    
    if not expanded:
        show_idx = indices[:limit]
    else:
        p_start = (page - 1) * per_page
        p_end = min(p_start + per_page, count)
        show_idx = _slice_indices(indices, p_start, p_end)
    
    # Grid Display
    items_show = df.loc[show_idx]
    records = items_show.to_dict(orient="records")
    
    # Render grid with per-item checkbox
    for i in range(0, len(records), ncols):
        cols = st.columns(ncols)
        for c, rec in zip(cols, records[i:i+ncols]):
            oid = str(rec.get("obj_id", ""))
            img = rec.get("img_url", "")
            gt_label = rec.get("gt_person_id", None)
            
            with c:
                try:
                    st.image(img, use_container_width=True)
                except Exception:  # noqa: BLE001
                    st.error("Img Error")
                
                item_key = f"merge_item_{group_idx}_{cand_idx}_{c_gid}_{oid}"
                if item_key not in st.session_state:
                    st.session_state[item_key] = True
                
                st.checkbox(
                    "合并",
                    key=item_key,
                    label_visibility="visible"
                )
                if gt_label is not None:
                    st.caption(f"{oid} | gt={gt_label}")
                else:
                    st.caption(f"{oid}")
    
    _render_expand_trigger(total=count, collapsed_n=limit, cfg=cfg, key_parts=["merge_cand", group_idx, c_gid])
    
    st.markdown("---")
    if st.button("✅ 确认合并到主簇", type="primary", key=f"merge_btn_{group_idx}_{cand_idx}_{c_gid}"):
        recs_to_save = []
        all_oids = [str(x) for x in c_df["obj_id"].tolist()]
        for oid in all_oids:
            item_key = f"merge_item_{group_idx}_{cand_idx}_{c_gid}_{oid}"
            if item_key not in st.session_state:
                st.session_state[item_key] = True
            if st.session_state.get(item_key, True):
                recs_to_save.append({
                    "time": time.time(),
                    "anno_type": "MERGE",
                    "obj_id": oid,
                    "source_cid": str(c_gid),
                    "target_cid": str(main_gid)
                })
        
        if recs_to_save:
            annotation_utils.append_annotation_rows(cfg["anno_path"], recs_to_save)
            st.success(f"已合并 {len(recs_to_save)} 张图片到簇 {main_gid}！")
            time.sleep(1.0)
            st.rerun()
        else:
            st.warning("未选中任何图片。")


def _render_image_grid_with_input(df, img_col, obj_id_col, default_label_col, key_prefix, ncols):
    records = df.to_dict(orient="records")
    for i in range(0, len(records), ncols):
        cols = st.columns(ncols)
        for c, rec in zip(cols, records[i:i+ncols]):
            oid = str(rec.get(obj_id_col, ""))
            img = rec.get(img_col, "")
            val_from_data = rec.get(default_label_col)
            display_val = int(val_from_data) if val_from_data is not None else 0
            gt_label = rec.get("gt_person_id", None)

            with c:
                try:
                    st.image(img, use_container_width=True)
                except:
                    st.error("Img Error")
                st.number_input(
                    "Cluster ID", value=display_val, min_value=0, max_value=99999999, step=1,
                    key=f"{key_prefix}_{oid}", label_visibility="collapsed"
                )
                if gt_label is not None:
                    st.caption(f"{oid} | gt={gt_label}")
                else:
                    st.caption(f"{oid}")


def _load_annotations_df(path: str) -> pd.DataFrame:
    anno_df = annotation_utils.load_or_create_annotations(path)
    if "obj_id" in anno_df.columns:
        anno_df["obj_id"] = anno_df["obj_id"].astype(str)
    return anno_df


def _clear_annotations_by_type(path: str, anno_type: str) -> int:
    if not path:
        return 0
    if not os.path.exists(path):
        return 0
    try:
        df = pd.read_csv(path)
    except Exception:
        return 0
    if "anno_type" not in df.columns:
        return 0
    before = len(df)
    df = df[df["anno_type"].astype(str) != str(anno_type)]
    removed = int(before - len(df))
    df.to_csv(path, index=False)
    return removed


def main():
    ensure_state()
    st.set_page_config(page_title="Annotation - Face Clustering Analyzer", layout="wide", initial_sidebar_state="expanded")
    ui_utils.load_app_style()
    st.title("标注：拆分与修正")
    st.caption("基于分析结果进行人工标注：拆分不纯簇、合并相似簇。")

    df = get_df()
    if df is None:
        st.info("请先在 Home 页面加载数据。")
        return

    cfg = _render_sidebar(df)
    
    # State key for calculation results
    calc_key = f"anno_calc_{get_df_rev()}_{cfg['target_col']}_{cfg['anno_type']}_{cfg['sort_by']}"
    
    if cfg["run_btn"]:
        st.session_state["last_anno_calc_key"] = calc_key
        st.session_state.pop(calc_key, None) 
    
    if st.session_state.get("last_anno_calc_key") != calc_key:
        st.info("👈 请在侧边栏配置参数并点击 **“开始加载/计算”** 按钮以加载待标注数据。")
        return

    feat_col = get_feature_col()
    if feat_col not in df.columns:
        st.error(f"特征列 {feat_col} 不存在。")
        return

    # --- Calculation with Progress ---
    @st.cache_data
    def _get_anno_data(_df, t_col, f_col, a_type, s_th, s_topk, dedup_scatter, _rev):
        ph = st.empty()
        prog = ph.progress(0.0, text="准备数据...")
        
        def cb(frac, msg):
            prog.progress(min(0.99, float(frac)), text=msg)
            
        feats, row_idx = data_utils.extract_feature_matrix(_df, feature_col=f_col, ok_only=False, progress_callback=cb)
        
        if a_type == "拆分 (Split)":
            prog.progress(0.5, text="计算方差...")
            gdf = analysis_utils.group_variances(_df, group_key=t_col, feats=feats, feats_row_indices=row_idx, progress_callback=cb)
            res = gdf.sort_values("variance", ascending=False).reset_index(drop=True)
            prog.progress(1.0, text="完成")
            ph.empty()
            return res, feats, row_idx
        else:
            # Merge (Scatter)
            prog.progress(0.2, text="计算散度 (Faiss)...")
            groups = analysis_utils.compute_scatter_groups(
                _df, group_key=t_col, feats=feats, feats_row_indices=row_idx,
                sim_th=s_th, sim_topk=s_topk, use_gpu_prefer=True, progress_callback=cb
            )
            # Filter groups with candidates
            groups = [g for g in groups if len(g.candidates) > 0]
            if dedup_scatter:
                groups = analysis_utils.deduplicate_scatter_groups(groups)
            groups.sort(key=lambda g: g.group_min_sim, reverse=True)
            prog.progress(1.0, text="完成")
            ph.empty()
            return groups, feats, row_idx

    data_res, all_feats, all_row_idx = _get_anno_data(
        df,
        cfg["target_col"],
        feat_col,
        cfg["anno_type"],
        cfg["sim_th"],
        cfg["scatter_topk"],
        cfg["dedup_scatter"],
        get_df_rev(),
    )
    
    feat_pos_map = pd.Series(np.arange(len(all_row_idx)), index=all_row_idx)
    
    # --- Rendering ---
    start, end = cfg["start"] - 1, cfg["end"] - 1
    end = min(end, len(data_res) - 1)
    
    with st.expander("标注信息", expanded=True):
        st.caption(f"当前标注文件: {cfg['anno_path']}")
        if st.session_state.get("anno_out_col") in (None, "", "cluster_anno"):
            st.session_state["anno_out_col"] = "cluster_id_anno"
        out_col = st.text_input("输出标签列名", key="anno_out_col")
        c_opt1, c_opt2 = st.columns(2)
        with c_opt1:
            normalize_labels = st.checkbox(
                "使用统一标签格式",
                value=False,
                help="勾选后会对应用后的标签统一编码：data['输出标签'] = pd.factorize(data['输出标签'])[0]",
            )
        with c_opt2:
            save_to_source = st.checkbox(
                "保存更新到源文件",
                value=False,
                help="勾选后会在应用标注时将更新结果保存回原始输入文件。",
            )
        if st.button("✅ 应用当前标注", type="primary", use_container_width=True, key="apply_annotations"):
            if not out_col.strip():
                st.warning("请输入输出标签列名。")
            else:
                anno_df = _load_annotations_df(cfg["anno_path"])
                if len(anno_df) == 0:
                    st.warning("标注文件为空，无法应用。")
                else:
                    try:
                        updated = annotation_utils.apply_annotations_to_df(
                            df,
                            anno_df,
                            out_col.strip(),
                            base_col=cfg["target_col"],
                        normalize_labels=normalize_labels,
                        )
                    except TypeError:
                        # Backward compatibility for older annotation_utils signature
                        if cfg["target_col"] in df.columns:
                            df[out_col.strip()] = df[cfg["target_col"]].astype(str)
                        updated = annotation_utils.apply_annotations_to_df(
                            df,
                            anno_df,
                            out_col.strip(),
                        )
                        if normalize_labels:
                            df[out_col.strip()] = pd.factorize(df[out_col.strip()])[0]
                    st.session_state[KEYS.df] = df
                    bump_df_rev()
                    st.session_state["last_apply_info"] = {
                        "col": out_col.strip(),
                        "count": updated,
                        "data_path": str(st.session_state.get(KEYS.data_path, "")),
                    }
                    if save_to_source:
                        data_path = str(st.session_state.get(KEYS.data_path, ""))
                        if not data_path:
                            st.warning("保存失败：未找到原始输入文件路径。")
                        else:
                            try:
                                data_utils.save_dataframe(df, data_path)
                                st.success(f"已保存更新到源文件: `{data_path}`")
                            except Exception as e:  # noqa: BLE001
                                st.error(f"保存失败：{e}")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("清空拆分标注", use_container_width=True, key="clear_split_anno"):
                removed = _clear_annotations_by_type(cfg["anno_path"], "SPLIT")
                st.success(f"已清空拆分标注 {removed} 条。")
                st.rerun()
        with c2:
            if st.button("清空合并标注", use_container_width=True, key="clear_merge_anno"):
                removed = _clear_annotations_by_type(cfg["anno_path"], "MERGE")
                st.success(f"已清空合并标注 {removed} 条。")
                st.rerun()
        anno_df = _load_annotations_df(cfg["anno_path"])
        if len(anno_df) == 0:
            st.info("标注文件为空。")
        else:
            st.dataframe(anno_df, use_container_width=True)

    st.info(f"模式: {cfg['anno_type']} | 显示: {start+1} - {end+1} (共 {len(data_res)} 组)")
    last_apply = st.session_state.get("last_apply_info")
    if isinstance(last_apply, dict):
        st.success(f"已应用标注到列 `{last_apply.get('col')}`，更新 {last_apply.get('count', 0)} 条样本。")
        if last_apply.get("data_path"):
            st.caption(f"数据文件: `{last_apply.get('data_path')}`")
    st.divider()
    
    # --- SPLIT MODE ---
    for i in range(start, end + 1):
        if cfg["anno_type"] == "拆分 (Split)":
            item = data_res.iloc[i]
            gid = item[cfg["target_col"]]
            size = int(item["size"])
            var_val = item.get("variance", 0.0)
            
            with st.container(border=True):
                st.markdown(
                    f"""
                    <div class="cluster-header">
                      <div><div class="cluster-title">#{i+1} Group: {gid}</div></div>
                      <div class="chips"><span class="chip">size={size:,}</span> <span class="chip">var={var_val:.4f}</span></div>
                    </div>
                    """, unsafe_allow_html=True
                )
                
                sub_df = df[df[cfg["target_col"]] == gid].copy()
                valid_idx = sub_df.index.intersection(all_row_idx)
                sub_feats = all_feats[feat_pos_map.reindex(valid_idx).values.astype(int)]
                
                with st.expander("🛠️ 拆分工具", expanded=True):
                    c1, c2, c3 = st.columns([1, 2, 1])
                    with c1:
                        method = st.selectbox(f"Method", ["Infomap", "HAC", "KMeans"], key=f"m_{i}")
                    with c2:
                        if method == "HAC": param = st.slider(f"Threshold", 0.0, 1.0, 0.48, 0.01, key=f"p_{i}")
                        elif method == "Infomap": param = st.slider(f"Threshold", 0.0, 1.0, 0.4, 0.01, key=f"p_{i}")
                        else: param = st.slider(f"K", 2, max(2, min(size, 50)), 2, 1, key=f"p_{i}")
                    
                    if len(sub_feats) > 0:
                        pred = _run_sub_clustering(sub_feats, method, param)
                        sub_df["_temp"] = 0
                        sub_df.loc[valid_idx, "_temp"] = pred
                        c3.metric("当前拆分", f"{len(np.unique(pred))} 簇")
                    else:
                        sub_df["_temp"] = 0
                
                with st.form(key=f"split_{i}"):
                    lbls = sorted(sub_df["_temp"].unique())
                    for lbl in lbls:
                        items = sub_df[sub_df["_temp"] == lbl]
                        st.markdown(f"**🔹 Sub-cluster {lbl}** <span style='color:gray'>({len(items)})</span>", unsafe_allow_html=True)
                        
                        show_n = cfg["per_group_images"]
                        if len(items) > show_n:
                            st.caption(f"Show {show_n}/{len(items)}")
                            items_show = items.head(show_n)
                        else:
                            items_show = items
                            
                        _render_image_grid_with_input(items_show, "img_url", "obj_id", "_temp", f"s_{i}_{lbl}", cfg["img_cols"])
                        st.markdown('<div style="margin-bottom: 1.2rem;"></div>', unsafe_allow_html=True)
                    
                    if st.form_submit_button("✅ 确认拆分"):
                        recs = []
                        for lbl in lbls:
                            items = sub_df[sub_df["_temp"] == lbl]
                            limit = cfg["per_group_images"]
                            show_items = items.head(limit)
                            for _, r in show_items.iterrows():
                                oid = str(r["obj_id"])
                                k = f"s_{i}_{lbl}_{oid}"
                                val = st.session_state.get(k, int(lbl))
                                recs.append((oid, val))
                            if len(items) > limit:
                                for _, r in items.iloc[limit:].iterrows():
                                    recs.append((str(r["obj_id"]), int(lbl)))
                        
                        if recs:
                            oids, ls = zip(*recs)
                            rows = annotation_utils.create_split_records(list(oids), str(gid), list(ls))
                            annotation_utils.append_annotation_rows(cfg["anno_path"], rows)
                            st.success("Saved!")
                            time.sleep(0.5)
                            st.rerun()

        # --- MERGE MODE ---
        else:
            g = data_res[i] # ScatterGroup
            gid = g.main_label
            
            with st.container(border=True):
                st.markdown(
                    f"""
                    <div class="cluster-header">
                      <div><div class="cluster-title">#{i+1} Main: {gid}</div></div>
                      <div class="chips"><span class="chip">size={g.main_size:,}</span> <span class="chip">min_sim={g.group_min_sim:.3f}</span></div>
                    </div>
                    """, unsafe_allow_html=True
                )
                
                main_df = df[df[cfg["target_col"]] == gid]
                st.markdown(f"##### 🟢 主簇: {gid} <span style='color:grey;font-size:0.9em'>(size={len(main_df):,})</span>", unsafe_allow_html=True)
                
                main_indices = main_df.index.to_list()
                limit = int(cfg["per_group_images"])
                expanded, page, per_page = _get_expand_state(
                    total=len(main_indices),
                    collapsed_n=limit,
                    cfg=cfg,
                    key_parts=["merge_main", i, gid],
                )
                if not expanded:
                    show_idx = main_indices[:limit]
                else:
                    p_start = (page - 1) * per_page
                    p_end = min(p_start + per_page, len(main_indices))
                    show_idx = _slice_indices(main_indices, p_start, p_end)
                ui_utils.render_image_grid(
                    df.loc[show_idx],
                    ncols=cfg["img_cols"],
                    max_images=len(show_idx),
                    metadata_cols=[c for c in ["gt_person_id"] if c in df.columns],
                )
                _render_expand_trigger(total=len(main_indices), collapsed_n=limit, cfg=cfg, key_parts=["merge_main", i, gid])
                
                st.divider()
                st.markdown("#### 候选合并簇 (Candidates)")
                
                cand_limit = int(cfg.get("cand_limit", 3))
                cands = g.candidates[:cand_limit]
                
                # Render each candidate as a separate block with its own form
                for c_idx, cand in enumerate(cands):
                    c_gid = cand.label
                    # 调用渲染函数，支持全选/部分选
                    _render_merge_candidate_block(df, c_gid, gid, c_idx, i, cfg)
                    st.markdown('<div style="margin-bottom: 1.5rem;"></div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
