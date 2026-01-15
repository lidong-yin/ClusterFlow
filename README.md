# ClusterFlow

基于 **Streamlit** 的人脸聚类、可视化分析、标注与评估工具（模块化、可扩展、偏工程化实现）。

## 运行方式

1) 安装依赖（建议 Python 3.10+）：

```bash
pip install -r requirements.txt
```

2) 启动：

```bash
streamlit run app.py
```

## 数据要求

支持 `pkl / pickle / parquet / csv` 文件，界面中填写**服务器端路径**加载。需要将数据组织成以以下结构。

### 必需字段

- `obj_id`: 每个样本唯一标识
- `img_url`: 人脸图片访问路径（URL 或本地路径）
- `gt_person_id`: Ground Truth 身份/簇标签

### 可选字段

- `feature`: 人脸特征向量（list / np.ndarray），用于方差/散度/相似度等分析
- `cluster_id*`: 任意聚类结果标签列（如 `cluster_id`, `cluster_id_infomap` 等）
- `ok`: bool，有效样本标记


## 功能概览

- **Home**：加载数据、字段校验、全局统计、簇大小分布图（gt 与可选 cluster_id* 列）
- **Clustering**：HAC / Infomap / KMeans 三种聚类（需要基于`feature`列），写回标签列并支持保存
- **Analysis**：按簇大小/簇内方差/散度（跨簇高相似邻居）排序展示；支持 cluster/obj 搜索、TopK 相似样本、1v1 相似度
- **Annotation**：拆分或合并簇、修改标签列、保存修改
- **Evaluation**：计算 Pairwise F1, BCubed F1 等聚类指标

## Faiss 与 Infomap 说明（可选依赖）

优先使用 **Faiss**（IP + L2 normalize 即余弦相似度）来做 TopK 检索；若环境未安装 faiss，会给出清晰提示并禁用依赖 Faiss 的功能（如散度分析/TopK 相似）。

Infomap 依赖 `infomap` 包；未安装时同样会提示并禁用 Infomap 聚类。

## 目录结构

```
.
├── app.py
├── pages/
│   ├── 01_Home.py
│   ├── 02_Clustering.py
│   ├── 03_Analysis.py
│   ├── 04_Annotation.py
│   └── 05_Evaluation.py
├── src/
│   ├── state.py
│   ├── data_utils.py
│   ├── clustering_utils.py
│   ├── analysis_utils.py
│   ├── annotation_utils.py
│   ├── eval_utils.py
│   ├── faiss_utils.py
│   ├── plot_utils.py
│   └── ui_utils.py
└── assets/
    └── style.css
```

