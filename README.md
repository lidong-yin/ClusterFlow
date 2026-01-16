<p align="center">
  <img src="assets/ClusterFlow.png" width="80%" />
</p>

<p align="center">
  <strong>基于 Streamlit 的人脸聚类、可视化分析、标注与评估一站式工具。</strong>
</p>

<p align="center">
  集成了数据加载、聚类算法运行、结果分析、人工辅助标注（拆分/合并）以及评估等全流程功能。适合快速验证聚类效果、清洗数据或进行 Case 分析。
</p>

<p align="center">
  👉 <a href="ClusterFlow.md"><strong>查看详细用户手册 (User Manual)</strong></a>
</p>

---

## ✨ 核心特性

*   **模块化架构**：UI 与逻辑分离，基于 Streamlit 多页面应用 (MPA) 设计，易于扩展。
*   **高性能计算**：利用 **Faiss** (GPU/CPU) 加速大规模向量检索与 KNN 图构建。
*   **多算法支持**：内置 **HAC** (层次聚类)、**Infomap** (社群发现)、**KMeans** 等主流算法。
*   **交互式分析**：支持按簇大小、方差（纯度）、散度（跨簇相似度）等多维度排序查看。
*   **人工标注**：提供可视化的“拆分”与“合并”工具，直接修正聚类结果并保存。
*   **评估体系**：集成 Pairwise F1、BCubed F1 等标准评估指标。

---

## 🚀 快速开始

### 1. 环境准备

推荐使用 Python 3.10+ 环境。

```bash
# 1. 克隆项目
git clone https://github.com/lidong-yin/ClusterFlow.git
cd ClusterFlow

# 2. 安装基础依赖
pip install -r requirements.txt

# 3. (可选但推荐) 安装 Faiss 与 Infomap 以获得完整功能
# conda install -c conda-forge faiss-gpu  # 或 faiss-cpu
# pip install infomap
```

### 2. 启动应用

```bash
streamlit run app.py
```

---

## 📂 数据输入要求

支持 `pkl`, `pickle`, `parquet`, `csv` 格式文件。界面中需输入**服务器端绝对路径**进行加载。

### 必需字段

| 字段名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `obj_id` | `str` / `int` | 样本唯一标识 |
| `img_url` | `str` | 图片访问路径（URL 或本地路径） |
| `gt_person_id` | `str` / `int` | Ground Truth 身份/簇标签 |

### 可选字段

| 字段名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `feature` | `list` / `ndarray` | 人脸特征向量 (如 512维)，用于聚类/分析/检索 |
| `cluster_id*` | `int` | 任意聚类结果列（如 `cluster_id_hac`） |
| `ok` | `bool` | 样本有效性标记 (True/False) |

---

## 📚 目录结构

```text
.
├── app.py                  # 入口文件
├── pages/                  # Streamlit 页面
│   ├── 01_Home.py          # 数据概览
│   ├── 02_Clustering.py    # 算法执行
│   ├── 03_Analysis.py      # 分析可视化
│   ├── 04_Annotation.py    # 人工标注
│   └── 05_Evaluation.py    # 指标评估
├── src/                    # 核心逻辑
│   ├── clustering_utils.py # 聚类算法实现
│   ├── analysis_utils.py   # 分析逻辑
│   ├── faiss_utils.py      # Faiss 封装
│   └── ...
└── assets/                 # 静态资源
```

