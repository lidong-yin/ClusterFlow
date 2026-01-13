from __future__ import annotations

import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class AnnotationConfig:
    obj_id_col: str = "obj_id"
    src_label_col: str = "gt_person_id"


def load_annotations_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def save_annotations_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)


def append_annotation_rows(
    ann_df: Optional[pd.DataFrame],
    *,
    rows: list[dict],
) -> pd.DataFrame:
    base = ann_df if ann_df is not None else pd.DataFrame()
    add = pd.DataFrame(rows)
    if len(base) == 0:
        return add
    return pd.concat([base, add], ignore_index=True)

