from __future__ import annotations

"""
Annotation utilities (V2.0 placeholder).

V2.0 docs describe:
- CSV I/O for annotations
- apply annotations to generate corrected data
- split/merge operations with UUID generation

This module intentionally keeps a minimal API surface now, so later work can
extend without touching pages too much.
"""

from dataclasses import dataclass
from typing import Optional

import pandas as pd


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

