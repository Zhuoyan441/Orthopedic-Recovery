# -*- coding: utf-8 -*-
"""
融合决策模块（原型阶段）
- 加载 ICF、步态、IMU 模块输出
- 融合特征并计算动态阈值
- 输出融合决策 CSV
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

REQUIRED_ICF_COLUMNS = {
    "patient_id",
    "time_step",
    "rehab_phase",
    "icf_total",
    "rom",
    "vas",
}
REQUIRED_GAIT_COLUMNS = {"patient_id", "gait_abnormal_prob"}
REQUIRED_SENSOR_COLUMNS = {"patient_id", "action_type", "quality_score"}


@dataclass
class FusionConfig:
    icf_path: str
    gait_path: str
    sensor_path: str
    output_path: str


def _validate_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name} 缺少列: {sorted(missing)}")


def load_icf_data(data_path: str) -> pd.DataFrame:
    """加载 ICF 数据，并保留每位患者的最新记录。"""
    df = pd.read_csv(data_path)
    if df.empty:
        return df
    _validate_columns(df, REQUIRED_ICF_COLUMNS, "ICF")
    df = df.copy()
    df["time_step"] = pd.to_numeric(df["time_step"], errors="coerce")
    df = df.sort_values(["patient_id", "time_step"], ascending=[True, True])
    latest = df.groupby("patient_id", as_index=False).tail(1)
    return latest[["patient_id", "rehab_phase", "icf_total"]]


def load_gait_data(data_path: str) -> pd.DataFrame:
    """加载步态数据，并按患者聚合异常概率。"""
    df = pd.read_csv(data_path)
    if df.empty:
        return df
    _validate_columns(df, REQUIRED_GAIT_COLUMNS, "Gait")
    df = df.copy()
    df["gait_abnormal_prob"] = pd.to_numeric(df["gait_abnormal_prob"], errors="coerce")
    agg = df.groupby("patient_id", as_index=False)["gait_abnormal_prob"].mean()
    return agg


def load_sensor_data(data_path: str) -> pd.DataFrame:
    """加载 IMU 数据，并计算动作风险。"""
    df = pd.read_csv(data_path)
    if df.empty:
        return df
    _validate_columns(df, REQUIRED_SENSOR_COLUMNS, "Sensor")
    df = df.copy()
    df["quality_score"] = pd.to_numeric(df["quality_score"], errors="coerce")
    if df["quality_score"].max(skipna=True) > 1:
        df["quality_score"] = df["quality_score"] / 100.0
    df["quality_score"] = df["quality_score"].clip(lower=0, upper=1)
    df["action_risk_score"] = 1.0 - df["quality_score"]
    agg = df.groupby("patient_id", as_index=False)["action_risk_score"].mean()
    return agg


def _normalize_series(series: pd.Series) -> pd.Series:
    if series.dropna().empty:
        return pd.Series([0.5] * len(series), index=series.index)
    min_v = series.min(skipna=True)
    max_v = series.max(skipna=True)
    if pd.isna(min_v) or pd.isna(max_v) or min_v == max_v:
        return pd.Series([0.5] * len(series), index=series.index)
    return (series - min_v) / (max_v - min_v)


def _dynamic_threshold(phase: str, icf_norm: float) -> float:
    base_map = {"early": 0.55, "mid": 0.60, "late": 0.65}
    base = base_map.get(str(phase).lower(), 0.60)
    threshold = base + 0.10 * (icf_norm - 0.5)
    return float(max(0.30, min(0.85, threshold)))


def fuse_and_decide(
    icf_df: pd.DataFrame,
    gait_df: pd.DataFrame,
    sensor_df: pd.DataFrame,
) -> pd.DataFrame:
    """融合多源特征并输出风险等级。"""
    if icf_df.empty and gait_df.empty and sensor_df.empty:
        return pd.DataFrame(
            columns=[
                "patient_id",
                "rehab_phase",
                "icf_total",
                "gait_abnormal_prob",
                "action_risk_score",
                "final_risk_level",
            ]
        )

    merged = pd.merge(icf_df, gait_df, on="patient_id", how="outer")
    merged = pd.merge(merged, sensor_df, on="patient_id", how="outer")

    merged["icf_total"] = pd.to_numeric(merged["icf_total"], errors="coerce")
    merged["gait_abnormal_prob"] = pd.to_numeric(merged["gait_abnormal_prob"], errors="coerce")
    merged["action_risk_score"] = pd.to_numeric(merged["action_risk_score"], errors="coerce")

    merged["gait_abnormal_prob"] = merged["gait_abnormal_prob"].fillna(0.5)
    merged["action_risk_score"] = merged["action_risk_score"].fillna(0.5)
    merged["icf_total"] = merged["icf_total"].fillna(merged["icf_total"].median())

    icf_norm = _normalize_series(merged["icf_total"])
    merged["_risk_score"] = (
        0.5 * merged["gait_abnormal_prob"]
        + 0.3 * merged["action_risk_score"]
        + 0.2 * (1.0 - icf_norm)
    )

    def _decide_row(row: pd.Series) -> str:
        threshold = _dynamic_threshold(row.get("rehab_phase"), icf_norm.loc[row.name])
        risk = row["_risk_score"]
        if risk >= threshold:
            return "high"
        if risk >= threshold - 0.10:
            return "medium"
        return "low"

    merged["final_risk_level"] = merged.apply(_decide_row, axis=1)

    return merged[
        [
            "patient_id",
            "rehab_phase",
            "icf_total",
            "gait_abnormal_prob",
            "action_risk_score",
            "final_risk_level",
        ]
    ]


def run_fusion_pipeline(config: FusionConfig) -> pd.DataFrame:
    """运行完整融合流程并保存结果。"""
    icf_df = load_icf_data(config.icf_path)
    gait_df = load_gait_data(config.gait_path)
    sensor_df = load_sensor_data(config.sensor_path)
    output_df = fuse_and_decide(icf_df, gait_df, sensor_df)
    output_df.to_csv(config.output_path, index=False)
    return output_df
