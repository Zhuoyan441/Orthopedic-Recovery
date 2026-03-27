# -*- coding: utf-8 -*-
"""
多模态融合模块接口 (Streamlit Demo 版)
核心逻辑提取自 decision_advanced.ipynb
实现了对新版字段名的兼容，保留了动态融合策略。
"""
import os
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class FusionConfig:
    icf_path: str
    gait_path: str
    sensor_path: str
    output_dir: str


def load_and_align_data(config: FusionConfig) -> pd.DataFrame:
    """加载其他三个模块的数据，并按 patient_id 对齐最新的数据记录"""
    # 1. 加载 ICF 数据
    if os.path.exists(config.icf_path):
        icf_df = pd.read_csv(config.icf_path)
        # 兼容新版 assessment_date 和旧版 time_step
        sort_col = "assessment_date" if "assessment_date" in icf_df.columns else "time_step"
        if sort_col in icf_df.columns:
            icf_df = icf_df.sort_values(["patient_id", sort_col])
        icf_latest = icf_df.groupby("patient_id", as_index=False).tail(1)
        # 优先读取 pred_icf，如果没有则读 icf_total
        icf_score_col = "pred_icf" if "pred_icf" in icf_latest.columns else "icf_total"
        icf_latest = icf_latest[["patient_id", icf_score_col]].rename(columns={icf_score_col: "icf_score"})
    else:
        icf_latest = pd.DataFrame(columns=["patient_id", "icf_score"])

    # 2. 加载 Gait 步态数据
    if os.path.exists(config.gait_path):
        gait_df = pd.read_csv(config.gait_path)
        # 兼容新版 anomaly_prob 和旧版 gait_abnormal_prob
        g_col = "anomaly_prob" if "anomaly_prob" in gait_df.columns else "gait_abnormal_prob"
        gait_df[g_col] = pd.to_numeric(gait_df[g_col], errors="coerce")
        gait_agg = gait_df.groupby("patient_id", as_index=False)[g_col].mean()
        gait_agg = gait_agg.rename(columns={g_col: "gait_score"})
    else:
        gait_agg = pd.DataFrame(columns=["patient_id", "gait_score"])

    # 3. 加载 Sensor IMU 数据
    if os.path.exists(config.sensor_path):
        sensor_df = pd.read_csv(config.sensor_path)
        s_col = "quality_score"
        if s_col in sensor_df.columns:
            sensor_df[s_col] = pd.to_numeric(sensor_df[s_col], errors="coerce")
            sensor_agg = sensor_df.groupby("patient_id", as_index=False)[s_col].mean()
            sensor_agg = sensor_agg.rename(columns={s_col: "sensor_score"})
        else:
            sensor_agg = pd.DataFrame(columns=["patient_id", "sensor_score"])
    else:
        sensor_agg = pd.DataFrame(columns=["patient_id", "sensor_score"])

    # 外连接合并
    merged = pd.merge(icf_latest, gait_agg, on="patient_id", how="outer")
    merged = pd.merge(merged, sensor_agg, on="patient_id", how="outer")

    return merged


# ==============================================================================
# 核心融合逻辑 (decision_advanced.ipynb)
# ==============================================================================
def advanced_fusion_predict(row: pd.Series) -> pd.Series:
    """
    基于 decision_advanced.ipynb 的高级策略：
    - 根据特征值域自适应归一化
    - 阶段感知的模态权重 (此处使用动态计算的权重代理 RL bandit 的输出)
    - 动态阈值判定
    """
    # 提取并清理特征 (处理空值)
    i_val = float(row.get("icf_score", np.nan))
    g_val = float(row.get("gait_score", np.nan))
    s_val = float(row.get("sensor_score", np.nan))

    i_val = 100.0 if np.isnan(i_val) else i_val
    g_val = 0.5 if np.isnan(g_val) else g_val
    s_val = 0.5 if np.isnan(s_val) else s_val

    # 1. 数值归一化 (兼容 ICF 的 0-200 分制)
    icf_norm = i_val / 200.0 if i_val > 10 else (
        1.0 / (1.0 + np.exp(-i_val)) if abs(i_val) > 1 else np.clip(i_val, 0, 1))
    gait_norm = 1.0 / (1.0 + np.exp(-g_val)) if abs(g_val) > 1 else np.clip(g_val, 0, 1)
    sensor_norm = 1.0 / (1.0 + np.exp(-s_val)) if abs(s_val) > 1 else np.clip(s_val, 0, 1)

    # 转换为风险维度：ICF越大越严重，Gait异常率越大越严重，Sensor质量分越低风险越大
    icf_risk = icf_norm
    gait_risk = gait_norm
    sensor_risk = 1.0 - sensor_norm

    # 2. 动态权重分配 (提取自 decision_advanced 的上下文策略)
    # 当步态异常极高时，赋予步态更高权重；当传感器动作极不规范时，增加传感器权重
    w_gait = 0.45 + (0.1 if gait_risk > 0.7 else 0)
    w_sensor = 0.30 + (0.1 if sensor_risk > 0.7 else 0)
    w_icf = 1.0 - w_gait - w_sensor
    # 确保权重合法
    w_icf = max(0.1, w_icf)
    total_w = w_gait + w_sensor + w_icf
    w_gait, w_sensor, w_icf = w_gait / total_w, w_sensor / total_w, w_icf / total_w

    # 3. 综合风险得分计算
    latent_risk = (w_gait * gait_risk) + (w_icf * icf_risk) + (w_sensor * sensor_risk)

    # 4. 动态阈值判定 (依据 decision_advanced.ipynb 中的 trace 记录计算)
    # 基础阈值随 ICF 基线浮动
    dynamic_threshold = 0.53 + 0.1 * (icf_risk - 0.5)
    dynamic_threshold = np.clip(dynamic_threshold, 0.40, 0.75)

    if latent_risk >= dynamic_threshold + 0.1:
        risk_level = "High"
    elif latent_risk >= dynamic_threshold:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return pd.Series({
        "icf_total": i_val,  # 保留原始值供 XAI 模块使用
        "gait_anomaly_prob": g_val,  # 保留原始值供 XAI 模块使用
        "imu_quality_mean": s_val,  # 保留原始值供 XAI 模块使用
        "risk_score": float(latent_risk),
        "dynamic_threshold": float(dynamic_threshold),
        "risk_level": risk_level
    })


# ==============================================================================

def generate_patient_report(patient_id: str, row_data: pd.Series, out_dir: str) -> dict:
    """生成符合要求的 JSON 报告"""
    # 组装 JSON
    report = {
        "patient_id": patient_id,
        "risk_score": round(row_data["risk_score"], 3),
        "risk_level": row_data["risk_level"],
        "dynamic_threshold_used": round(row_data["dynamic_threshold"], 3),
        "details": {
            "icf_raw": round(row_data["icf_total"], 3),
            "gait_raw": round(row_data["gait_anomaly_prob"], 3),
            "sensor_raw": round(row_data["imu_quality_mean"], 3)
        },
        "explain_one_line": f"经高级多模态模型融合(含动态阈值分配)，综合风险为 {row_data['risk_score']:.2f}，超过动态阈值 {row_data['dynamic_threshold']:.2f}，评估等级为 {row_data['risk_level']}。"
    }

    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"{patient_id}_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report


def run_fusion_pipeline(config: FusionConfig) -> pd.DataFrame:
    """供外部或 main.py 统一调用的管线入口"""
    # 1. 加载并对齐数据
    merged_df = load_and_align_data(config)
    if merged_df.empty:
        return pd.DataFrame()

    # 2. 执行 decision_advanced.ipynb 中的高级融合逻辑
    results = merged_df.apply(advanced_fusion_predict, axis=1)
    final_df = pd.concat([merged_df[["patient_id"]], results], axis=1)

    # 3. 结果保存
    os.makedirs(config.output_dir, exist_ok=True)
    final_df.to_csv(os.path.join(config.output_dir, "fusion_output.csv"), index=False)

    return final_df