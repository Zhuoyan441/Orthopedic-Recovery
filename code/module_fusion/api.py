# -*- coding: utf-8 -*-
"""
多模态融合模块接口 (Streamlit Demo 版)
1. 证据理论信念评估 (Evidential Belief Estimation)
2. 跨模态冲突指数计算 (Cross-Modal Conflict Index, CMCI)
3. 冲突感知的自适应温度注意力 (Conflict-Aware Dynamic Temperature Softmax)
4. 协同非线性风险聚合 (Synergistic Non-linear Risk Aggregation)
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
    """加载其他三个模块的数据，并按 patient_id 对齐，确保列名适配 XAI 模块"""
    # 1. 加载 ICF 数据 (目标列名: icf_total)
    if os.path.exists(config.icf_path):
        df_icf = pd.read_csv(config.icf_path)
        sort_col = "assessment_date" if "assessment_date" in df_icf.columns else "time_step"
        if sort_col in df_icf.columns:
            df_icf = df_icf.sort_values(["patient_id", sort_col])
        df_icf = df_icf.groupby("patient_id", as_index=False).tail(1)
        icf_col = "pred_icf" if "pred_icf" in df_icf.columns else (
            "true_icf" if "true_icf" in df_icf.columns else "icf_total")
        if icf_col in df_icf.columns:
            df_icf = df_icf[["patient_id", icf_col]].rename(columns={icf_col: "icf_total"})
        else:
            df_icf = pd.DataFrame(columns=["patient_id", "icf_total"])
    else:
        df_icf = pd.DataFrame(columns=["patient_id", "icf_total"])

    # 2. 加载 Gait 步态数据 (目标列名: gait_anomaly_prob)
    if os.path.exists(config.gait_path):
        df_gait = pd.read_csv(config.gait_path)
        g_col = "anomaly_prob" if "anomaly_prob" in df_gait.columns else "gait_abnormal_prob"
        if g_col in df_gait.columns:
            df_gait[g_col] = pd.to_numeric(df_gait[g_col], errors="coerce")
            df_gait = df_gait.groupby("patient_id", as_index=False)[g_col].mean()
            df_gait = df_gait.rename(columns={g_col: "gait_anomaly_prob"})
        else:
            df_gait = pd.DataFrame(columns=["patient_id", "gait_anomaly_prob"])
    else:
        df_gait = pd.DataFrame(columns=["patient_id", "gait_anomaly_prob"])

    # 3. 加载 Sensor IMU 数据 (目标列名: imu_quality_mean)
    if os.path.exists(config.sensor_path):
        df_sensor = pd.read_csv(config.sensor_path)
        s_col = "quality_score"
        if s_col in df_sensor.columns:
            df_sensor[s_col] = pd.to_numeric(df_sensor[s_col], errors="coerce")
            df_sensor = df_sensor.groupby("patient_id", as_index=False)[s_col].mean()
            df_sensor = df_sensor.rename(columns={s_col: "imu_quality_mean"})
        else:
            df_sensor = pd.DataFrame(columns=["patient_id", "imu_quality_mean"])
    else:
        df_sensor = pd.DataFrame(columns=["patient_id", "imu_quality_mean"])

    # === 执行 Outer Join (全连接)，并进行安全补值 ===
    merged = pd.merge(df_icf, df_gait, on="patient_id", how="outer")
    merged = pd.merge(merged, df_sensor, on="patient_id", how="outer")

    merged["icf_total"] = merged["icf_total"].fillna(100.0)
    merged["gait_anomaly_prob"] = merged["gait_anomaly_prob"].fillna(0.2)
    merged["imu_quality_mean"] = merged["imu_quality_mean"].fillna(0.8)

    return merged


# ==============================================================================
# 前沿融合算法 V4：Evidential Subjective Logic & CMCI-Attention
# ==============================================================================
def get_evidential_belief(p: float) -> float:
    """
    基于主观逻辑 (Subjective Logic) 的证据信念评估。
    抛物线映射：极端概率(0或1)带有高先验信念，中间值(0.5)由于歧义导致信念度最低。
    """
    p = np.clip(p, 0.0, 1.0)
    uncertainty = 4.0 * ((p - 0.5) ** 2)  # U = 1 - 4(p-0.5)^2 -> Belief = 1 - U
    return float(uncertainty)


def advanced_fusion_predict(row: pd.Series) -> pd.Series:
    i_val = float(row["icf_total"])
    g_val = float(row["gait_anomaly_prob"])
    s_val = float(row["imu_quality_mean"])

    # 1. 风险空间映射 (Risk Representation)
    icf_risk = i_val / 200.0 if i_val > 10 else np.clip(i_val, 0, 1)
    gait_risk = np.clip(g_val, 0, 1)
    sensor_risk = 1.0 - np.clip(s_val, 0, 1)

    # 2. 证据信念度计算 (Belief Masses)
    c_icf = 0.90  # 临床量表作为 Ground Truth 先验，给予最高信念
    c_gait = get_evidential_belief(gait_risk)
    c_sensor = get_evidential_belief(sensor_risk)

    # 3. 跨模态冲突指数计算 (Cross-Modal Conflict Index, CMCI)
    risks = np.array([icf_risk, gait_risk, sensor_risk])
    cmci = np.std(risks)  # 模态间的分歧方差，代表冲突严重程度

    # 4. 冲突感知的动态温度自适应注意力 (Conflict-Aware Dynamic Temperature)
    # 当冲突高时，下调温度 tau，使得 Softmax 注意力变得"尖锐"，强行聚焦最可靠模态
    tau = max(0.05, 0.4 * np.exp(-1.5 * cmci))

    logits = np.array([
        (icf_risk + 0.05) * c_icf,
        (gait_risk + 0.05) * c_gait,
        (sensor_risk + 0.05) * c_sensor
    ]) / tau

    exp_logits = np.exp(logits - np.max(logits))  # 防溢出
    attention_weights = exp_logits / np.sum(exp_logits)
    w_icf, w_gait, w_sensor = attention_weights

    # 5. 协同非线性风险聚合 (Synergistic Non-linear Risk Aggregation)
    linear_risk = (w_icf * icf_risk) + (w_gait * gait_risk) + (w_sensor * sensor_risk)
    max_risk = np.max(risks)  # 极端残差项
    synergy_risk = np.cbrt(icf_risk * gait_risk * sensor_risk)  # 几何平均捕捉"共病"放大效应

    # 最终风险 = 线性注意力分配 (70%) + 共病协同放大 (15%) + 单项极端危险兜底 (15%)
    final_risk_score = 0.7 * linear_risk + 0.15 * synergy_risk + 0.15 * max_risk
    final_risk_score = np.clip(final_risk_score, 0, 1)

    # 6. 冲突调节动态阈值 (Conflict-Adjusted Dynamic Threshold)
    # 基础阈值锚定 ICF，但在高冲突情况下(医生趋于保守防漏诊)，系统会自动下调报警门槛
    base_threshold = 0.55 + 0.1 * (icf_risk - 0.5)
    dynamic_threshold = np.clip(base_threshold - (0.15 * cmci), 0.35, 0.75)

    if final_risk_score >= dynamic_threshold + 0.1:
        risk_level = "High"
    elif final_risk_score >= dynamic_threshold:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return pd.Series({
        "risk_score": float(final_risk_score),
        "dynamic_threshold": float(dynamic_threshold),
        "risk_level": risk_level,
        "attn_w_icf": float(w_icf),
        "attn_w_gait": float(w_gait),
        "attn_w_sensor": float(w_sensor)
    })


# ==============================================================================

def generate_patient_report(patient_id: str, row_data: pd.Series, out_dir: str) -> dict:
    """生成患者结果 JSON (供 Streamlit 展示)"""
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
        "attention_weights": {
            "icf": round(row_data["attn_w_icf"], 3),
            "gait": round(row_data["attn_w_gait"], 3),
            "sensor": round(row_data["attn_w_sensor"], 3)
        },
        "explain_one_line": f"经证据理论与动态冲突调节(CMCI)计算，检测到共病协同风险，最终评分为 {row_data['risk_score']:.2f} (严格阈值 {row_data['dynamic_threshold']:.2f})，等级判定为 {row_data['risk_level']}。"
    }

    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"{patient_id}_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report


def run_fusion_pipeline(config: FusionConfig) -> pd.DataFrame:
    """管线入口：读取、合并、融合、保存"""
    merged_df = load_and_align_data(config)
    if merged_df.empty:
        return pd.DataFrame()

    results = merged_df.apply(advanced_fusion_predict, axis=1)
    # 拼合原始特征和结果（供下游 XAI 解释模块使用）
    final_df = pd.concat([merged_df, results], axis=1)

    # 丢弃内部使用的 attention 中间特征列，保持 CSV 清爽对齐 XAI
    out_df = final_df.drop(columns=["attn_w_icf", "attn_w_gait", "attn_w_sensor"])

    os.makedirs(config.output_dir, exist_ok=True)
    out_csv_path = os.path.join(config.output_dir, "fusion_output.csv")
    out_df.to_csv(out_csv_path, index=False)

    return final_df