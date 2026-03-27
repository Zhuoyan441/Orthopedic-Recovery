# -*- coding: utf-8 -*-
"""
多模态融合模块接口
1. 基于信息熵的不确定性估计 (Uncertainty Estimation)
2. 风险感知注意力机制 (Risk-Aware Attention via Softmax)
3. 极端风险残差保留 (Max-Risk Residual)
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
# 前沿融合算法：Uncertainty-Aware Cross-Modal Attention (2024-2025架构)
# ==============================================================================
def calculate_certainty(p: float) -> float:
    """计算模态的确信度 (基于归一化香农信息熵 Shannon Entropy)
    当概率 p 接近 0.5 时，熵最大，确信度最低 (噪音大)；接近 0 或 1 时，确信度最高。
    """
    p = np.clip(p, 1e-5, 1.0 - 1e-5)
    entropy = - (p * np.log2(p) + (1 - p) * np.log2(1 - p))
    return float(1.0 - entropy)


def advanced_fusion_predict(row: pd.Series) -> pd.Series:
    i_val = float(row["icf_total"])
    g_val = float(row["gait_anomaly_prob"])
    s_val = float(row["imu_quality_mean"])

    # 1. 风险空间映射 (Risk Representation)
    icf_risk = i_val / 200.0 if i_val > 10 else np.clip(i_val, 0, 1)
    gait_risk = np.clip(g_val, 0, 1)
    sensor_risk = 1.0 - np.clip(s_val, 0, 1)  # 质量越低，风险越高

    # 2. 模态确信度评估 (Uncertainty Estimation)
    # 临床量表(ICF)被视为高先验确信度，而传感器(Gait/IMU)的确信度动态计算
    c_icf = 0.85
    c_gait = calculate_certainty(gait_risk)
    c_sensor = calculate_certainty(sensor_risk)

    # 3. 风险感知注意力分配 (Risk-Aware Softmax Attention)
    # Logit = 风险严重程度 * 信息确信度
    tau = 0.3  # Temperature scaling (温度系数控制注意力的尖锐度)
    logits = np.array([
        (icf_risk + 0.1) * c_icf,
        (gait_risk + 0.1) * c_gait,
        (sensor_risk + 0.1) * c_sensor
    ]) / tau

    exp_logits = np.exp(logits - np.max(logits))  # 防溢出处理
    attention_weights = exp_logits / np.sum(exp_logits)
    w_icf, w_gait, w_sensor = attention_weights

    # 4. 注意力加权融合 + 极端风险残差连接 (Max-Risk Residual)
    # 防止多模态平均效应掩盖了某一个极度危险的信号
    attention_fused_risk = (w_icf * icf_risk) + (w_gait * gait_risk) + (w_sensor * sensor_risk)
    max_risk = max(icf_risk, gait_risk, sensor_risk)
    alpha = 0.15  # 残差系数

    final_risk_score = (1 - alpha) * attention_fused_risk + alpha * max_risk

    # 5. 基于临床先验的动态阈值判定
    dynamic_threshold = np.clip(0.55 + 0.1 * (icf_risk - 0.5), 0.40, 0.75)

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
        "explain_one_line": f"基于信息熵与风险感知注意力机制融合，系统将主要注意力分配给特定模态，最终综合风险评分为 {row_data['risk_score']:.2f} (阈值 {row_data['dynamic_threshold']:.2f})，等级为 {row_data['risk_level']}。"
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
    # 拼合原始特征和结果（供 XAI 解释模块��用）
    final_df = pd.concat([merged_df, results], axis=1)

    # 丢弃内部使用的 attention 中间特征列，保持 CSV 清爽
    out_df = final_df.drop(columns=["attn_w_icf", "attn_w_gait", "attn_w_sensor"])

    os.makedirs(config.output_dir, exist_ok=True)
    out_csv_path = os.path.join(config.output_dir, "fusion_output.csv")
    out_df.to_csv(out_csv_path, index=False)

    return final_df