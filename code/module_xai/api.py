# -*- coding: utf-8 -*-
"""
多模态可解释性模块 (XAI)
完全对齐并读取 Fusion 模块的高级策略输出结果，计算特征的相对危险贡献度。
"""
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 修复 matplotlib 图表中文及负号无法正常显示的问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def load_inputs(input_path: str) -> pd.DataFrame:
    """
    读取 Fusion 模块输出的融合表 (fusion_output.csv)。
    期望包含 Fusion 模块算好的: risk_score, risk_level
    以及原始特征: icf_total, gait_anomaly_prob, imu_quality_mean
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"找不到融合数据: {input_path}")
    return pd.read_csv(input_path)


def pick_patient_row(df: pd.DataFrame, patient_id: str = None) -> pd.Series:
    if df.empty:
        raise ValueError("输入表为空，无法做XAI。")
    if "patient_id" not in df.columns:
        raise ValueError("输入表没有 patient_id 列。")
    if patient_id is None:
        raise ValueError("请明确传入 patient_id，不要默认取第一行。")

    sub = df[df["patient_id"].astype(str) == str(patient_id)]
    if sub.empty:
        raise ValueError(f"找不到 patient_id={patient_id}")

    # 如果有日期列，优先取最新一条
    if "assessment_date" in sub.columns:
        sub = sub.copy()
        sub["assessment_date"] = pd.to_datetime(sub["assessment_date"], errors="coerce")
        sub = sub.sort_values("assessment_date")

    return sub.iloc[-1]


def explain_patient_row(row: pd.Series) -> dict:
    """
    对齐 Fusion 模块逻辑：
    直接读取 Fusion 给出的 risk_score 和 risk_level，不重新计算总分！
    仅计算各特征对于“高风险”的相对贡献度（SHAP 模拟）。
    """
    # 1. 严格信任并读取 Fusion 模块的结论
    risk_score = float(row.get("risk_score", 0.0))
    risk_level = str(row.get("risk_level", "Unknown"))

    # 2. 读取原始特征
    icf_total = float(row.get("icf_total", 0))
    gait_prob = float(row.get("gait_anomaly_prob", 0))
    imu_quality = float(row.get("imu_quality_mean", 1.0))

    # 3. 对齐 Fusion 模块的单边风险映射逻辑
    icf_risk = icf_total / 200.0 if icf_total > 10 else np.clip(icf_total, 0, 1)
    gait_risk = np.clip(gait_prob, 0, 1)
    sensor_risk = 1.0 - np.clip(imu_quality, 0, 1)

    # 4. 计算相对致险贡献度百分比
    total_risk_pool = icf_risk + gait_risk + sensor_risk + 1e-9  # 加小常数防除零

    top_contributors = sorted(
        [
            {"feature": "ICF临床量表", "value": icf_total, "contribution": float(icf_risk / total_risk_pool)},
            {"feature": "Gait步态异常", "value": gait_prob, "contribution": float(gait_risk / total_risk_pool)},
            {"feature": "IMU动作不规范", "value": imu_quality, "contribution": float(sensor_risk / total_risk_pool)},
        ],
        key=lambda x: x["contribution"],
        reverse=True,
    )

    explain_one_line = (
        f"经多模态融合大脑评估，综合风险为 {risk_score:.2f} (等级: {risk_level})。"
        f"其中导致风险升高的最大贡献项是【{top_contributors[0]['feature']}】，致险占比达 {top_contributors[0]['contribution'] * 100:.1f}%。"
    )

    return {
        "patient_id": str(row.get("patient_id", "unknown")),
        "risk_score": risk_score,
        "risk_level": risk_level,
        "top_contributors": top_contributors,
        "explain_one_line": explain_one_line,
    }


def save_outputs(report: dict, out_dir: str) -> None:
    """
    保存 JSON 和解释贡献度图表。
    """
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, f"{report['patient_id']}_xai_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 画相对贡献度图
    feats = [x["feature"] for x in report["top_contributors"]]
    vals = [x["contribution"] * 100 for x in report["top_contributors"]]  # 转为百分比

    plt.figure(figsize=(7, 4))
    bars = plt.barh(feats[::-1], vals[::-1], color=['#4C72B0', '#DD8452', '#C44E52'])  # 倒序让最高贡献在上面
    plt.xlabel("致险贡献度占比 (%)", fontsize=12)
    plt.title(f"多模态风险归因分析 (Patient: {report['patient_id']})", fontsize=14)

    # 标上数值
    for bar in bars:
        plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                 f"{bar.get_width():.1f}%", va='center', fontsize=11)

    plt.xlim(0, 100)
    plt.tight_layout()

    png_path = os.path.join(out_dir, f"{report['patient_id']}_xai_contributions.png")
    plt.savefig(png_path, dpi=200)
    plt.close()

    print(f"[XAI] 已保存解释报告: {json_path}")
    print(f"[XAI] 已保存归因图表: {png_path}")


def run_xai(input_path: str, out_dir: str, patient_id: str = None) -> dict:
    df = load_inputs(input_path)
    row = pick_patient_row(df, patient_id=patient_id)
    report = explain_patient_row(row)
    save_outputs(report, out_dir)
    return report