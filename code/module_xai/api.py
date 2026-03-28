# code/xai/api.py
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_inputs(input_path: str) -> pd.DataFrame:
    """
    读取融合后的输入表。
    期望至少包含：
    patient_id, icf_total, gait_anomaly_prob, imu_quality_mean
    """
    return pd.read_csv(input_path)


def pick_patient_row(df: pd.DataFrame, patient_id: str = None) -> pd.Series:
    """
    选出一个 patient 的一行数据。
    如果没传 patient_id，就默认取第一行。
    """
    if df.empty:
        raise ValueError("输入表为空，无法做XAI。")

    if patient_id is None:
        return df.iloc[0]

    if "patient_id" not in df.columns:
        raise ValueError("输入表没有 patient_id 列。")

    sub = df[df["patient_id"] == patient_id]
    if sub.empty:
        raise ValueError(f"找不到 patient_id={patient_id}")

    return sub.iloc[0]


def explain_patient_row(row: pd.Series) -> dict:
    """
    根据单个 patient 的特征，计算一个简单的解释结果。
    这是一个可演示版本，不依赖训练好的复杂模型。
    """
    # 取特征，缺失则给默认值
    icf_total = float(row.get("icf_total", 0))
    gait_prob = float(row.get("gait_anomaly_prob", 0))
    imu_quality = float(row.get("imu_quality_mean", 0))

    # 如果 icf_total 像 0-100 的分数，简单压到 0-1
    if icf_total > 1.5:
        icf_norm = icf_total / 100.0
    else:
        icf_norm = icf_total

    # 风险贡献：ICF 越高（或越差）越危险、步态异常越危险、动作质量越低越危险
    contrib_icf = 0.5 * icf_norm
    contrib_gait = 0.3 * gait_prob
    contrib_imu = 0.2 * (1.0 - imu_quality)

    risk_score = contrib_icf + contrib_gait + contrib_imu
    risk_score = float(np.clip(risk_score, 0, 1))

    if risk_score < 0.33:
        risk_level = "Low"
    elif risk_score < 0.66:
        risk_level = "Medium"
    else:
        risk_level = "High"

    top_contributors = sorted(
        [
            {"feature": "icf_total", "value": icf_total, "contribution": float(contrib_icf)},
            {"feature": "gait_anomaly_prob", "value": gait_prob, "contribution": float(contrib_gait)},
            {"feature": "imu_quality_mean", "value": imu_quality, "contribution": float(contrib_imu)},
        ],
        key=lambda x: x["contribution"],
        reverse=True,
    )

    explain_one_line = (
        f"主要由 {top_contributors[0]['feature']} 贡献最高，"
        f"综合风险为 {risk_score:.2f}，等级为 {risk_level}。"
    )

    return {
        "patient_id": row.get("patient_id", "unknown"),
        "risk_score": risk_score,
        "risk_level": risk_level,
        "top_contributors": top_contributors,
        "explain_one_line": explain_one_line,
    }


def save_outputs(report: dict, out_dir: str) -> None:
    """
    保存 JSON 和解释图。
    """
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, f"{report['patient_id']}_xai_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 画贡献图
    feats = [x["feature"] for x in report["top_contributors"]]
    vals = [x["contribution"] for x in report["top_contributors"]]

    plt.figure(figsize=(6, 4))
    plt.bar(feats, vals)
    plt.ylabel("Contribution")
    plt.title(f"XAI Explanation - {report['patient_id']}")
    plt.tight_layout()

    png_path = os.path.join(out_dir, f"{report['patient_id']}_xai_contributions.png")
    plt.savefig(png_path, dpi=200)
    plt.close()

    print(f"Saved JSON: {json_path}")
    print(f"Saved PNG : {png_path}")


def run_xai(input_path: str, out_dir: str, patient_id: str = None) -> dict:
    """
    主流程：读取 -> 选择病人 -> 解释 -> 保存
    """
    df = load_inputs(input_path)
    row = pick_patient_row(df, patient_id=patient_id)
    report = explain_patient_row(row)
    save_outputs(report, out_dir)
    return report