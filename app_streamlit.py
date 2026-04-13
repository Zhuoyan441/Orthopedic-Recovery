import os
import json
import pandas as pd
import streamlit as st

from code.module_gait.api import (
    load_data as gait_load_data,
    get_all_patients as gait_get_all_patients,
    get_patient_data as gait_get_patient_data,
    get_one_sample as gait_get_one_sample,
    compute_risk_score as gait_compute_risk_score,
)

from code.module_fusion.api import (
    FusionConfig,
    run_fusion_pipeline,
    generate_patient_report,
)

from code.module_xai.api import run_xai


st.set_page_config(page_title="Orthopedic Recovery Demo", layout="wide")
st.title("骨科康复智能系统 Demo")
st.caption("Streamlit 展示版：IMU + Gait + ICF + Fusion + XAI")


# =========================
# 路径配置
# =========================
IMU_PATH = "data/sensor/demo_output/imu_action_scores.csv"
GAIT_PATH = "data/gait/demo_output/gait_features.csv"
ICF_PATH = "data/icf/demo_output/icf_time_series.csv"

FUSION_OUT_DIR = "data/fusion/demo_output"
FUSION_INPUT_PATH = os.path.join(FUSION_OUT_DIR, "fusion_input.csv")
FUSION_OUTPUT_PATH = os.path.join(FUSION_OUT_DIR, "fusion_output.csv")

XAI_OUT_DIR = "data/xai/demo_output"
XAI_INPUT_PATH = FUSION_OUTPUT_PATH

import re

def normalize_patient_id(pid):
    m = re.search(r"\d+", str(pid))
    if not m:
        return str(pid)
    return f"P{int(m.group()):03d}"

def normalize_df_patient_id(df):
    if not df.empty and "patient_id" in df.columns:
        df = df.copy()
        df["patient_id"] = df["patient_id"].apply(normalize_patient_id)
    return df
# =========================
# 工具函数
# =========================
@st.cache_data
def safe_read_csv(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


def get_all_patient_ids(imu_df, gait_df, icf_df, fusion_df):
    ids = set()
    for df in [imu_df, gait_df, icf_df, fusion_df]:
        if not df.empty and "patient_id" in df.columns:
            ids.update(df["patient_id"].astype(str).unique().tolist())
    return sorted(list(ids))


def build_fusion_input(imu_df, gait_df, icf_df):
    """
    把 IMU + Gait + ICF 合成融合输入表。
    这个表会给 fusion 和 xai 用。
    """
    if imu_df.empty or icf_df.empty:
        return pd.DataFrame()

    imu = imu_df.copy()
    gait = gait_df.copy()
    icf = icf_df.copy()

    # ICF：取最后一次评估
    if "assessment_date" in icf.columns:
        icf["assessment_date"] = pd.to_datetime(icf["assessment_date"], errors="coerce")
        icf_last = (
            icf.sort_values("assessment_date")
               .groupby("patient_id", as_index=False)
               .tail(1)
        )
    else:
        icf_last = icf.groupby("patient_id", as_index=False).last()

    if "icf_total" not in icf_last.columns:
        icf_last["icf_total"] = 100.0

    icf_last = icf_last[["patient_id", "icf_total"]]

    # IMU 聚合
    if "quality_score" in imu.columns:
        imu_agg = imu.groupby("patient_id", as_index=False).agg(
            imu_quality_mean=("quality_score", "mean")
        )
    else:
        imu_agg = pd.DataFrame(columns=["patient_id", "imu_quality_mean"])

    # Gait 聚合
    if (not gait.empty) and ("anomaly_prob" in gait.columns):
        gait_agg = gait.groupby("patient_id", as_index=False).agg(
            gait_anomaly_prob=("anomaly_prob", "mean")
        )
    else:
        gait_agg = pd.DataFrame(columns=["patient_id", "gait_anomaly_prob"])

    merged = icf_last.merge(imu_agg, on="patient_id", how="left")
    merged = merged.merge(gait_agg, on="patient_id", how="left")

    merged["imu_quality_mean"] = merged["imu_quality_mean"].fillna(0.8)
    merged["gait_anomaly_prob"] = merged["gait_anomaly_prob"].fillna(0.2)

    os.makedirs(FUSION_OUT_DIR, exist_ok=True)
    merged.to_csv(FUSION_INPUT_PATH, index=False)
    return merged


# =========================
# 读取数据
# =========================
imu_df = normalize_df_patient_id(safe_read_csv(IMU_PATH))
gait_df = normalize_df_patient_id(safe_read_csv(GAIT_PATH))
icf_df = normalize_df_patient_id(safe_read_csv(ICF_PATH))
fusion_output_df = normalize_df_patient_id(safe_read_csv(FUSION_OUTPUT_PATH))

all_patient_ids = get_all_patient_ids(imu_df, gait_df, icf_df, fusion_output_df)
if not all_patient_ids:
    all_patient_ids = ["P0001"]

patient_id = st.sidebar.selectbox("选择 patient_id", all_patient_ids)

st.sidebar.markdown("---")
st.sidebar.markdown("### 操作")
build_fusion_btn = st.sidebar.button("生成/刷新 Fusion 输入")
run_fusion_btn = st.sidebar.button("运行 Fusion")
run_xai_btn = st.sidebar.button("运行 XAI")


# =========================
# 总览卡片
# =========================
col1, col2, col3, col4 = st.columns(4)
col1.metric("IMU 样本数", len(imu_df))
col2.metric("Gait 样本数", len(gait_df))
col3.metric("ICF 记录数", len(icf_df))
col4.metric("Fusion 输出行数", len(fusion_output_df))

st.markdown("---")


# =========================
# 点击生成 fusion input
# =========================
if build_fusion_btn:
    fusion_input_df = build_fusion_input(imu_df, gait_df, icf_df)
    st.success(f"Fusion 输入已生成：{FUSION_INPUT_PATH}")
    st.dataframe(fusion_input_df.head(20))

    try:
        config = FusionConfig(
            icf_path=ICF_PATH,
            gait_path=GAIT_PATH,
            sensor_path=IMU_PATH,
            output_dir=FUSION_OUT_DIR,
        )
        fusion_output_df = run_fusion_pipeline(config)
        st.success("Fusion pipeline 已刷新")
        st.dataframe(fusion_output_df.head(20))
    except Exception as e:
        st.warning(f"Fusion pipeline 运行失败：{e}")


# =========================
# Tabs
# =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["IMU", "Gait", "ICF", "Fusion", "XAI"])


# ---------- IMU ----------
with tab1:
    st.subheader("IMU 动作识别")
    if imu_df.empty:
        st.warning("没有找到 IMU CSV：data/sensor/demo_output/imu_action_scores.csv")
    else:
        patient_imu = imu_df[imu_df["patient_id"].astype(str) == str(patient_id)]
        st.dataframe(patient_imu.head(20))

        if "action_type" in patient_imu.columns:
            st.write("动作分布")
            st.bar_chart(patient_imu["action_type"].value_counts())

        if "quality_score" in patient_imu.columns:
            st.metric("平均质量分", f"{patient_imu['quality_score'].mean():.4f}")


# ---------- Gait ----------
with tab2:
    st.subheader("步态特征与异常检测")
    if gait_df.empty:
        st.warning("没有找到 Gait CSV：data/gait/demo_output/gait_features.csv")
    else:
        patient_gait = gait_df[gait_df["patient_id"] == patient_id]
        st.dataframe(patient_gait.head(20))

        if not patient_gait.empty and "anomaly_prob" in patient_gait.columns:
            st.metric("步态风险分", f"{gait_compute_risk_score(patient_id):.4f}")

        if not patient_gait.empty and "time" in patient_gait.columns and "knee_angle" in patient_gait.columns:
            st.write("knee_angle 随 time 变化")
            plot_df = patient_gait[["time", "knee_angle"]].copy()
            plot_df = plot_df.sort_values("time")
            st.line_chart(plot_df.set_index("time"))

        if not patient_gait.empty and "anomaly_prob" in patient_gait.columns:
            st.write("anomaly_prob")
            st.bar_chart(patient_gait["anomaly_prob"])


# ---------- ICF ----------
with tab3:
    st.subheader("ICF 时间序列")
    if icf_df.empty:
        st.warning("没有找到 ICF CSV：data/icf/demo_output/icf_time_series.csv")
    else:
        patient_icf = icf_df[icf_df["patient_id"].astype(str) == str(patient_id)]
        st.dataframe(patient_icf.head(20))

        if "assessment_date" in patient_icf.columns and "icf_total" in patient_icf.columns:
            patient_icf = patient_icf.copy()
            patient_icf["assessment_date"] = pd.to_datetime(patient_icf["assessment_date"], errors="coerce")
            patient_icf = patient_icf.sort_values("assessment_date")
            st.line_chart(patient_icf.set_index("assessment_date")[["icf_total"]])


# ---------- Fusion ----------
with tab4:
    st.subheader("融合结果")
    if fusion_output_df.empty:
        st.warning("还没有融合结果，请先点击左侧“生成/刷新 Fusion 输入”")
    else:
        patient_fusion = fusion_output_df[fusion_output_df["patient_id"].astype(str) == str(patient_id)]
        st.dataframe(patient_fusion.head(20))

        if run_fusion_btn:
            try:
                config = FusionConfig(
                    icf_path=ICF_PATH,
                    gait_path=GAIT_PATH,
                    sensor_path=IMU_PATH,
                    output_dir=FUSION_OUT_DIR,
                )
                final_df = run_fusion_pipeline(config)
                st.success("Fusion 已运行完成")

                patient_row = final_df[final_df["patient_id"].astype(str) == str(patient_id)]
                if not patient_row.empty:
                    report = generate_patient_report(patient_id, patient_row.iloc[0], FUSION_OUT_DIR)
                    st.json(report)

                    report_path = os.path.join(FUSION_OUT_DIR, f"{patient_id}_report.json")
                    if os.path.exists(report_path):
                        with open(report_path, "r", encoding="utf-8") as f:
                            report_text = f.read()
                        st.download_button(
                            "下载 patient report JSON",
                            data=report_text,
                            file_name=f"{patient_id}_report.json",
                            mime="application/json"
                        )
                else:
                    st.warning("Fusion 输出里没有找到这个 patient_id")
            except Exception as e:
                st.error(f"Fusion 运行失败：{e}")


# ---------- XAI ----------
with tab5:
    st.subheader("XAI 可解释性")
    st.write("点击按钮后，会基于 fusion_output.csv 生成解释图与 JSON 报告。")

    if run_xai_btn:
        try:
            if not os.path.exists(XAI_INPUT_PATH):
                st.warning("还没有 fusion_output.csv，请先点击左侧“生成/刷新 Fusion 输入”或“运行 Fusion”")
            else:
                report = run_xai(
                    input_path=XAI_INPUT_PATH,
                    out_dir=XAI_OUT_DIR,
                    patient_id=patient_id
                )
                st.success("XAI 运行成功")
                st.json(report)

                png_path = os.path.join(XAI_OUT_DIR, f"{patient_id}_xai_contributions.png")
                if os.path.exists(png_path):
                    st.image(png_path, caption="XAI 解释图", use_container_width=True)

                json_path = os.path.join(XAI_OUT_DIR, f"{patient_id}_xai_report.json")
                if os.path.exists(json_path):
                    with open(json_path, "r", encoding="utf-8") as f:
                        st.download_button(
                            "下载 XAI JSON",
                            data=f.read(),
                            file_name=f"{patient_id}_xai_report.json",
                            mime="application/json"
                        )
        except Exception as e:
            st.error(f"XAI 运行失败：{e}")


st.markdown("---")
st.write("如果模块代码更新了，先点左侧“生成/刷新 Fusion 输入”，再点“运行 Fusion”或“运行 XAI”。")