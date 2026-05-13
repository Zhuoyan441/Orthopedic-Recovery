import os
import json
import pandas as pd
import streamlit as st
import re

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

# ---------- 页面配置 ----------
st.set_page_config(page_title="骨科康复 · 多模态监测", page_icon="📐", layout="wide")
st.title("📐 骨科康复 · 多模态监测")
st.caption("综合动作、步态、功能量表数据，辅助康复评估")

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


# =========================
# 工具函数（不变）
# =========================
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
    if imu_df.empty or icf_df.empty:
        return pd.DataFrame()
    imu = imu_df.copy()
    gait = gait_df.copy()
    icf = icf_df.copy()
    if "assessment_date" in icf.columns:
        icf["assessment_date"] = pd.to_datetime(icf["assessment_date"], errors="coerce")
        icf_last = icf.sort_values("assessment_date").groupby("patient_id", as_index=False).tail(1)
    else:
        icf_last = icf.groupby("patient_id", as_index=False).last()
    if "icf_total" not in icf_last.columns:
        icf_last["icf_total"] = 100.0
    icf_last = icf_last[["patient_id", "icf_total"]]
    if "quality_score" in imu.columns:
        imu_agg = imu.groupby("patient_id", as_index=False).agg(imu_quality_mean=("quality_score", "mean"))
    else:
        imu_agg = pd.DataFrame(columns=["patient_id", "imu_quality_mean"])
    if (not gait.empty) and ("anomaly_prob" in gait.columns):
        gait_agg = gait.groupby("patient_id", as_index=False).agg(gait_anomaly_prob=("anomaly_prob", "mean"))
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

# ---------- 侧边栏 ----------
st.sidebar.header("⚙️ 操作台")
patient_id = st.sidebar.selectbox("就诊者", all_patient_ids)

st.sidebar.markdown("---")
st.sidebar.markdown("### 数据处理")
build_fusion_btn = st.sidebar.button("📂 更新数据源", use_container_width=True)
run_fusion_btn = st.sidebar.button("📊 综合分析", type="primary", use_container_width=True)
run_xai_btn = st.sidebar.button("🔍 查看评估解读", use_container_width=True)

# ---------- 数据概览 ----------
with st.expander("📁 数据概览", expanded=False):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("动作样本数", f"{len(imu_df)} 条")
    col2.metric("步态样本数", f"{len(gait_df)} 条")
    col3.metric("量表记录数", f"{len(icf_df)} 条")
    col4.metric("融合结果数", f"{len(fusion_output_df)} 条")

st.markdown("---")

# ---------- 生成 fusion input 逻辑 ----------
if build_fusion_btn:
    with st.spinner("更新数据源中..."):
        fusion_input_df = build_fusion_input(imu_df, gait_df, icf_df)
        st.success(f"数据源已更新：{FUSION_INPUT_PATH}")
        try:
            config = FusionConfig(
                icf_path=ICF_PATH,
                gait_path=GAIT_PATH,
                sensor_path=IMU_PATH,
                output_dir=FUSION_OUT_DIR,
            )
            fusion_output_df = run_fusion_pipeline(config)
            st.success("融合数据已同步")
        except Exception as e:
            st.warning(f"同步失败：{e}")

# ---------- 个体档案 ----------
st.subheader(f"就诊者 {patient_id} 康复档案")
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["活动记录", "步态特征", "功能量表", "综合评估", "结果解读"]
)

# ----- 活动记录 -----
with tab1:
    if imu_df.empty:
        st.warning("未找到动作数据文件")
    else:
        patient_imu = imu_df[imu_df["patient_id"].astype(str) == str(patient_id)]
        col_img, col_data = st.columns([1, 2])
        with col_img:
            if "quality_score" in patient_imu.columns:
                score = patient_imu['quality_score'].mean()
                st.metric("动作质量均值", f"{score:.2f} / 1.0",
                          delta="较好" if score > 0.7 else "需关注",
                          delta_color="normal" if score > 0.7 else "inverse")
            if "action_type" in patient_imu.columns:
                st.write("活动类型分布")
                st.bar_chart(patient_imu["action_type"].value_counts())
        with col_data:
            st.write("最近活动记录")
            st.dataframe(patient_imu.head(20), use_container_width=True, hide_index=True)

# ----- 步态特征 -----
with tab2:
    if gait_df.empty:
        st.warning("未找到步态数据文件")
    else:
        patient_gait = gait_df[gait_df["patient_id"] == patient_id]
        col_risk, col_chart = st.columns([1, 2])
        with col_risk:
            if not patient_gait.empty and "anomaly_prob" in patient_gait.columns:
                risk_score = gait_compute_risk_score(patient_id)
                st.metric("步态异常指数", f"{risk_score:.2f}",
                          delta="较高" if risk_score > 0.6 else "正常",
                          delta_color="inverse" if risk_score > 0.6 else "normal")
            if not patient_gait.empty and "anomaly_prob" in patient_gait.columns:
                st.write("异常概率分布")
                st.bar_chart(patient_gait["anomaly_prob"])
        with col_chart:
            if not patient_gait.empty and "time" in patient_gait.columns and "knee_angle" in patient_gait.columns:
                st.write("膝关节角度变化")
                plot_df = patient_gait[["time", "knee_angle"]].copy().sort_values("time")
                st.line_chart(plot_df.set_index("time"))
        with st.expander("详细步态数据"):
            st.dataframe(patient_gait.head(20), use_container_width=True, hide_index=True)

# ----- 功能量表 -----
with tab3:
    if icf_df.empty:
        st.warning("未找到ICF量表数据")
    else:
        patient_icf = icf_df[icf_df["patient_id"].astype(str) == str(patient_id)]
        if "assessment_date" in patient_icf.columns and "icf_total" in patient_icf.columns:
            patient_icf = patient_icf.copy()
            patient_icf["assessment_date"] = pd.to_datetime(patient_icf["assessment_date"], errors="coerce")
            patient_icf = patient_icf.sort_values("assessment_date")
            latest_icf = patient_icf.iloc[-1]["icf_total"] if not patient_icf.empty else 0
            st.metric("最新功能评分", f"{latest_icf:.1f}", help="基于ICF的综合功能评分")
            st.write("功能评分变化趋势")
            st.line_chart(patient_icf.set_index("assessment_date")[["icf_total"]])
        with st.expander("历史评估记录"):
            st.dataframe(patient_icf, use_container_width=True, hide_index=True)

# ----- 综合评估 -----
with tab4:
    if fusion_output_df.empty:
        st.info("尚未生成综合评估结果，请先在左侧点击「综合分析」")
    else:
        patient_fusion = fusion_output_df[fusion_output_df["patient_id"].astype(str) == str(patient_id)]
        if run_fusion_btn:
            with st.spinner("正在分析数据，生成评估报告..."):
                try:
                    config = FusionConfig(
                        icf_path=ICF_PATH,
                        gait_path=GAIT_PATH,
                        sensor_path=IMU_PATH,
                        output_dir=FUSION_OUT_DIR,
                    )
                    final_df = run_fusion_pipeline(config)
                    st.success("评估完成")
                    patient_row = final_df[final_df["patient_id"].astype(str) == str(patient_id)]
                    if not patient_row.empty:
                        generate_patient_report(patient_id, patient_row.iloc[0], FUSION_OUT_DIR)
                        st.markdown("### 评估报告")
                        report_path = os.path.join(FUSION_OUT_DIR, f"{patient_id}_report.json")
                        if os.path.exists(report_path):
                            with open(report_path, "r", encoding="utf-8") as f:
                                report_data = json.load(f)
                            st.json(report_data)
                            with open(report_path, "r", encoding="utf-8") as f:
                                st.download_button(
                                    "下载报告(JSON)",
                                    data=f.read(),
                                    file_name=f"{patient_id}_report.json",
                                    mime="application/json"
                                )
                        else:
                            st.warning("未找到报告文件")
                    else:
                        st.warning("未能定位该就诊者数据")
                except Exception as e:
                    st.error(f"评估失败：{e}")
        else:
            st.dataframe(patient_fusion.head(20), use_container_width=True, hide_index=True)

# ----- 结果解读 -----
with tab5:
    st.markdown("### 评估结果解读")
    st.caption("基于融合分析的结果归因，了解哪些指标对评估影响较大")
    if run_xai_btn:
        with st.spinner("生成解读中..."):
            try:
                if not os.path.exists(XAI_INPUT_PATH):
                    st.warning("缺少融合数据，请先生成综合评估")
                else:
                    report = run_xai(
                        input_path=XAI_INPUT_PATH,
                        out_dir=XAI_OUT_DIR,
                        patient_id=patient_id
                    )
                    st.success("解读已生成")
                    st.markdown("#### 关键指标贡献度")
                    col_img, col_json = st.columns([1.5, 1])
                    with col_img:
                        png_path = os.path.join(XAI_OUT_DIR, f"{patient_id}_xai_contributions.png")
                        if os.path.exists(png_path):
                            st.image(png_path, caption=f"就诊者 {patient_id} 评估贡献因素", use_container_width=True)
                        else:
                            st.info("未生成可视化图表")
                    with col_json:
                        st.markdown("##### 详细数据 (JSON)")
                        parsed_xai_data = None
                        if isinstance(report, dict):
                            parsed_xai_data = report
                        elif isinstance(report, str):
                            try:
                                cleaned_report = report.strip().strip("'").strip('"').replace("'", '"')
                                parsed_xai_data = json.loads(cleaned_report)
                            except:
                                st.warning("无法解析为JSON，显示原始内容")
                                st.write(report)
                        if parsed_xai_data:
                            st.json(parsed_xai_data)
                        json_path = os.path.join(XAI_OUT_DIR, f"{patient_id}_xai_report.json")
                        if os.path.exists(json_path):
                            with open(json_path, "r", encoding="utf-8") as f:
                                st.download_button(
                                    "下载解读数据(JSON)",
                                    data=f.read(),
                                    file_name=f"{patient_id}_xai_report.json",
                                    mime="application/json",
                                    use_container_width=True
                                )
            except Exception as e:
                st.error(f"解读生成失败：{e}")
    else:
        st.info("点击左侧「查看评估解读」按钮，获取本就诊者的评估归因分析")

st.markdown("---")
st.caption("提示：更新数据源后，请先点击「更新数据源」再执行「综合分析」。")