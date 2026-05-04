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

st.set_page_config(page_title="骨科康复智能系统大屏", page_icon="🏥", layout="wide")
st.title("🏥 基于 ICF 的骨科康复阈值提醒系统")
st.caption("多模态数据监控大屏：IMU 传感器 | 步态特征 | ICF 量表 | 融合大脑 | XAI 可解释性")

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
# 工具函数
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

# 侧边栏：控制面板
st.sidebar.header("⚙️ 控制面板")
patient_id = st.sidebar.selectbox("🧑‍⚕️ 选择患者 ID", all_patient_ids)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🚀 系统管线操作")
build_fusion_btn = st.sidebar.button("🔄 生成/刷新 Fusion 输入", use_container_width=True)
run_fusion_btn = st.sidebar.button("🧠 运行 Fusion (大脑融合)", type="primary", use_container_width=True)
run_xai_btn = st.sidebar.button("🔍 运行 XAI (生成解释)", use_container_width=True)

# =========================
# 系统总览卡片 (全局状态)
# =========================
with st.expander("📊 数据库全局状态大盘", expanded=False):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("IMU 动作样本数", f"{len(imu_df)} 条")
    col2.metric("Gait 步态样本数", f"{len(gait_df)} 条")
    col3.metric("ICF 临床记录数", f"{len(icf_df)} 条")
    col4.metric("Fusion 融合数", f"{len(fusion_output_df)} 条")

st.markdown("---")

# =========================
# 点击生成 fusion input 逻辑
# =========================
if build_fusion_btn:
    with st.spinner("正在聚合多模态数据..."):
        fusion_input_df = build_fusion_input(imu_df, gait_df, icf_df)
        st.success(f"✅ Fusion 输入已生成：{FUSION_INPUT_PATH}")

        try:
            config = FusionConfig(
                icf_path=ICF_PATH,
                gait_path=GAIT_PATH,
                sensor_path=IMU_PATH,
                output_dir=FUSION_OUT_DIR,
            )
            fusion_output_df = run_fusion_pipeline(config)
            st.success("✅ Fusion pipeline 已自动刷新")
        except Exception as e:
            st.warning(f"⚠️ Fusion pipeline 运行失败：{e}")

# =========================
# 个体患者大屏 (Tabs)
# =========================
st.subheader(f"患者 {patient_id} 综合康复档案")
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["⌚ IMU 动作监测", "🚶 Gait 步态分析", "📋 ICF 临床量表", "🧠 Fusion 融合诊断", "💡 XAI 智能可解释"])

# ---------- IMU ----------
with tab1:
    if imu_df.empty:
        st.warning("没有找到 IMU CSV：data/sensor/demo_output/imu_action_scores.csv")
    else:
        patient_imu = imu_df[imu_df["patient_id"].astype(str) == str(patient_id)]

        col_img, col_data = st.columns([1, 2])
        with col_img:
            if "quality_score" in patient_imu.columns:
                score = patient_imu['quality_score'].mean()
                st.metric("IMU 平均动作质量分", f"{score:.2f} / 1.0",
                          delta="良好" if score > 0.7 else "需纠正", delta_color="normal" if score > 0.7 else "inverse")
            if "action_type" in patient_imu.columns:
                st.write("📈 **动作类型分布**")
                st.bar_chart(patient_imu["action_type"].value_counts())

        with col_data:
            st.write("📄 **详细日志**")
            st.dataframe(patient_imu.head(20), use_container_width=True, hide_index=True)

# ---------- Gait ----------
with tab2:
    if gait_df.empty:
        st.warning("没有找到 Gait CSV：data/gait/demo_output/gait_features.csv")
    else:
        patient_gait = gait_df[gait_df["patient_id"] == patient_id]

        col_risk, col_chart = st.columns([1, 2])
        with col_risk:
            if not patient_gait.empty and "anomaly_prob" in patient_gait.columns:
                risk_score = gait_compute_risk_score(patient_id)
                st.metric("异常步态风险分", f"{risk_score:.2f}",
                          delta="高危" if risk_score > 0.6 else "正常",
                          delta_color="inverse" if risk_score > 0.6 else "normal")

            if not patient_gait.empty and "anomaly_prob" in patient_gait.columns:
                st.write("📊 **异常概率分布**")
                st.bar_chart(patient_gait["anomaly_prob"])

        with col_chart:
            if not patient_gait.empty and "time" in patient_gait.columns and "knee_angle" in patient_gait.columns:
                st.write("📉 **膝关节角度变化 (Knee Angle vs Time)**")
                plot_df = patient_gait[["time", "knee_angle"]].copy()
                plot_df = plot_df.sort_values("time")
                st.line_chart(plot_df.set_index("time"))

        with st.expander("查看底层步态特征数据"):
            st.dataframe(patient_gait.head(20), use_container_width=True, hide_index=True)

# ---------- ICF ----------
with tab3:
    if icf_df.empty:
        st.warning("没有找到 ICF CSV：data/icf/demo_output/icf_time_series.csv")
    else:
        patient_icf = icf_df[icf_df["patient_id"].astype(str) == str(patient_id)]

        if "assessment_date" in patient_icf.columns and "icf_total" in patient_icf.columns:
            patient_icf = patient_icf.copy()
            patient_icf["assessment_date"] = pd.to_datetime(patient_icf["assessment_date"], errors="coerce")
            patient_icf = patient_icf.sort_values("assessment_date")

            latest_icf = patient_icf.iloc[-1]["icf_total"] if not patient_icf.empty else 0
            st.metric("最新 ICF 综合得分", f"{latest_icf:.1f}", "康复进展跟踪")

            st.write("📈 **ICF 得分时间序列**")
            st.line_chart(patient_icf.set_index("assessment_date")[["icf_total"]])

        with st.expander("查看临床评估详单"):
            st.dataframe(patient_icf, use_container_width=True, hide_index=True)

# ---------- Fusion ----------
with tab4:
    if fusion_output_df.empty:
        st.info("💡 尚未生成融合结果，请点击左侧控制面板的 **[运行 Fusion]** 按钮。")
    else:
        patient_fusion = fusion_output_df[fusion_output_df["patient_id"].astype(str) == str(patient_id)]

        if run_fusion_btn:
            with st.spinner("系统大脑正在进行综合研判..."):
                try:
                    config = FusionConfig(
                        icf_path=ICF_PATH,
                        gait_path=GAIT_PATH,
                        sensor_path=IMU_PATH,
                        output_dir=FUSION_OUT_DIR,
                    )
                    final_df = run_fusion_pipeline(config)
                    st.success("✅ 融合诊断完毕！")

                    patient_row = final_df[final_df["patient_id"].astype(str) == str(patient_id)]
                    if not patient_row.empty:
                        # 运行生成报告
                        _ = generate_patient_report(patient_id, patient_row.iloc[0], FUSION_OUT_DIR)

                        st.markdown("### 📋 智能诊断报告")
                        report_path = os.path.join(FUSION_OUT_DIR, f"{patient_id}_report.json")

                        if os.path.exists(report_path):
                            with open(report_path, "r", encoding="utf-8") as f:
                                report_data = json.load(f)
                            st.json(report_data)

                            with open(report_path, "r", encoding="utf-8") as f:
                                st.download_button(
                                    "📥 下载完整 JSON 报告",
                                    data=f.read(),
                                    file_name=f"{patient_id}_report.json",
                                    mime="application/json"
                                )
                        else:
                            st.warning("未找到生成的 JSON 报告文件。")
                    else:
                        st.warning("融合大表中未能定位该患者，请检查上游数据一致性。")
                except Exception as e:
                    st.error(f"❌ Fusion 运行失败：{e}")
        else:
            st.dataframe(patient_fusion.head(20), use_container_width=True, hide_index=True)

# ---------- XAI ----------
with tab5:
    st.markdown("### 💡 AI 决策可解释性分析")
    st.caption("基于 SHAP/LIME 算法解释模型基于哪几个维度的特征触发了预警，打破医疗 AI '黑盒'。")

    if run_xai_btn:
        with st.spinner("正在生成可解释性归因图表..."):
            try:
                if not os.path.exists(XAI_INPUT_PATH):
                    st.warning("⚠️ 找不到融合层输出的数据，请先执行 Fusion！")
                else:
                    report = run_xai(
                        input_path=XAI_INPUT_PATH,
                        out_dir=XAI_OUT_DIR,
                        patient_id=patient_id
                    )
                    st.success("✅ 解释已生成！")

                    st.markdown("#### 🔍 特征归因概览")
                    # 调整左右比例，把图表空间放大 (1.5 : 1)
                    col_img, col_json = st.columns([1.5, 1])

                    with col_img:
                        st.markdown("##### 📊 关键特征贡献度可视化")
                        png_path = os.path.join(XAI_OUT_DIR, f"{patient_id}_xai_contributions.png")
                        if os.path.exists(png_path):
                            st.image(png_path, caption=f"【{patient_id}】高危预警贡献特征分布", use_container_width=True)
                        else:
                            st.info("暂无解释可视化图像生成。")

                    with col_json:
                        st.markdown("##### 📝 详细解释数据 (JSON)")

                        # 【核心修复】：无论上游返回什么奇葩字符串，强制将其转换为 Python 字典后再渲染
                        parsed_xai_data = None
                        if isinstance(report, dict):
                            parsed_xai_data = report
                        elif isinstance(report, str):
                            try:
                                # 尝试修复某些可能被多次 json.dumps 嵌套的字符串
                                cleaned_report = report.strip().strip("'").strip('"')
                                # 有时候字符串里有单引号需要换成双引号才能被标准 json 解析
                                cleaned_report = cleaned_report.replace("'", '"')
                                parsed_xai_data = json.loads(cleaned_report)
                            except Exception as parse_str_e:
                                # 如果实在解析不了字符串，兜底处理
                                st.warning(f"未能将返回的文本解析为标准JSON结构，将原样显示文本。")
                                st.write(report)

                        if parsed_xai_data is not None:
                            st.json(parsed_xai_data)

                        # 如果有本地文件，提供下载功能
                        json_path = os.path.join(XAI_OUT_DIR, f"{patient_id}_xai_report.json")
                        if os.path.exists(json_path):
                            with open(json_path, "r", encoding="utf-8") as f:
                                st.download_button(
                                    "📥 下载特征贡献度 JSON",
                                    data=f.read(),
                                    file_name=f"{patient_id}_xai_report.json",
                                    mime="application/json",
                                    use_container_width=True
                                )
            except Exception as e:
                st.error(f"❌ XAI 运行失败：{e}")
    else:
        st.info("👉 点击左侧控制面板的 **[运行 XAI]** 按钮生成特征贡献度解释。")

st.markdown("---")
st.caption("💡 提示：若后台模型代码更新，请按顺序点击侧边栏的刷新与运行按钮以加载最新权重。")