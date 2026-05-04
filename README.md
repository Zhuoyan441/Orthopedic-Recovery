# Orthopedic-Recovery

基于 ICF 框架与深度学习的骨科康复阈值提醒系统研究

本项目通过多模态数据（ICF 疗效评估、步态特征、IMU 动作识别）融合，输出患者康复风险评分，并提供可解释性分析。

## 快速开始（Quick Start）

### 1. 环境准备
- Python 3.8+ （推荐使用虚拟环境）
- 安装依赖：
  ```bash
  pip install -r requirements.txt
  ```

### 2. 克隆仓库
```bash
git clone https://github.com/Zhuoyan441/Orthopedic-Recovery.git
cd Orthopedic-Recovery
```

### 3. 运行 Streamlit 演示
```bash
streamlit run app_streamlit.py
```

说明：app_streamlit.py 由项目协调负责人提供，请确保已拉取最新代码。

---

## 项目当前目录结构（主要）

```bash
Orthopedic-Recovery/
├── run_fusion_demo.py
├── main.py
├── requirements.txt
├── data/
│   ├── icf/
│   ├── gait/
│   ├── sensor/
│   └── fusion/
│       └── demo_output/
├── docs/
│   ├── 数据模拟对照表.md
│   └── ...
└── fusion/
```

各模块源代码将陆续放入 code/ 目录。

---

## 各模块输出文件标准（待实现）

- ICF：data/icf/demo_output/icf_time_series.csv （字段：patient_id, assessment_date, true_icf, pred_icf）
- 步态：data/gait/demo_output/gait_features.csv （字段：patient_id, time, knee_angle, ankle_angle, anomaly_prob）
- IMU：data/sensor/demo_output/imu_action_scores.csv （字段：patient_id, action, prob, quality_score）
- 融合：data/fusion/demo_output/example_fusion_output.csv （字段：risk_score 等）

详细字段定义见 docs/data_interface_standard.md。

---

## 注意事项

- 各模块请按上述路径输出文件，字段与标准一致。
- 运行 streamlit run app_streamlit.py 前请确保已安装依赖。

---

## 贡献者

- 韩卓妍：项目协调 + XAI + 最终 demo 整合
- 周子昊：IMU 动作识别模块
- 王浩然：步态特征提取模块
- 周亦沁：ICF 仿真与预测模块
- 胡昕璟：多模态融合模块
- 庄贻媛：文档与数据整理
