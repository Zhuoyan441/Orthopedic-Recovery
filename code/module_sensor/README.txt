# IMU 动作识别模块 (har_action_recognition)

**负责人：周子昊 **

## 1. 模块简介
本模块是 SITP 医疗健康监测项目的子模块，主要负责通过智能手机 IMU 传感器数据（加速度计、陀螺仪）识别患者的实时动作。
- **核心模型**：1D-ResNeXt 深度残差网络
- **特征维度**：561 维时效频域特征
- **识别类别**：行走、上楼、下楼、坐下、站立、躺下 (共 6 类)

## 2. 目录结构
```text
/ (项目根目录)
├── code/
│   └── module_sensor/
│       ├── api.py           # 模型接口与推理逻辑
│       ├── main.py          # 主执行脚本（生成交付文件）
│       ├── imu_model.pth    # 预训练权重文件
│       └── README.md        # 本说明文档
└── data/
    └── sensor/
        ├── raw/             # 存放 X_test.txt 和 y_test.txt
        └── demo_output/     # 存放输出的 CSV 与混淆矩阵图
3. 验收标准达成说明
根据小组最新要求，本模块已完成以下配置：

字段对齐：imu_action_scores.csv 包含 sample_id, patient_id, action, prob, quality_score。

性能评估：输出 confusion_matrix.png 和包含 10 条样本的 misclassified_examples.csv。

数据预处理：代码内置 StandardScaler 标准化逻辑，确保推理准确率。

4. 关于 Patient 映射的 Fallback 说明 (重要)

本模块目前状态：

当前方案：采用 临时均匀映射 (Fallback)。

映射规则：采用块状分配，将测试集样本按序平均分配给 50 名病人（约每人连续 59 条记录）。”

注意：该映射仅用于满足 Streamlit 端的展示需求，不代表真实的医学临床采集对应关系。

5. 快速开始
确保原始数据集 X_test.txt 和 y_test.txt 已放置在 data/sensor/raw/ 目录下。

在 code/module_sensor/ 路径下运行：

Bash
python main.py
运行完成后，可在 data/sensor/demo_output/ 查看生成的演示文件。