import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from api import IMUPredictor


def main():
    # 路径配置（遵循小组统一结构）
    DATA_BASE_DIR = "../../data/sensor/"
    X_TEST_PATH = os.path.join(DATA_BASE_DIR, "raw/UCI HAR Dataset/test/X_test.txt")
    Y_TEST_PATH = os.path.join(DATA_BASE_DIR, "raw/UCI HAR Dataset/test/y_test.txt")
    OUTPUT_DIR = os.path.join(DATA_BASE_DIR, "demo_output/")
    MAPPING_PATH = os.path.join(DATA_BASE_DIR, "mapping_sample2patient.csv")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 加载模型
    predictor = IMUPredictor(weight_path='imu_model.pth')

    # 2. 读取并处理 UCI-HAR 数据
    print("读取数据并进行标准化...")
    try:
        # 使用 sep='\s+' 解决 FutureWarning
        X_raw = pd.read_csv(X_TEST_PATH, sep='\s+', header=None).values
        y_true = pd.read_csv(Y_TEST_PATH, sep='\s+', header=None).values.flatten()

        # 标签对齐：1-6 转 0-5
        if y_true.min() == 1:
            y_true = y_true - 1

        # 标准化（解决效果差的关键，确保推理数据分布与训练一致）
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)
    except Exception as e:
        print(f"❌ 加载数据失败: {e}")
        return

    # 3. 推理
    res = predictor.predict(X_scaled)
    acc = accuracy_score(y_true, res['preds'])
    print(f"✅ 任务完成！测试集准确率: {acc * 100:.2f}%")

    # 4. 生成 imu_action_scores.csv (严格符合学姐要求的字段)
    num_patients = 50
    num_samples = len(res['preds'])
    samples_per_patient = num_samples // num_patients
    patient_ids = []
    for i in range(num_samples):
        # 计算当前样本属于第几个病人 (0-49)
        p_idx = i // samples_per_patient
        # 确保索引不超过 49 (防止最后几个样本溢出)
        if p_idx >= num_patients:
            p_idx = num_patients - 1
        patient_ids.append(f'P{p_idx + 1:03d}')
    df_scores = pd.DataFrame({
        'sample_id': range(num_samples),
        'patient_id': patient_ids,
        'action': res['action_names'],
        'prob': [np.round(res['probs'][i][res['preds'][i]], 4) for i in range(num_samples)],
        'quality_score': np.round(res['confidences'] * 100, 2)
    })
    df_scores.to_csv(os.path.join(OUTPUT_DIR, "imu_action_scores.csv"), index=False)

    # 5. 生成混淆矩阵 confusion_matrix.png
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文乱码
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, res['preds'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=predictor.actions,
                yticklabels=predictor.actions)
    plt.title(f"IMU 动作识别混淆矩阵 (Acc: {acc * 100:.1f}%)")
    plt.xlabel("预测动作")
    plt.ylabel("真实动作")
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))

    # 6. 生成 misclassified_examples.csv (FP/FN 样本)
    mask = res['preds'] != y_true
    errors = df_scores[mask].copy()
    errors['true_label'] = [predictor.actions[i] for i in y_true[mask]]
    # 提取前 10 条（包含可能的 FP/FN）供演示
    errors.head(10).to_csv(os.path.join(OUTPUT_DIR, "misclassified_examples.csv"), index=False)

    # 7. 生成映射文件 mapping_sample2patient.csv
    df_scores[['sample_id', 'patient_id']].to_csv(MAPPING_PATH, index=False)

    print(f"🚀 交付文件已成功生成至: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()