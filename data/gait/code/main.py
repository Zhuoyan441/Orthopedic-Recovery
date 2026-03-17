# =========================================
# KIMORE 步态异常检测（最终高准确率版🔥）
# =========================================

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# =========================================
# 1. 路径
# =========================================
BASE_PATH = r"D:\kimore\KIMORE_DATASET"

# =========================================
# 2. 读取数据
# =========================================
X_all = []
y_all = []

for ex_folder in os.listdir(BASE_PATH):
    ex_path = os.path.join(BASE_PATH, ex_folder)

    if not os.path.isdir(ex_path):
        continue

    print(f"\n===== 处理 {ex_folder} =====")

    X_file = os.path.join(ex_path, "Train_X.csv")
    Y_file = os.path.join(ex_path, "Train_Y.csv")

    if not os.path.exists(X_file):
        print("❌ 缺少:", X_file)
        continue

    X = pd.read_csv(X_file, header=None).values
    y = pd.read_csv(Y_file, header=None).values.flatten()

    print("读取成功:", X.shape, y.shape)

    X_all.append(X)
    y_all.append(y)

# =========================================
# 3. 帧 → 动作样本（增强特征🔥）
# =========================================
X_seq = []
y_seq = []

for i in range(len(X_all)):
    X = X_all[i]
    y = y_all[i]

    num_samples = len(y)
    frames_per_sample = X.shape[0] // num_samples

    print(f"每个样本帧数: {frames_per_sample}")

    for j in range(num_samples):
        start = j * frames_per_sample
        end = (j + 1) * frames_per_sample

        sample = X[start:end]

        # ===== 高级特征（核心🔥）=====
        mean_feat = np.mean(sample, axis=0)
        std_feat = np.std(sample, axis=0)
        max_feat = np.max(sample, axis=0)
        min_feat = np.min(sample, axis=0)

        # 动态变化（关键）
        diff_feat = np.mean(np.diff(sample, axis=0), axis=0)

        feature = np.concatenate([
            mean_feat,
            std_feat,
            max_feat,
            min_feat,
            diff_feat
        ])  # 500维

        X_seq.append(feature)
        y_seq.append(int(y[j]))

# =========================================
# 4. 转 numpy
# =========================================
X = np.array(X_seq)
y = np.array(y_seq)

print("\n原始标签范围:", np.min(y), np.max(y))

# =========================================
# 5. 自适应阈值（二分类🔥）
# =========================================
threshold = np.percentile(y, 60)

y = np.array([0 if label >= threshold else 1 for label in y])

print("转换后二分类分布:", np.bincount(y))

# =========================================
# 6. 标准化
# =========================================
scaler = StandardScaler()
X = scaler.fit_transform(X)

# =========================================
# 7. 划分数据
# =========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================================
# 8. 转 tensor
# =========================================
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# =========================================
# 9. 类别不平衡处理🔥
# =========================================
class_counts = np.bincount(y_train.numpy())
weights = 1.0 / class_counts
weights = torch.tensor(weights, dtype=torch.float32)

# =========================================
# 10. 模型（强化版🔥）
# =========================================
class GaitModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)

model = GaitModel(X.shape[1])

# =========================================
# 11. 训练
# =========================================
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(40):
    model.train()

    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# =========================================
# 12. 测试
# =========================================
model.eval()
with torch.no_grad():
    preds = model(X_test)
    y_pred = torch.argmax(preds, dim=1)

# =========================================
# 13. 评估
# =========================================
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n========================")
print("最终结果")
print("========================")
print("Accuracy:", acc)
print("Confusion Matrix:\n", cm)
# =========================================
# 14. 生成 gait_features.csv（给Streamlit用🔥）
# =========================================
output_dir = "data/gait/demo_output"
os.makedirs(output_dir, exist_ok=True)

gait_rows = []
patient_id = 0

for i in range(len(X_all)):
    X_raw = X_all[i]
    y_raw = y_all[i]

    num_samples = len(y_raw)
    frames = X_raw.shape[0] // num_samples

    for j in range(num_samples):
        sample = X_raw[j*frames:(j+1)*frames]

        time = np.arange(frames)

        # 简化关节角（用于展示）
        knee_angle = np.mean(sample[:, :10], axis=1)
        ankle_angle = np.mean(sample[:, 10:20], axis=1)

        gait_cycle = np.sin(np.linspace(0, 2*np.pi, frames))

        # anomaly_prob（用于展示，不参与训练）
        anomaly_prob = 1 - (y_raw[j] / max(y_raw))

        for t in range(frames):
            gait_rows.append([
                f"p{patient_id}",
                t,
                knee_angle[t],
                ankle_angle[t],
                gait_cycle[t],
                anomaly_prob
            ])

        patient_id += 1

df = pd.DataFrame(gait_rows, columns=[
    "patient_id",
    "time",
    "knee_angle",
    "ankle_angle",
    "gait_cycle",
    "anomaly_prob"
])

csv_path = os.path.join(output_dir, "gait_features.csv")
df.to_csv(csv_path, index=False)

print("✅ gait_features.csv 已生成:", csv_path)

# =========================================
# 15. 生成示例步态图
# =========================================
import matplotlib.pyplot as plt

example_patient = df["patient_id"].iloc[0]
example_df = df[df["patient_id"] == example_patient]

plt.figure()
plt.plot(example_df["time"], example_df["knee_angle"], label="knee")
plt.plot(example_df["time"], example_df["ankle_angle"], label="ankle")
plt.legend()
plt.title("Gait Feature Example")

img1_path = os.path.join(output_dir, "gait_feature_example.png")
plt.savefig(img1_path, dpi=300)
plt.close()

print("✅ gait_feature_example.png 已生成:", img1_path)

# =========================================
# 16. 异常概率分布图
# =========================================
plt.figure()
plt.hist(df["anomaly_prob"], bins=30)
plt.title("Anomaly Probability Distribution")

img2_path = os.path.join(output_dir, "gait_anomaly_hist.png")
plt.savefig(img2_path, dpi=300)
plt.close()

print("✅ gait_anomaly_hist.png 已生成:", img2_path)

# =========================================
# 17. 输出目录提示
# =========================================
print("\n📂 所有输出文件位置:")
print(os.path.abspath(output_dir))