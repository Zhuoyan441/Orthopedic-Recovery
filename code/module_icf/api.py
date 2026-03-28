"""
ICF模块接口文件
提供数据生成、模型预测、结果保存等核心接口
"""
import os
import sys
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# 添加项目根目录到路径（确保能导入/读取demo_output）
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# 基础配置
np.random.seed(42)
random.seed(42)
DEVICE = torch.device('cpu')
OUTPUT_DIR = 'demo_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== 数据生成接口 ====================
def gen_phases(n_visits=6):
    """生成6个时间步的康复阶段"""
    return ['早期']*2 + ['中期']*2 + ['晚期']*2

def sample_from_phase(phase):
    """生成ICF评分（40-200整数）"""
    if phase == '早期':
        icf = np.random.randint(120, 181)
    elif phase == '中期':
        icf = np.random.randint(80, 121)
    else:
        icf = np.random.randint(40, 81)
    return int(np.clip(icf, 40, 200))

def gen_rom_vas(phase, t, base_rom, base_vas):
    """生成ROM（0-150一位小数）、VAS（0-10整数）"""
    if phase == '早期':
        trend_rom = 5.0
        trend_vas = 2.0
    elif phase == '中期':
        trend_rom = 3.0
        trend_vas = 1.0
    else:
        trend_rom = 1.0
        trend_vas = 0.5

    noise = np.random.normal(0, 1.0) if phase == '晚期' else np.random.normal(0, 2.0)
    rom = round(np.clip(base_rom + trend_rom * t + noise, 0, 150), 1)
    vas = int(round(np.clip(base_vas - trend_vas * t + noise, 0, 10)))
    return rom, vas

def generate_icf_data(n_patients=50, save_path=None):
    """
    生成ICF仿真数据
    :param n_patients: 患者数量
    :param save_path: 保存路径（默认demo_output/icf_time_series.csv）
    :return: 生成的DataFrame
    """
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, 'icf_time_series.csv')

    data = []
    start_date = datetime(2024, 1, 1)

    for pid in range(1, n_patients+1):
        patient_id = f"P{pid:03d}"
        n_visits = 6
        phases = gen_phases(n_visits)
        base_rom = random.uniform(20, 50)
        base_vas = random.uniform(6, 9)

        for t in range(n_visits):
            assess_date = start_date + timedelta(weeks=2*t)
            assessment_date = assess_date.strftime('%Y-%m-%d')
            phase = phases[t]
            icf_total = sample_from_phase(phase)
            rom, vas = gen_rom_vas(phase, t, base_rom, base_vas)

            data.append({
                'patient_id': patient_id,
                'assessment_date': assessment_date,
                'rehab_phase': phase,
                'time_step': t+1,
                'icf_total': icf_total,
                'rom': rom,
                'vas': vas
            })

    # 生成并验证数据
    df = pd.DataFrame(data)
    _validate_icf_data(df)

    # 保存数据
    df.to_csv(save_path, index=False, encoding='utf-8')
    print(f"✅ ICF仿真数据已生成：{save_path}")
    return df

def _validate_icf_data(df):
    """内部函数：验证ICF数据格式/范围"""
    # ICF验证
    if not ((df['icf_total'] >= 40) & (df['icf_total'] <= 200) & (df['icf_total'].dtype == 'int64')).all():
        raise ValueError("❌ ICF数据不符合要求：需40-200整数！")
    # ROM验证
    rom_check = (df['rom'] >= 0) & (df['rom'] <= 150)
    rom_decimal_check = df['rom'].apply(lambda x: len(str(x).split('.')[-1]) <= 1)
    if not (rom_check & rom_decimal_check).all():
        raise ValueError("❌ ROM数据不符合要求：需0-150一位小数！")
    # VAS验证
    if not ((df['vas'] >= 0) & (df['vas'] <= 10) & (df['vas'].dtype == 'int64')).all():
        raise ValueError("❌ VAS数据不符合要求：需0-10整数！")
    # 时间步验证
    if not all(df.groupby('patient_id').size() == 6):
        raise ValueError("❌ 存在非6个时间步的患者！")

# ==================== 预测模型接口 ====================
class ICFPredictorTransformer(nn.Module):
    """ICF预测Transformer模型"""
    def __init__(self, input_size=3, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embedding(x)
        seq_len = x.size(1)
        positions = torch.arange(seq_len).unsqueeze(0).to(x.device)
        pos_encoding = torch.zeros_like(x)
        for i in range(0, self.d_model, 2):
            pos_encoding[:, :, i] = torch.sin(positions / (10000 ** (i / self.d_model)))
            if i + 1 < self.d_model:
                pos_encoding[:, :, i+1] = torch.cos(positions / (10000 ** (i / self.d_model)))
        x = x + pos_encoding
        x = self.transformer(x)
        return self.fc(x[:, -1, :])

def create_sequences(df, n_history=3):
    """构建时序序列（用于模型输入）"""
    sequences, targets, patient_ids, assess_dates = [], [], [], []
    for pid, group in df.groupby('patient_id'):
        group = group.sort_values('time_step')
        features = group[['icf_total', 'rom', 'vas']].values
        dates = group['assessment_date'].values

        for i in range(len(features) - n_history):
            sequences.append(features[i:i+n_history])
            targets.append(features[i+n_history, 0])
            patient_ids.append(pid)
            assess_dates.append(dates[i+n_history])
    return np.array(sequences), np.array(targets), patient_ids, assess_dates

def train_icf_model(df, epochs=200, batch_size=16, lr=0.001):
    """
    训练ICF预测模型
    :param df: ICF数据DataFrame
    :param epochs: 训练轮次
    :param batch_size: 批次大小
    :param lr: 学习率
    :return: 训练好的模型、scaler、测试集结果
    """
    # 构建序列（返回4个值：X, y_true, patient_ids, assess_dates）
    X, y_true, patient_ids, assess_dates = create_sequences(df)
    
    # ========== 修复核心：train_test_split返回8个值 ==========
    # 输入4个列表 → 返回8个列表（每个输入拆分为train+test）
    X_train, X_test, y_train, y_test, pid_train, pid_test, date_train, date_test = train_test_split(
        X, y_true, patient_ids, assess_dates, test_size=0.2, random_state=42
    )

    # 标准化
    scaler = StandardScaler()
    shape_train = X_train.shape
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(shape_train)
    X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    # 转换为张量
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(DEVICE)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(DEVICE)

    # 初始化模型
    model = ICFPredictorTransformer().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 训练模型
    print(f"🚀 开始训练ICF预测模型（{epochs}轮）...")
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=batch_size, shuffle=True
    )

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")

    # 测试集预测
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor).cpu().numpy().flatten()

    # 计算指标
    mse = np.mean((y_test - y_pred)**2)
    r2 = r2_score(y_test, y_pred)
    print(f"✅ 模型训练完成！测试集MSE: {mse:.4f}, R²: {r2:.4f}")

    # 返回测试集结果时，只保留需要的部分（X_test, y_test, y_pred, pid_test, date_test）
    return model, scaler, (X_test, y_test, y_pred, pid_test, date_test)

def save_icf_predictions(test_results, save_path=None):
    """
    保存预测结果到icf_predictions.csv
    :param test_results: 测试集结果 (X_test, y_test, y_pred, pid_test, date_test)
    :param save_path: 保存路径（默认demo_output/icf_predictions.csv）
    :return: 预测结果DataFrame
    """
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, 'icf_predictions.csv')

    X_test, y_test, y_pred, pid_test, date_test = test_results
    # 整理预测结果
    df_pred = pd.DataFrame({
        'patient_id': pid_test,
        'assessment_date': date_test,
        'true_icf': y_test.astype(int),
        'pred_icf': np.clip(np.round(y_pred).astype(int), 40, 200)
    })

    # 去重并排序
    df_pred = df_pred.drop_duplicates(subset=['patient_id', 'assessment_date'])
    df_pred = df_pred.sort_values(['patient_id', 'assessment_date']).reset_index(drop=True)

    # 保存
    df_pred.to_csv(save_path, index=False, encoding='utf-8')
    print(f"✅ 预测结果已保存：{save_path}")
    return df_pred

def save_model(model, save_path=None):
    """保存训练好的模型"""
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, 'icf_transformer_model.pth')
    torch.save(model.state_dict(), save_path)
    print(f"✅ 模型已保存：{save_path}")
    return save_path