import torch
import torch.nn as nn
import numpy as np


# 1D-ResNeXt 网络结构（与周子昊训练的 imu_model.pth 保持一致）
class ResNeXtBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, cardinality=32, stride=1):
        super().__init__()
        D = cardinality * 4
        self.conv1 = nn.Conv1d(in_channels, D, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(D)
        self.conv2 = nn.Conv1d(D, D, kernel_size=3, stride=stride, groups=cardinality, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(D)
        self.conv3 = nn.Conv1d(D, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += identity
        return self.relu(out)


class ResNeXt1D(nn.Module):
    def __init__(self, in_channels=561, num_classes=6, cardinality=32):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(64, 64, 3, cardinality)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, cardinality):
        layers = [ResNeXtBlock1D(in_channels, out_channels, cardinality)]
        for _ in range(1, blocks):
            layers.append(ResNeXtBlock1D(out_channels, out_channels, cardinality))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class IMUPredictor:
    def __init__(self, weight_path='imu_model.pth'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ResNeXt1D()
        try:
            state_dict = torch.load(weight_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
        except FileNotFoundError:
            print(f"Error: 找不到权重文件 {weight_path}")

        # 对应 UCI-HAR 官方动作顺序
        self.actions = ["行走", "上楼", "下楼", "坐下", "站立", "躺下"]

    def predict(self, input_data):
        """
        input_data: 应该是形状为 (N, 561) 的 numpy 数组
        """
        self.model.eval()
        data = np.array(input_data)
        if len(data.shape) == 1:
            data = data.reshape(1, -1)

        # 转换为 Tensor 满足 Conv1d 维度: (Batch, Channels, Length) -> (N, 561, 1)
        x = torch.tensor(data, dtype=torch.float32).unsqueeze(2).to(self.device)

        with torch.no_grad():
            outputs = self.model(x)
            probs = torch.softmax(outputs, dim=1)
            confidences, preds = torch.max(probs, dim=1)

        return {
            "preds": preds.cpu().numpy(),
            "probs": probs.cpu().numpy(),
            "confidences": confidences.cpu().numpy(),
            "action_names": [self.actions[i] for i in preds.cpu().numpy()]
        }