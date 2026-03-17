# code/module_gait/api.py

import pandas as pd

def load_gait_data():
    df = pd.read_csv("data/gait/demo_output/gait_features.csv")
    return df


def get_patient_gait(patient_id):
    df = load_gait_data()
    return df[df["patient_id"] == patient_id]


def compute_risk_score(patient_df):
    # 简单平均异常概率作为风险
    return patient_df["anomaly_prob"].mean()