# code/module_gait/api.py

import pandas as pd

DATA_PATH = "data/gait/demo_output/gait_features.csv"

def load_data():
    return pd.read_csv(DATA_PATH)

def get_all_patients():
    df = load_data()
    return sorted(df["patient_id"].unique().tolist())

def get_patient_data(patient_id):
    df = load_data()
    return df[df["patient_id"] == patient_id]

def get_one_sample(patient_id):
    df = get_patient_data(patient_id)
    sid = df["sample_id"].iloc[0]
    return df[df["sample_id"] == sid]

def compute_risk_score(patient_id):
    df = get_patient_data(patient_id)
    return float(df["anomaly_prob"].mean())
