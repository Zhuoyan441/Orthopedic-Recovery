"""
icf_data_simulator.py
生成带临床逻辑的ICF康复仿真数据（标准格式版）
ICF范围: 40-200整数 | ROM范围: 0-150一位小数 | VAS范围: 0-10整数
作者：周亦沁
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# 设置随机种子（保证每次结果一致）
np.random.seed(42)

def generate_patient_data(patient_id, n_assessments=6):
    """
    生成单个患者的时序数据
    评估时间点：术后1周、2周、1个月、2个月、3个月、6个月
    time_step: 0,1,2,3,4,5
    
    数据范围：
    - icf_total: 40-200 整数（与ROM/VAS强相关）
    - rom: 0-150 一位小数（单位：度）
    - vas: 0-10 整数
    """
    data = []
    assessment_days = [7, 14, 30, 60, 90, 180]  # 天数
    
    for time_step, day in enumerate(assessment_days):
        # 确定康复阶段（基于实际天数判断）
        if day <= 14:
            rehab_phase = "早期"
        elif day <= 90:
            rehab_phase = "中期"
        else:
            rehab_phase = "晚期"
        
        # ==================== 先计算ROM（0-150一位小数） ====================
        # ROM恢复速度比ICF快
        if rehab_phase == "早期":
            rom_base = np.random.uniform(20, 50)
        elif rehab_phase == "中期":
            # 中期ROM提升快
            rom_base = 50 + (day - 14) / 76 * 60
        else:
            # 晚期接近正常活动度
            rom_base = np.random.uniform(110, 150)
        
        # 限制范围并保留一位小数
        rom = round(np.clip(rom_base + np.random.normal(0, 8), 0, 150), 1)
        
        # ==================== 再计算VAS（0-10整数） ====================
        # 疼痛随时间递减，VAS为整数
        if rehab_phase == "早期":
            vas_base = np.random.uniform(5, 10)
        elif rehab_phase == "中期":
            vas_base = 8 - (day - 14) / 76 * 4
        else:
            vas_base = np.random.uniform(0, 3)
        
        # 限制范围并四舍五入为整数
        vas = int(np.clip(round(vas_base + np.random.normal(0, 0.5)), 0, 10))
        
        # ==================== 最后计算ICF（40-200整数，与ROM/VAS强相关） ====================
        # 强相关关系：ROM越高 → ICF越高；VAS越低 → ICF越高
        base_icf = np.random.uniform(60, 80)  # 基础分（早期）
        
        if rehab_phase == "早期":
            rom_factor = (rom / 150) * 30  # ROM贡献30分
            vas_penalty = vas * 2  # VAS每分扣2分ICF
            icf_total = base_icf + rom_factor - vas_penalty + np.random.randint(-5, 5)
        elif rehab_phase == "中期":
            rom_factor = (rom / 150) * 60
            vas_penalty = vas * 1.5
            icf_total = base_icf + 30 + rom_factor - vas_penalty + np.random.randint(-5, 5)
        else:
            rom_factor = (rom / 150) * 80
            vas_penalty = vas * 1
            icf_total = base_icf + 60 + rom_factor - vas_penalty + np.random.randint(-3, 3)
        
        # 限制范围并确保为整数
        icf_total = int(np.clip(round(icf_total), 40, 200))
        
        # ==================== 按标准格式组装记录 ====================
        record = {
            "patient_id": f"P{patient_id:03d}",
            "time_step": time_step,
            "rehab_phase": rehab_phase,
            "icf_total": icf_total,
            "rom": rom,
            "vas": vas,
        }
        data.append(record)
    
    return pd.DataFrame(data)

def generate_dataset(n_patients=50):
    """生成完整数据集（标准格式）"""
    all_data = []
    print(f"开始生成 {n_patients} 名患者的仿真数据...")
    print("数据范围：ICF 40-200整数 | ROM 0-150度（一位小数）| VAS 0-10整数")
    
    for pid in range(1, n_patients + 1):
        patient_data = generate_patient_data(pid)
        all_data.append(patient_data)
    
    df = pd.concat(all_data, ignore_index=True)
    
    # 确保列顺序完全符合要求：patient_id, time_step, rehab_phase, icf_total, rom, vas
    df = df[["patient_id", "time_step", "rehab_phase", "icf_total", "rom", "vas"]]
    
    # 保存到当前目录
    save_path = "demo_output/icf_time_series.csv"
    df.to_csv(save_path, index=False, encoding='utf-8-sig')
    
    print(f"\n✅ 数据生成成功！保存至: {save_path}")
    print(f"\n数据格式验证（前5行）：")
    print(df.head())
    print(f"\n数据统计：")
    print(df[['icf_total', 'rom', 'vas']].describe())
    print(f"\n各康复阶段分布:")
    print(df['rehab_phase'].value_counts())
    print(f"\n各时间步分布:")
    print(df['time_step'].value_counts().sort_index())
    
    return df

def load_icf_data(data_path: str) -> pd.DataFrame:
    """
    统一的ICF临床数据加载接口
    
    输入:
        data_path: CSV文件路径（如 "simulated_icf_clinical_data.csv"）
    
    输出:
        DataFrame，包含标准化列：
        - patient_id: 患者ID（字符串）
        - time_step: 时间步（0-5，整数）
        - rehab_phase: 康复阶段（早期/中期/晚期）
        - icf_total: ICF总分（40-200，整数）
        - rom: 关节活动度（0-150，一位小数，单位：度）
        - vas: 疼痛评分（0-10，整数）
    
    使用示例:
        >>> df = load_icf_data("simulated_icf_clinical_data.csv")
        >>> print(df.head())
    """
    required_columns = ['patient_id', 'time_step', 'rehab_phase', 'icf_total', 'rom', 'vas']
    
    try:
        df = pd.read_csv(data_path, encoding='utf-8-sig')
        
        # 数据验证
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"数据文件缺少必要列: {missing_cols}")
        
        # 数据类型转换（确保范围正确）
        df['patient_id'] = df['patient_id'].astype(str)
        df['time_step'] = df['time_step'].astype(int)
        df['rehab_phase'] = df['rehab_phase'].astype(str)
        df['icf_total'] = df['icf_total'].astype(int)  # ICF是整数
        df['rom'] = df['rom'].astype(float)  # ROM是浮点数
        df['vas'] = df['vas'].astype(int)  # VAS是整数
        
        # 范围验证
        assert df['icf_total'].between(40, 200).all(), "ICF总分不在40-200范围内"
        assert df['rom'].between(0, 150).all(), "ROM不在0-150范围内"
        assert df['vas'].between(0, 10).all(), "VAS不在0-10范围内"
        
        print(f"✅ 成功加载数据: {data_path}")
        print(f"   记录数: {len(df)}")
        print(f"   患者数: {df['patient_id'].nunique()}")
        print(f"   ICF范围: {df['icf_total'].min()}-{df['icf_total'].max()}")
        print(f"   ROM范围: {df['rom'].min():.1f}-{df['rom'].max():.1f}")
        print(f"   VAS范围: {df['vas'].min()}-{df['vas'].max()}")
        
        return df
        
    except Exception as e:
        print(f"❌ 加载数据时出错: {str(e)}")
        raise

# 使用示例
if __name__ == "__main__":
    # 生成数据
    df_generated = generate_dataset(n_patients=50)
    
    # 测试加载接口
    df_loaded = load_icf_data("demo_output/icf_time_series.csv")
    
    # 验证数据一致性
    assert len(df_generated) == len(df_loaded)
    assert list(df_generated.columns) == list(df_loaded.columns)
    print("\n✅ 数据生成与加载接口测试通过！")
    print("✅ 数据范围符合统一标准：ICF 40-200整数 | ROM 0-150一位小数 | VAS 0-10整数")