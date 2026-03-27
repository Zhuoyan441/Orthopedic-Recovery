# -*- coding: utf-8 -*-
"""
多模态融合模块 - 独立执行脚本 (Risk-Aware Attention 架构)
"""
import argparse
import os
from api import FusionConfig, run_fusion_pipeline, generate_patient_report


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(current_dir))

    parser = argparse.ArgumentParser(description="Multimodal Fusion Module (Evidential Attention)")
    parser.add_argument("--icf_path", type=str,
                        default=os.path.join(root_dir, "data/icf/demo_output/icf_time_series.csv"))
    parser.add_argument("--gait_path", type=str,
                        default=os.path.join(root_dir, "data/gait/demo_output/gait_features.csv"))
    parser.add_argument("--sensor_path", type=str,
                        default=os.path.join(root_dir, "data/sensor/demo_output/imu_action_scores.csv"))
    parser.add_argument("--out_dir", type=str, default=os.path.join(current_dir, "demo_output"))
    parser.add_argument("--test_patient", type=str, default="P0001")

    args = parser.parse_args()

    print("=== 初始化多模态融合流程 (Uncertainty & Risk-Aware Attention) ===")
    config = FusionConfig(
        icf_path=args.icf_path,
        gait_path=args.gait_path,
        sensor_path=args.sensor_path,
        output_dir=args.out_dir
    )

    fused_df = run_fusion_pipeline(config)
    if fused_df.empty:
        print("[Error] 输入数据为空。")
        return

    print(f"[Success] 多模态注意力融合计算完成！共处理 {len(fused_df)} 位患者。")
    print(f"[Success] 结果保存至: {os.path.join(config.output_dir, 'fusion_output.csv')}")

    target_id = args.test_patient if args.test_patient in fused_df["patient_id"].values else \
    fused_df["patient_id"].iloc[0]
    row_data = fused_df[fused_df["patient_id"] == target_id].iloc[0]
    report = generate_patient_report(target_id, row_data, config.output_dir)

    print(f"\n[Demo] 测试患者 {target_id} 的动态融合分析：")
    print(
        f"  > 模型自动分配注意力权重: ICF({report['attention_weights']['icf']}), Gait({report['attention_weights']['gait']}), Sensor({report['attention_weights']['sensor']})")
    print(f"  > 个体化判定阈值: {report['dynamic_threshold_used']}")
    print(f"  > 综合风险得分: {report['risk_score']} => 风险等级: {report['risk_level']}")


if __name__ == "__main__":
    main()