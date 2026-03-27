# -*- coding: utf-8 -*-
"""
多模态融合模块 - 独立执行脚本 (基于 decision_advanced 逻辑)
"""
import argparse
import os
from api import FusionConfig, run_fusion_pipeline, generate_patient_report


def main():
    # 获取当前脚本所在目录 (即 code/module_fusion/)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 获取项目根目录 (即 Orthopedic-Recovery/)
    root_dir = os.path.dirname(os.path.dirname(current_dir))

    parser = argparse.ArgumentParser(description="Multimodal Fusion Module (Advanced)")

    # 输入数据：指向项目根目录下的 data 文件夹
    parser.add_argument("--icf_path", type=str,
                        default=os.path.join(root_dir, "data/icf/demo_output/icf_time_series.csv"))
    parser.add_argument("--gait_path", type=str,
                        default=os.path.join(root_dir, "data/gait/demo_output/gait_features.csv"))
    parser.add_argument("--sensor_path", type=str,
                        default=os.path.join(root_dir, "data/sensor/demo_output/imu_action_scores.csv"))

    # ================= 修改了这里 =================
    # 输出路径：强制指向与 main.py 同目录下的 demo_output 文件夹
    # 结果会保存在 code/module_fusion/demo_output/ 下
    parser.add_argument("--out_dir", type=str,
                        default=os.path.join(current_dir, "demo_output"))
    # ==============================================

    # 默认选一个测试患者生成报告，对齐任务分配的 "demo_output/patient_001_report.json"
    parser.add_argument("--test_patient", type=str, default="P0001")

    args = parser.parse_args()

    print("=== 初始化多模态融合流程 (Advanced Strategy) ===")
    config = FusionConfig(
        icf_path=args.icf_path,
        gait_path=args.gait_path,
        sensor_path=args.sensor_path,
        output_dir=args.out_dir
    )

    # 1. 运行融合管线
    fused_df = run_fusion_pipeline(config)
    if fused_df.empty:
        print("[Warning] 输入数据为空，请检查各模块前置输出。")
        return

    print(f"[Success] 成功执行高级融合逻辑！共处理 {len(fused_df)} 位患者。")
    print(f"[Success] 结果已保存至: {os.path.join(config.output_dir, 'fusion_output.csv')}")

    # 2. 生成单病人 JSON 报告 (为了满足最终 Demo 要求)
    # 自动容错：如果指定的 test_patient 没数据，拿表里第一个人
    target_id = args.test_patient if args.test_patient in fused_df["patient_id"].values else \
    fused_df["patient_id"].iloc[0]

    row_data = fused_df[fused_df["patient_id"] == target_id].iloc[0]
    report = generate_patient_report(target_id, row_data, config.output_dir)

    print(f"\n[Demo] 已生成病人 {target_id} 的最终报告：")
    print(f"  - 动态阈值: {report['dynamic_threshold_used']}")
    print(f"  - 风险得分: {report['risk_score']}")
    print(f"  - 风险等级: {report['risk_level']}")
    print(f"  - JSON位置: {os.path.join(config.output_dir, f'{target_id}_report.json')}")


if __name__ == "__main__":
    main()