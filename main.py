# -*- coding: utf-8 -*-

"""
Orthopedic Recovery Intelligent System
Main Entry (Prototype Stage)

当前阶段目标：
- 验证各模块数据接口是否统一
- 不关心模型精度，只关心流程是否跑通
"""

from fusion.decision import FusionConfig, run_fusion_pipeline

def main():
    print("=== 骨科康复智能系统：模拟验证阶段 ===")

    print("1. ICF 模块数据接口：待接入")
    print("2. 步态分析模块数据接口：待接入")
    print("3. IMU 动作识别模块数据接口：待接入")
    print("4. 多模态融合模块：已接入")

    """ 取消注释以下代码以运行多模态融合模块示例 """
    # config = FusionConfig(
    #     icf_path="data/icf/example_icf_output.csv",
    #     gait_path="data/gait/example_gait_output.csv",
    #     sensor_path="data/sensor/example_sensor_output.csv",
    #     output_path="data/fusion/example_fusion_output.csv",
    # )
    # output_df = run_fusion_pipeline(config)
    # print(f"融合模块输出条数: {len(output_df)}")
    # print(f"融合结果保存到: {config.output_path}")

    print("系统主流程结构已建立")

if __name__ == "__main__":
    main()
