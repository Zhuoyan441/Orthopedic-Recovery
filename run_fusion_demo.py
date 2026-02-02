# -*- coding: utf-8 -*-
"""融合模块运行脚本（原型阶段）"""
from fusion.decision import FusionConfig, run_fusion_pipeline


def main() -> None:
    config = FusionConfig(
        icf_path="data/icf/example_icf_output.csv",
        gait_path="data/gait/example_gait_output.csv",
        sensor_path="data/sensor/example_sensor_output.csv",
        output_path="data/fusion/example_fusion_output.csv",
    )
    output_df = run_fusion_pipeline(config)
    print(f"Fusion output rows: {len(output_df)}")
    print(f"Saved to: {config.output_path}")


if __name__ == "__main__":
    main()
