# -*- coding: utf-8 -*-
"""
可解释性模块 (XAI) - 独立执行脚本
完全依赖 Fusion 模块的结论，进行归因画图。
"""
import argparse
import os
import sys

# 动态获取路径，解决 ModuleNotFoundError 问题
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(current_dir)

from api import run_xai


def main():
    parser = argparse.ArgumentParser(description="XAI Explainer for Orthopaedic Rehab")

    # 默认强制读取你刚才做好的 Fusion 输出文件
    default_input = os.path.join(root_dir, "code", "module_fusion", "demo_output", "fusion_output.csv")

    parser.add_argument("--input_path", type=str, default=default_input, help="指向融合模块输出的 CSV 路径")
    parser.add_argument("--out_dir", type=str, default=os.path.join(current_dir, "demo_output"),
                        help="XAI 结果保存路径")
    parser.add_argument("--patient_id", type=str, default="P0001", help="需要解释的病历号")

    args = parser.parse_args()

    print("=== 初始化可解释性分析 (XAI) ===")
    try:
        report = run_xai(
            input_path=args.input_path,
            out_dir=args.out_dir,
            patient_id=args.patient_id
        )
        print("\n[XAI 最终诊断文本]:")
        print(f"👉 {report['explain_one_line']}")
    except Exception as e:
        print(f"[XAI 错误] {e}")
        print("提示：请先确保已经运行了 code/module_fusion/main.py 并且生成了 fusion_output.csv！")


if __name__ == "__main__":
    main()