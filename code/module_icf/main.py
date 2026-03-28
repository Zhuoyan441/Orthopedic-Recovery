"""
ICF模块主程序
运行流程：生成仿真数据 → 训练预测模型 → 保存预测结果
"""
import os
import sys

# ========== 终极方案：直接导入同目录的api.py ==========
# 获取当前main.py所在目录（module_icf）
current_dir = os.path.dirname(os.path.abspath(__file__))
# 将当前目录加入Python路径（确保能找到api.py）
sys.path.append(current_dir)

# 直接导入同目录的api.py（无需包结构）
import api

def main():
    """主函数：执行ICF数据生成+模型预测全流程"""
    # 1. 生成ICF仿真数据（50个患者，固定6个时间步）
    df_icf = api.generate_icf_data(n_patients=50)

    # 2. 训练ICF预测模型
    model, scaler, test_results = api.train_icf_model(df_icf, epochs=200)

    # 3. 保存预测结果（icf_predictions.csv）
    df_pred = api.save_icf_predictions(test_results)

    # 4. 保存训练好的模型
    api.save_model(model)

    # 输出最终结果汇总
    print("\n🎉 全流程执行完成！")
    print(f"- 仿真数据：demo_output/icf_time_series.csv（{len(df_icf)}条记录）")
    print(f"- 预测结果：demo_output/icf_predictions.csv（{len(df_pred)}条记录）")
    print(f"- 模型文件：demo_output/icf_transformer_model.pth")

if __name__ == '__main__':
    # 确保demo_output目录存在
    os.makedirs('demo_output', exist_ok=True)
    main()