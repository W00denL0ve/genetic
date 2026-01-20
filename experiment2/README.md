# 实验二：遗传算法求解 TSP 报告与代码

本文件夹包含实验二的代码实现与运行说明。

目录结构：
- code/tsp_ga.py：遗传算法实现主文件（PMX、OX、变异、锦标赛选择、精英策略等）。
- code/utils.py：辅助函数（保存/加载、绘图）。
- figures/：运行后生成收敛曲线和最优路径图。
- results/：保存运行日志与结果摘要。

快速运行：

1. 安装依赖（虚拟环境）：
   pip install -r requirements.txt
2. 进入目录：
   cd "genetic\experiment2\code"
3. 运行示例：
   python tsp_ga.py --pop_size 200 --generations 500 --crossover 0.9 --mutation 0.2 --seed 42

输出：在上级目录的 figures/ 和 results/ 下生成图片与日志。
