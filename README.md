# 遗传算法实验（实验一与实验二）

本仓库包含《人工智能导论》两项遗传算法实验的代码与报告模板：

- 实验一（最值求解）：位于 `experiment1/`，脚本 `experiment1_ga.py`。
- 实验二（TSP 求解）：位于 `experiment2/`，实现与基准脚本位于 `experiment2/code/`。

环境与依赖

- 推荐 Python 版本：3.10（最低 3.8）。
- 在仓库根目录可安装依赖：
  pip install -r requirements.txt
- 额外（可选，运行基准脚本需要）：
  pip install pandas

快速运行说明（Windows PowerShell）

1. 实验一（函数最小值）
   cd "genetic\experiment1"
   python experiment1_ga.py --pop_size 100 --generations 200 --crossover 0.8 --mutation 0.02

   输出（在当前目录）：收敛曲线 PNG 与日志 TXT，例如 `ga_experiment1_results.png` 与 `ga_experiment1_log.txt`。

2. 实验二（TSP）
   cd "genetic\experiment2\code"
   python tsp_ga.py --pop_size 200 --generations 500 --crossover 0.9 --mutation 0.2 --seed 42

   输出：上级目录的 `experiment2/figures/`（图片）和 `experiment2/results/`（日志、调试信息、基准 CSV）。

3. 基准测试（参数敏感性 / 算子对比）
   cd "genetic\experiment2\code"
   # 参数敏感性（示例）
   python benchmark.py --mode sensitivity --repeats 3
   # 算子对比（示例）
   python benchmark.py --mode operators --repeats 3

注意与建议

- 为保证可复现，使用 `--seed` 固定随机种子；进行统计实验时对每组参数做多次重复（建议 20~30 次）并记录均值与标准差。
- 如果生成的图文件名过长或想避免覆盖，可在 `tsp_ga.py` 中调整文件名设置（目前使用包含关键参数的文件名）。
- 若需在更大规模上运行基准实验，建议在更高性能机器或用较小 repeats 进行预调试。

文件结构概要

- experiment1/
  - experiment1_ga.py
  - requirements.txt
  - README.md
- experiment2/
  - code/
    - tsp_ga.py
    - utils.py
    - benchmark.py
  - figures/
  - results/
- experiment2_report.md（实验二报告框架与实现说明）

如需我将任何运行结果截图或关键数据嵌入实验报告（`experiment2_report.md`），或把实验二的完整代码附录插入报告，请告诉我具体要求。
