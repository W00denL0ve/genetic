# 实验二报告框架：遗传算法求解 TSP 闏题

1. 绪论

   1.1 背景与意义

   - 旅行商问题（TSP, Travelling Salesman Problem）是组合优化中的经典问题：给定若干城市及两两之间的距离，要求寻找一条最短闭合路径，使旅行商恰好访问每个城市一次并回到起点。TSP 在物流配送、印刷电路板布线、机器人路径规划、基因排序等实际问题中具有重要应用价值。
   - TSP 属于 NP-hard 问题，当城市数量较大时，精确算法计算量呈指数增长。启发式和元启发式算法（如遗传算法、模拟退火、粒子群等）在求取近似最优解方面表现良好，能够在可接受时间内给出质量较高的解。
   - 本实验旨在使用遗传算法（GA）求解 20 个城市的随机 TSP，比较不同编码、交叉与变异算子的影响，并在报告中分析算法收敛特性与参数敏感性。

   1.2 实验目标

   - 实现基于顺序编码（permutation encoding）的遗传算法以解决 20 城市 TSP；
   - 设计并实现常用的交叉（例如 PMX / OX）与变异（交换/倒位）算子；
   - 探究参数（种群大小、代数、交叉概率、变异概率、精英数）对算法收敛速度和解质量的影响；
   - 可视化最优路径与收敛曲线，记录实验现象并给出改进建议（如引入 2-opt 局部搜索）。

2. 相关理论与技术基础

   2.1 TSP 问题形式化

   - 给定城市集合 C = {c_1, c_2, ..., c_n}，及城市 i 与 j 之间的距离 d(i,j)（本实验使用欧氏距离）。目标是寻找一条排列 π = (π_1, π_2, ..., π_n)，使得目标函数 L(π) 最小：

     L(π) = d(π_n, π_1) + sum_{k=1}^{n-1} d(π_k, π_{k+1})

   - 约束：π 为城市索引的一次排列（每个城市恰好出现一次）。

   2.2 遗传算法概述

   - 遗传算法是一类随机化的全局优化方法，主要受生物进化启发，核心操作为编码、选择、交叉、变异与替换。GA 在排列类问题（如 TSP）上常用顺序编码（直接表示访问序列），并采用专门的交叉/变异算子以保持可行性（避免重复城市或遗漏城市）。

   2.3 针对 TSP 的常用编码与算子

   - 编码方式：顺序编码（Permutation Representation）。染色体长度为 n，染色体元素为城市索引的一个排列。

   - 交叉算子（示例）：
     - 部分映射交叉（PMX，Partially Mapped Crossover）：通过映射区段交换片段并修复冲突，保证子代为合法排列。适用于保持局部顺序信息。
     - 顺序交叉（OX，Order Crossover）：选择一个区间保留父代片段，然后按父代中出现的顺序填充剩余位置，保留相对顺序信息。
     - 循环交叉（CX）：保持每个基因在父代中对应位置的循环关系，适用于某些特殊情形。

   - 变异算子（示例）：
     - 交换变异（swap）：随机选择两个位置交换，简单且能引入局部结构变化。
     - 逆转变异（inversion）：随机选取区间并反转，该操作常用于改变路径的局部顺序，配合 2-opt 局部优化效果好。
     - 插入变异（insertion）：将某一基因移至另一位置。

   - 选择算子：
     - 锦标赛选择（Tournament）：从若干个体中随机抽取 k 个，选出适应度最优（这里为路径长度最小者）作为父代。实现简单且易控制选择压力。
     - 轮盘赌选择（Roulette）：基于适应度比例选择，但在最小化问题中需将适应度转换或归一化，且对极端值敏感。

   - 精英策略：为防止优秀个体被淘汰，常保留若干最优个体直接进入下一代（elitism），以保证解质量单调不降。

3. 系统 / 模型设计

   3.1 总体流程

   - 系统流程（伪代码）：

     1. 生成城市坐标与距离矩阵；
     2. 初始化种群（随机排列若干个体）；
     3. 评估每个个体的路径长度（适应度）；
     4. 记录并保存当前代的最优与平均值；
     5. 选择父代（例如锦标赛选择）；
     6. 按交叉概率对父代成对进行交叉（PMX / OX），生成子代；
     7. 对子代按变异概率执行变异算子（swap / inversion）；
     8. 合并精英个体与新子代形成下一代；
     9. 若达到最大代数或满足终止准则则结束，否则转第 3 步。

   3.2 数据结构与表示

   - 城市坐标：二维数组 City_Map（shape: n x 2），元素为浮点坐标值。
   - 距离矩阵：预计算对称距离矩阵 Dist（n x n），以加速适应度计算：Dist[i,j] = Euclidean(City_Map[i], City_Map[j])。
   - 染色体表示：长度为 n 的整数数组（例如 numpy array），每个元素为城市索引 0..n-1 的一次排列。

   3.3 适应度函数

   - 给定染色体 π，计算路径总长 L(π)（见 2.1），适应度即为 L(π)。由于为最小化问题，直接使用路径长度进行比较。

   3.4 算子设计细节

   - 初始化：随机产生 M 个合法排列作为初始种群。

   - 选择：采用锦标赛选择，参数 k（锦标赛规模）用于控制选择压力；每次选择从种群中随机抽取 k 个个体并返回最短路径者作为父代。

   - 交叉：首选 PMX 或 OX。以 PMX 为例实现要点：
     1. 随机选取两个交叉点 i<j；
     2. 将父母在区间 [i,j] 的片段互换到子代对应位置；
     3. 根据映射关系修复子代中重复或缺失的元素，保证为合法排列。

   - 变异：采用交换变异或逆转变异，变异后不需要额外修复，因为操作保持排列完整性。

   - 精英保留：每代保留 E 个最优个体直接进入下一代，剩余位置由子代填充。

   3.5 参数与接口

   - 主要参数：种群大小 pop_size、代数 generations、交叉概率 crossover_prob、变异概率 mutation_prob、锦标赛大小 tournament_k、精英数 elites。
   - 输入/输出接口：
     - 输入：城市坐标（随机生成或文件加载）、参数配置、随机种子；
     - 输出：最优路径（城市索引序列）、最短路径长度、收敛曲线图、最优路径可视化图、运行日志。

4. 系统 / 模型实现

   4.1 开发环境与依赖

   - 推荐 Python 版本：3.10（实验使用并验证为 Python 3.10）。
   - 主要依赖库：
     - numpy：数值计算与数组操作
     - matplotlib：结果可视化
     - typing（标准库，可选，用于类型注释）
   - 建议创建虚拟环境并安装依赖：
     - python -m venv .venv
     - .\.venv\Scripts\Activate.ps1
     - pip install numpy matplotlib

   4.2 文件与目录

   - 根目录：`experiment2/`。
     - `experiment2/code/`：实现文件 `tsp_ga.py`、`utils.py`。
     - `experiment2/figures/`：保存收敛曲线与路径可视化图像。
     - `experiment2/data/`：若需保存城市坐标或距离矩阵文件。
     - `experiment2/results/`：保存日志与结果摘要。
   - 示例：
     - `experiment2/code/tsp_ga.py`：主实现文件（包含 GA 流程与可配置参数）。
     - `experiment2/code/utils.py`：辅助函数（距离矩阵计算、绘图、加载/保存）。

   4.3 主要模块与函数说明

   - init_population(pop_size, n_cities)
     - 功能：随机生成 pop_size 个合法排列（每个为 0..n_cities-1 的置换）。
     - 返回：shape (pop_size, n_cities) 的整数数组。

   - compute_distance_matrix(city_map)
     - 功能：基于城市坐标计算对称欧氏距离矩阵 Dist（n x n）。
     - 返回：二维 numpy 数组 Dist。

   - evaluate_population(population, dist_matrix)
     - 功能：计算种群中每个个体的路径长度；返回长度数组与最优索引。

   - selection_tournament(population, fitness, k)
     - 功能：使用锦标赛选择从 population 中选择一个父个体（返回染色体）。

   - pmx_crossover(parent1, parent2)
     - 功能：实现部分映射交叉，返回两个子代排列，保证合法性。

   - order_crossover(parent1, parent2)
     - 功能：实现顺序交叉（OX），可作为可选交叉算子。

   - mutation_swap(individual)
     - 功能：在染色体上随机交换两个位置。

   - mutation_inversion(individual)
     - 功能：随机选择区间并反转该区间中的元素。

   - apply_elitism(population, fitness, elites)
     - 功能：选出前 elites 个最优个体直接进入下一代。

   - run_ga(config)
     - 功能：主流程——读取参数、初始化种群、循环执行选择/交叉/变异/替换、记录历史并输出最优解与图像。

   4.4 完整实现流程（伪代码）

   1. 参数与数据准备：
      - 读取或生成城市坐标 City_Map（shape: n x 2）；
      - 计算距离矩阵 Dist = compute_distance_matrix(City_Map)；
      - 从配置获取 pop_size、generations、crossover_prob、mutation_prob、tournament_k、elites、seed 等。

   2. 初始化：
      - population = init_population(pop_size, n)
      - fitness = evaluate_population(population, Dist)
      - history = {best: [], mean: []}

   3. 迭代主循环（for gen in range(generations)）:
      a. 记录当前最优与均值：best, mean -> history
      b. 精英保留：elite_individuals = apply_elitism(population, fitness, elites)
      c. 新子代列表 new_pop = []
      d. while len(new_pop) < pop_size - elites:
           - parent1 = selection_tournament(population, fitness, k=tournament_k)
           - parent2 = selection_tournament(population, fitness, k=tournament_k)
           - if rand() < crossover_prob:
               child1, child2 = pmx_crossover(parent1, parent2)
             else:
               child1, child2 = parent1.copy(), parent2.copy()
           - if rand() < mutation_prob: child1 = mutation_operator(child1)
           - if rand() < mutation_prob: child2 = mutation_operator(child2)
           - append child1 and child2 (截断以保持数量)
      e. population = concatenate(elite_individuals, new_pop)
      f. fitness = evaluate_population(population, Dist)
      g. （可选）早停检查

   4. 结束后：输出最优路径、最短长度，保存收敛曲线与路径可视化图。

   4.5 代码片段占位（便于复制粘贴或后续替换）

   - 文件：`experiment2/code/tsp_ga.py`

   代码片段：初始化与工具函数
   ```python
   # experiment2/code/tsp_ga.py
   # -*- coding: utf-8 -*-
   import numpy as np
   # 初始化种群
   def init_population(pop_size, n_cities):
       # TODO: 实现：返回 shape (pop_size, n_cities) 的整数排列数组
       pass

   # 计算距离矩阵
   def compute_distance_matrix(city_map):
       # TODO: 实现欧氏距离矩阵计算
       pass
   ```

   代码片段：评估与选择
   ```python
   # 评估函数
   def evaluate_population(population, dist_matrix):
       # TODO: 返回每个个体的路径长度数组
       pass

   # 锦标赛选择
   def selection_tournament(population, fitness, k=3):
       # TODO: 返回单个父染色体
       pass
   ```

   代码片段：交叉与变异
   ```python
   # PMX 交叉
   def pmx_crossover(p1, p2):
       # TODO: 实现 PMX，并返回两个子代
       pass

   # 交换变异
   def mutation_swap(individual):
       # TODO: 交换两个位置
       pass
   ```

   代码片段：主流程
   ```python
   def run_ga(city_map, pop_size=100, generations=500, crossover_prob=0.8, mutation_prob=0.1,
              tournament_k=3, elites=2, seed=None, verbose=True):
       # 1. 预处理
       Dist = compute_distance_matrix(city_map)
       population = init_population(pop_size, len(city_map))
       fitness = evaluate_population(population, Dist)

       history = {'best': [], 'mean': []}
       for gen in range(generations):
           # 记录
           # 选择/交叉/变异/精英替换
           # 更新 fitness
           # 可视化/日志
           pass
       # 返回最优解与历史
       return None
   ```

   4.6 可视化与保存

   - 绘制收敛曲线：best 与 mean 随代数变化曲线，保存为 PNG。
   - 绘制路径：根据最优染色体索引绘制城市坐标连线图，标注城市编号与起点/终点。
   - 保存日志：参数、运行时间、最优路径长度、最优序列、随机种子等信息写入文本文件。

   4.7 调试与复现建议

   - 固定随机种子（seed）以复现实验结果；对于统计实验请多次运行并计算均值与标准差。
   - 对关键算法（如 PMX）编写单元测试以验证算子保持解的合法性（无重复、无缺失）。
   - 在开发阶段先用较小的代数与人口快速验证逻辑，再放大参数进行完整实验。

5. 测试与结果分析
   - 实验设计：说明要比较的变量（如不同交叉/变异策略、不同参数组合），每组实验运行次数（建议多次取平均）。
   - 结果展示：
     - 收敛曲线（最优/平均路径长度随代数变化）示意图；
     - 最优路径可视化图（城市坐标与连接顺序）；
     - 表格：不同算法/参数下的最终最短路径长度与时间消耗；
   - 结果分析：讨论不同算子与参数对收敛速度与解质量的影响，是否存在早熟收敛，如何改进（例如增加变异、使用局部搜索混合等）。

6. 总结与展望
   - 总结主要结论（例如哪组参数/算子表现最好），算法优势与局限。
   - 展望：提出可能的改进方向，如混合局部搜索（GA+2-opt）、自适应参数控制、多种群/并行 GA、将问题规模扩大等。

7. 参考文献
   - 列出参考书目与论文（例如 Goldberg 的遗传算法书籍、常见 TSP/交叉算子论文等）。

附录
   - 实验一与实验二的结果截图
   - 代码清单：提供关键实现文件名与重要代码片段或将代码附录为独立文件（例如 `tsp_ga.py`）。
   - 运行说明：如何复现实验（命令行示例、随机种子控制）。
   - 实验原始数据：若保存随机种子、距离矩阵等，可附在此处或提供文件路径。

——

使用建议：
- 写报告时在每一节插入实验生成的图表和表格，图表文件放在 `experiment2/figures/` 下并在正文中引用文件名；代码放在 `experiment2/code/` 下。
- 若需要，我可以基于此框架生成初步代码模板（`tsp_ga.py`）、绘图脚本以及示例实验结果。
