# -*- coding: utf-8 -*-
"""
遗传算法求最小值问题：
目标函数 y = 10*sin(5x) + 7*|x-5| + 10，x ∈ [0, 10]
可通过命令行参数调整种群大小、迭代代数、交叉概率、变异概率等。
输出：在同目录生成图像文件 'ga_experiment1_results.png' 和日志 'ga_experiment1_log.txt'

用法示例：
python experiment1_ga.py --pop_size 100 --generations 200 --crossover 0.8 --mutation 0.02

作者：自动生成
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# 目标函数（欲最小化）
def objective(x):
    return 10.0 * np.sin(5.0 * x) + 7.0 * np.abs(x - 5.0) + 10.0

# 初始化实值染色体
def init_population(pop_size, low=0.0, high=10.0):
    return np.random.uniform(low, high, size=pop_size)

# 锦标赛选择
def tournament_selection(pop, fitness, k=3):
    pop_size = pop.shape[0]
    idx = np.random.randint(0, pop_size, size=k)
    best = idx[0]
    for i in idx[1:]:
        if fitness[i] < fitness[best]:  # 注意：适应度为目标值，越小越好
            best = i
    return pop[best]

# Blend 交叉（两点实数混合）
def blend_crossover(a, b, alpha=0.5):
    # 生成子代
    gamma = np.random.uniform(-alpha, 1 + alpha)
    child1 = (1 - gamma) * a + gamma * b
    gamma = np.random.uniform(-alpha, 1 + alpha)
    child2 = (1 - gamma) * b + gamma * a
    return child1, child2

# 高斯变异
def mutate(x, mutation_rate, sigma=0.1, low=0.0, high=10.0):
    if np.random.rand() < mutation_rate:
        x = x + np.random.normal(0, sigma)
        x = np.clip(x, low, high)
    return x

# 主遗传算法流程
def run_ga(pop_size=50, generations=100, crossover_prob=0.8, mutation_prob=0.02, tournament_k=3,
           low=0.0, high=10.0, seed=None, verbose=True):
    if seed is not None:
        np.random.seed(seed)

    pop = init_population(pop_size, low, high)
    best_history = []
    mean_history = []

    for gen in range(generations):
        fitness = objective(pop)
        best_idx = np.argmin(fitness)
        best_val = fitness[best_idx]
        mean_val = float(np.mean(fitness))
        best_history.append(best_val)
        mean_history.append(mean_val)

        if verbose and (gen % max(1, generations // 10) == 0 or gen == generations - 1):
            print(f"Gen {gen+1}/{generations}: best y={best_val:.6f} at x={pop[best_idx]:.6f}")

        # 新一代
        new_pop = []
        while len(new_pop) < pop_size:
            # 选择父母
            parent1 = tournament_selection(pop, fitness, k=tournament_k)
            parent2 = tournament_selection(pop, fitness, k=tournament_k)

            # 交叉
            if np.random.rand() < crossover_prob:
                child1, child2 = blend_crossover(parent1, parent2, alpha=0.5)
            else:
                child1, child2 = parent1, parent2

            # 变异
            child1 = mutate(child1, mutation_prob, sigma=0.1, low=low, high=high)
            child2 = mutate(child2, mutation_prob, sigma=0.1, low=low, high=high)

            new_pop.append(child1)
            if len(new_pop) < pop_size:
                new_pop.append(child2)

        pop = np.array(new_pop)

    # 最终评估
    final_fitness = objective(pop)
    best_idx = np.argmin(final_fitness)
    best_x = float(pop[best_idx])
    best_y = float(final_fitness[best_idx])

    results = {
        'best_x': best_x,
        'best_y': best_y,
        'best_history': best_history,
        'mean_history': mean_history,
        'final_population': pop,
    }
    return results


def save_results(results, params, out_prefix=None):
    if out_prefix is None:
        ts = time.strftime('%Y%m%d_%H%M%S')
        out_prefix = f'ga_experiment1_{ts}'

    img_path = out_prefix + '_results.png'
    log_path = out_prefix + '_log.txt'

    # 绘图：最优值随代数变化
    gens = np.arange(1, len(results['best_history']) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(gens, results['best_history'], label='best')
    plt.plot(gens, results['mean_history'], label='mean')
    plt.xlabel('Generation')
    plt.ylabel('Objective (y)')
    plt.title('GA optimization of y = 10*sin(5x) + 7|x-5| + 10')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(img_path, dpi=150)
    plt.close()

    # 日志
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write('Genetic Algorithm Experiment 1 Results\n')
        f.write('Parameters:\n')
        for k, v in params.items():
            f.write(f'  {k}: {v}\n')
        f.write('\nBest result:\n')
        f.write(f"  x = {results['best_x']:.8f}\n")
        f.write(f"  y = {results['best_y']:.8f}\n")

    return img_path, log_path


def parse_args():
    p = argparse.ArgumentParser(description='Genetic Algorithm - Experiment 1')
    p.add_argument('--pop_size', type=int, default=100, help='population size')
    p.add_argument('--generations', type=int, default=200, help='number of generations')
    p.add_argument('--crossover', type=float, default=0.8, help='crossover probability')
    p.add_argument('--mutation', type=float, default=0.02, help='mutation probability')
    p.add_argument('--tournament_k', type=int, default=3, help='tournament size')
    p.add_argument('--seed', type=int, default=None, help='random seed')
    p.add_argument('--no_verbose', action='store_true', help='disable verbose output')
    return p.parse_args()


def main():
    args = parse_args()
    params = {
        'pop_size': args.pop_size,
        'generations': args.generations,
        'crossover_prob': args.crossover,
        'mutation_prob': args.mutation,
        'tournament_k': args.tournament_k,
        'seed': args.seed,
    }

    print('Running GA experiment 1 with parameters:')
    for k, v in params.items():
        print(f'  {k}: {v}')

    results = run_ga(pop_size=args.pop_size,
                     generations=args.generations,
                     crossover_prob=args.crossover,
                     mutation_prob=args.mutation,
                     tournament_k=args.tournament_k,
                     seed=args.seed,
                     verbose=not args.no_verbose)

    img_path, log_path = save_results(results, params, out_prefix=os.path.join(os.getcwd(), 'ga_experiment1'))

    print('\nFinished. Best solution:')
    print(f"  x = {results['best_x']:.8f}")
    print(f"  y = {results['best_y']:.8f}")
    print(f"Saved plot to: {img_path}")
    print(f"Saved log to: {log_path}")


if __name__ == '__main__':
    main()
