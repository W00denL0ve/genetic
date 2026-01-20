# -*- coding: utf-8 -*-
"""
遗传算法求解 TSP（20个城市）
Python = 3.10
可运行：
python tsp_ga.py --pop_size 200 --generations 500 --crossover 0.9 --mutation 0.2 --seed 42

输出：在上级目录的 experiment2/figures/ 和 experiment2/results/ 生成图片与日志。
"""

import argparse
import os
import time
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import textwrap

# ---------------------- 辅助函数 ----------------------

def init_population(pop_size: int, n_cities: int) -> np.ndarray:
    """返回 shape (pop_size, n_cities) 的整数排列数组"""
    pop = np.empty((pop_size, n_cities), dtype=int)
    for i in range(pop_size):
        pop[i] = np.random.permutation(n_cities)
    return pop


def compute_distance_matrix(city_map: np.ndarray) -> np.ndarray:
    n = city_map.shape[0]
    Dist = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(city_map[i] - city_map[j])
            Dist[i, j] = d
            Dist[j, i] = d
    return Dist


def path_length(path: np.ndarray, Dist: np.ndarray) -> float:
    # path: sequence of city indices
    n = len(path)
    total = 0.0
    for i in range(n - 1):
        total += Dist[path[i], path[i + 1]]
    total += Dist[path[-1], path[0]]
    return total


def evaluate_population(population: np.ndarray, Dist: np.ndarray) -> np.ndarray:
    pop_size = population.shape[0]
    fitness = np.empty(pop_size, dtype=float)
    for i in range(pop_size):
        fitness[i] = path_length(population[i], Dist)
    return fitness


def tournament_selection(population: np.ndarray, fitness: np.ndarray, k: int = 3) -> np.ndarray:
    idx = np.random.randint(0, population.shape[0], size=k)
    best = idx[0]
    for j in idx[1:]:
        if fitness[j] < fitness[best]:
            best = j
    return population[best].copy()


def pmx_crossover(p1: np.ndarray, p2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """可靠的 PMX 实现：返回 (child1, child2, cut1, cut2)
    采用纯 Python 列表索引以避免 numpy.where 导致的数组到标量转换问题。
    """
    n = len(p1)
    p1_list = p1.tolist()
    p2_list = p2.tolist()
    # 随机交叉点
    a, b = sorted(np.random.choice(range(n), size=2, replace=False))

    # 初始化子代
    c1 = [-1] * n
    c2 = [-1] * n
    # 复制交叉区间
    for i in range(a, b + 1):
        c1[i] = p1_list[i]
        c2[i] = p2_list[i]

    # 填充函数：将 parent2 中在交叉区间的元素放到 c1 中相应的位置
    def fill_child(child, donor, other):
        # donor == parent2, other == parent1 when filling child1
        for i in range(a, b + 1):
            val = donor[i]
            if val not in child:
                pos = i
                # 寻找映射位置，循环直到找到空位
                while True:
                    mapped = other[pos]
                    pos = donor.index(mapped)
                    if child[pos] == -1:
                        child[pos] = val
                        break
        # 填充剩余位置
        for i in range(n):
            if child[i] == -1:
                for v in other:
                    if v not in child:
                        child[i] = v
                        break

    fill_child(c1, p2_list, p1_list)
    fill_child(c2, p1_list, p2_list)

    return np.array(c1, dtype=int), np.array(c2, dtype=int), a, b


def order_crossover(p1: np.ndarray, p2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = len(p1)
    i, j = sorted(np.random.choice(n, size=2, replace=False))
    def ox(a, b):
        child = -np.ones(n, dtype=int)
        child[i:j+1] = a[i:j+1]
        fill_idx = (j + 1) % n
        b_idx = (j + 1) % n
        while np.any(child == -1):
            if b[b_idx] not in child:
                child[fill_idx] = b[b_idx]
                fill_idx = (fill_idx + 1) % n
            b_idx = (b_idx + 1) % n
        return child
    return ox(p1, p2), ox(p2, p1)


def mutation_swap(individual: np.ndarray) -> np.ndarray:
    a, b = np.random.choice(len(individual), size=2, replace=False)
    individual = individual.copy()
    individual[a], individual[b] = individual[b], individual[a]
    return individual


def mutation_inversion(individual: np.ndarray) -> np.ndarray:
    a, b = sorted(np.random.choice(len(individual), size=2, replace=False))
    ind = individual.copy()
    ind[a:b+1] = ind[a:b+1][::-1]
    return ind


def apply_elitism(population: np.ndarray, fitness: np.ndarray, elites: int) -> np.ndarray:
    if elites <= 0:
        return np.empty((0, population.shape[1]), dtype=int)
    idx = np.argsort(fitness)
    return population[idx[:elites]].copy()


def plot_convergence(history_best: list, history_mean: list, out_path: str, params: dict = None):
    gens = np.arange(1, len(history_best) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(gens, history_best, label='best')
    plt.plot(gens, history_mean, label='mean')
    plt.xlabel('Generation')
    plt.ylabel('Path length')
    # 构建参数摘要：短标题 + 底部完整参数（换行显示）
    short = ''
    wrapped = ''
    if params:
        def fmt(v):
            return f"{v:.3f}" if isinstance(v, float) else str(v)
        # 选取关键参数用于短标题
        short_keys = ['pop', 'gen', 'cx', 'mu', 'k', 'elites']
        short_pairs = [f"{k}={fmt(params[k])}" for k in short_keys if k in params]
        if short_pairs:
            short = ' (' + ', '.join(short_pairs) + ')'
        # 完整参数以逗号分隔并换行
        full_pairs = [f"{k}={fmt(v)}" for k, v in params.items()]
        wrapped = textwrap.fill(', '.join(full_pairs), width=80)
    plt.title('Convergence' + short)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # 为底部 figtext 留出空间
    plt.subplots_adjust(bottom=0.20)
    if wrapped:
        plt.figtext(0.01, 0.01, wrapped, ha='left', fontsize=8)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_path(city_map: np.ndarray, best_path: np.ndarray, out_path: str, params: dict = None):
    coords = city_map[best_path]
    coords = np.vstack([coords, coords[0]])  # close loop
    plt.figure(figsize=(6, 6))
    plt.plot(coords[:, 0], coords[:, 1], '-o')
    for i, (x, y) in enumerate(city_map):
        plt.text(x, y, str(i), fontsize=8)
    # 简短参数摘要 + 底部完整参数
    short = ''
    wrapped = ''
    if params:
        def fmt(v):
            return f"{v:.3f}" if isinstance(v, float) else str(v)
        short_keys = ['pop', 'gen', 'cx', 'mu', 'k', 'elites']
        short_pairs = [f"{k}={fmt(params[k])}" for k in short_keys if k in params]
        if short_pairs:
            short = ' (' + ', '.join(short_pairs) + ')'
        full_pairs = [f"{k}={fmt(v)}" for k, v in params.items()]
        wrapped = textwrap.fill(', '.join(full_pairs), width=80)
    plt.title('Best path' + short)
    plt.axis('equal')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    if wrapped:
        plt.figtext(0.01, 0.01, wrapped, ha='left', fontsize=8)
    plt.savefig(out_path, dpi=150)
    plt.close()


def ensure_valid_permutation(individual: np.ndarray, n: int) -> np.ndarray:
    """检查并修复非法染色体（若包含重复或缺失），返回合法置换。"""
    if len(individual) != n or len(set(individual.tolist())) != n:
        # 若非法，返回一个随机合法置换以替代
        return np.random.permutation(n)
    return individual


# ---------------------- 主流程 ----------------------

def run_ga(city_map: np.ndarray, pop_size=100, generations=300, crossover_prob=0.9,
           mutation_prob=0.2, tournament_k=3, elites=2, crossover_op='pmx', mutation_op='swap',
           seed=None, out_dir=None, verbose=True):
    if seed is not None:
        np.random.seed(seed)

    n = city_map.shape[0]
    Dist = compute_distance_matrix(city_map)

    # 输出目录固定为相对于本文件的上级目录（退回到 experiment2），再访问 figures 和 results
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    figures_dir = os.path.join(base_dir, 'figures')
    results_dir = os.path.join(base_dir, 'results')
    debug_dir = os.path.join(results_dir, 'debug')
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)

    population = init_population(pop_size, n)
    fitness = evaluate_population(population, Dist)

    history_best = []
    history_mean = []

    for gen in range(generations):
        best_idx = int(np.argmin(fitness))
        history_best.append(float(fitness[best_idx]))
        history_mean.append(float(np.mean(fitness)))
        if verbose and (gen % max(1, generations // 10) == 0 or gen == generations - 1):
            print(f"Gen {gen+1}/{generations}: best={fitness[best_idx]:.4f}")

        elites_ind = apply_elitism(population, fitness, elites)
        new_pop = []
        while len(new_pop) < pop_size - elites:
            p1 = tournament_selection(population, fitness, k=tournament_k)
            p2 = tournament_selection(population, fitness, k=tournament_k)
            # 确保父代为合法置换，若不合法则修复为随机置换
            p1 = ensure_valid_permutation(p1, n)
            p2 = ensure_valid_permutation(p2, n)

            if np.random.rand() < crossover_prob:
                if crossover_op == 'pmx':
                    c1, c2, a, b = pmx_crossover(p1, p2)
                else:
                    c1, c2 = order_crossover(p1, p2)
                    a = b = None
            else:
                c1, c2 = p1.copy(), p2.copy()
                a = b = None

            # 交叉后合法性检查：若子代非法则记录调试日志并使用回退方案
            def is_valid_perm(arr):
                return isinstance(arr, np.ndarray) and set(arr.tolist()) == set(range(n))

            if not is_valid_perm(c1) or not is_valid_perm(c2):
                ts = time.strftime('%Y%m%d_%H%M%S')
                debug_path = os.path.join(debug_dir, f'debug_gen{gen+1}_{ts}.txt')
                with open(debug_path, 'w', encoding='utf-8') as df:
                    df.write(f'Generation: {gen+1}\n')
                    df.write(f'Parents:\n')
                    df.write('p1: ' + ' '.join(map(str, p1.tolist())) + '\n')
                    df.write('p2: ' + ' '.join(map(str, p2.tolist())) + '\n')
                    df.write(f'crossover_op: {crossover_op}, cut_points: {a},{b}\n')
                    df.write('Children (raw):\n')
                    df.write('c1: ' + ( ' '.join(map(str, c1.tolist())) if isinstance(c1, np.ndarray) else str(c1)) + '\n')
                    df.write('c2: ' + ( ' '.join(map(str, c2.tolist())) if isinstance(c2, np.ndarray) else str(c2)) + '\n')
                # 回退方案：尝试 OX 交叉，否则随机置换
                try:
                    c1f, c2f = order_crossover(p1, p2)
                except Exception:
                    c1f = np.random.permutation(n)
                    c2f = np.random.permutation(n)
                c1, c2 = c1f, c2f

            if np.random.rand() < mutation_prob:
                c1 = mutation_swap(c1) if mutation_op == 'swap' else mutation_inversion(c1)
            if np.random.rand() < mutation_prob and len(new_pop) + 1 < pop_size - elites:
                c2 = mutation_swap(c2) if mutation_op == 'swap' else mutation_inversion(c2)
            new_pop.append(c1)
            if len(new_pop) < pop_size - elites:
                new_pop.append(c2)

        population = np.vstack([elites_ind] + new_pop) if elites > 0 else np.array(new_pop)
        fitness = evaluate_population(population, Dist)

    # 结束
    best_idx = int(np.argmin(fitness))
    best_path = population[best_idx]
    best_len = float(fitness[best_idx])

    # 输出保存
    # figures_dir, results_dir 已在上方创建
    # 传递关键参数到绘图标题中
    plot_params = {
        'pop': pop_size,
        'gen': generations,
        'cx': crossover_prob,
        'mu': mutation_prob,
        'k': tournament_k,
        'elites': elites,
        'crossover_op': crossover_op,
        'mutation_op': mutation_op,
        'seed': seed,
    }

    # 生成基于关键参数的文件名（不含时间戳），并替换小数点以保持文件名安全
    param_str = f"pop{pop_size}_gen{generations}_cx{crossover_prob:.2f}_mu{mutation_prob:.2f}_k{tournament_k}_e{elites}_op{crossover_op}-{mutation_op}_s{seed if seed is not None else 'NA'}"
    param_str = param_str.replace('.', 'p')

    conv_path = os.path.join(figures_dir, f'convergence_{param_str}.png')
    path_path = os.path.join(figures_dir, f'best_path_{param_str}.png')
    log_path = os.path.join(results_dir, f'result_{param_str}.txt')

    plot_convergence(history_best, history_mean, conv_path, params=plot_params)
    plot_path(city_map, best_path, path_path, params=plot_params)

    with open(log_path, 'w', encoding='utf-8') as f:
        f.write('TSP GA Result\n')
        f.write(f'pop_size: {pop_size}\n')
        f.write(f'generations: {generations}\n')
        f.write(f'crossover_prob: {crossover_prob}\n')
        f.write(f'mutation_prob: {mutation_prob}\n')
        f.write(f'tournament_k: {tournament_k}\n')
        f.write(f'elites: {elites}\n')
        f.write(f'crossover_op: {crossover_op}\n')
        f.write(f'mutation_op: {mutation_op}\n')
        f.write(f'seed: {seed}\n')
        f.write(f'best_length: {best_len:.6f}\n')
        f.write('best_path:\n')
        f.write(' '.join(map(str, best_path.tolist())) + '\n')

    return {
        'best_path': best_path,
        'best_length': best_len,
        'convergence_image': conv_path,
        'path_image': path_path,
        'log': log_path,
    }


def parse_args():
    p = argparse.ArgumentParser(description='TSP GA')
    p.add_argument('--n_cities', type=int, default=20)
    p.add_argument('--pop_size', type=int, default=200)
    p.add_argument('--generations', type=int, default=500)
    p.add_argument('--crossover', type=float, default=0.9)
    p.add_argument('--mutation', type=float, default=0.2)
    p.add_argument('--tournament_k', type=int, default=3)
    p.add_argument('--elites', type=int, default=2)
    p.add_argument('--crossover_op', choices=['pmx', 'ox'], default='pmx')
    p.add_argument('--mutation_op', choices=['swap', 'inversion'], default='inversion')
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--out_dir', type=str, default=os.path.join(os.getcwd(), '..', 'experiment2'))
    return p.parse_args()


def main():
    args = parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)
    city_map = 100.0 * np.random.rand(args.n_cities, 2)
    res = run_ga(city_map, pop_size=args.pop_size, generations=args.generations,
                 crossover_prob=args.crossover, mutation_prob=args.mutation,
                 tournament_k=args.tournament_k, elites=args.elites,
                 crossover_op=args.crossover_op, mutation_op=args.mutation_op,
                 seed=args.seed, out_dir=os.path.dirname(args.out_dir), verbose=True)

    print('\nFinished. Best length: {:.6f}'.format(res['best_length']))
    print('Convergence image:', res['convergence_image'])
    print('Path image:', res['path_image'])
    print('Log:', res['log'])


if __name__ == '__main__':
    main()
