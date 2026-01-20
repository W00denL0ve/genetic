# -*- coding: utf-8 -*-
"""
批量实验脚本：参数敏感性评估与算子对比

用法示例：
python benchmark.py --mode sensitivity --repeats 20
python benchmark.py --mode operators --repeats 30

说明：脚本会调用同目录下的 tsp_ga.run_ga，并将每次运行结果保存为 CSV，
并生成简单的统计汇总图（均值与方差）。

注意：每组配置默认会多次运行(repeats)以计算均值与标准差；总体运行可能耗时较长，建议在小规模上先测试。
"""

import argparse
import csv
import os
import time
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
import secrets

from tsp_ga import run_ga


def ensure_dirs():
    # 输出目录固定为相对于本文件的上级目录（experiment2）
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    figures_dir = os.path.join(base_dir, 'figures')
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    return figures_dir, results_dir, base_dir


def save_csv(rows, headers, path):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def plot_summary_by_param(rows, headers, param_name, out_path):
    # rows: list of lists, headers: list of column names
    try:
        import pandas as pd
    except Exception:
        raise RuntimeError('pandas required for plotting summary. Install with pip install pandas')
    df = pd.DataFrame(rows, columns=headers)
    grouped = df.groupby(param_name)['best_length'].agg(['mean','std']).reset_index()
    plt.figure(figsize=(6,4))
    plt.errorbar(range(len(grouped[param_name])), grouped['mean'], yerr=grouped['std'], fmt='-o')
    plt.xticks(range(len(grouped[param_name])), grouped[param_name].astype(str), rotation=45)
    plt.xlabel(param_name)
    plt.ylabel('best_length')
    plt.title(f'Benchmark summary by {param_name}')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def run_sensitivity(repeats=20):
    figures_dir, results_dir, base_dir = ensure_dirs()
    # 参数范围（示例，可修改）
    pop_sizes = [100, 200, 400]
    generations_list = [200, 500, 1000]
    crossover_probs = [0.7, 0.9]
    mutation_probs = [0.05, 0.1, 0.2]
    tournament_ks = [2, 3]
    elites_list = [1, 2]

    headers = ['pop_size','generations','crossover_prob','mutation_prob','tournament_k','elites','run_seed','best_length','time_sec']
    rows = []
    summary_records = []
    total_configs = len(pop_sizes)*len(generations_list)*len(crossover_probs)*len(mutation_probs)*len(tournament_ks)*len(elites_list)
    print(f'Total configs: {total_configs}, repeats per config: {repeats}')

    config_idx = 0
    for (pop_size, generations, cx, mu, k, elites) in product(pop_sizes, generations_list, crossover_probs, mutation_probs, tournament_ks, elites_list):
        config_idx += 1
        print(f'Config {config_idx}/{total_configs}: pop={pop_size}, gen={generations}, cx={cx}, mu={mu}, k={k}, e={elites}')
        bests = []
        times = []
        for r in range(repeats):
            # 生成合法的 32-bit 整数随机种子，确保在 [0, 2**32-1] 范围内
            # 使用 secrets.randbelow 避免 numpy.randint 在某些平台上对 int32 上界的限制
            seed = int(secrets.randbelow(2**32))
            start = time.time()
            res = run_ga(np.random.RandomState(int(seed)).random((20,2))*100.0,
                         pop_size=pop_size, generations=generations, crossover_prob=cx, mutation_prob=mu,
                         tournament_k=k, elites=elites, crossover_op='pmx', mutation_op='inversion',
                         seed=int(seed), out_dir=None, verbose=False)
            elapsed = time.time() - start
            bests.append(res['best_length'])
            times.append(elapsed)
            rows.append([pop_size,generations,cx,mu,k,elites,int(seed),res['best_length'],elapsed])
        summary_records.append({'pop_size':pop_size,'generations':generations,'crossover_prob':cx,'mutation_prob':mu,'tournament_k':k,'elites':elites,
                                'mean':float(np.mean(bests)),'std':float(np.std(bests)),'mean_time':float(np.mean(times))})
    fname = os.path.join(results_dir, f'benchmark_sensitivity.csv')
    save_csv(rows, headers, fname)
    print('Saved raw results to', fname)
    # 绘图：按 pop_size 做示例图
    plot_summary_by_param(rows, headers, 'pop_size', os.path.join(figures_dir, 'benchmark_by_pop.png'))
    return rows, summary_records


def run_operators(repeats=30):
    figures_dir, results_dir, base_dir = ensure_dirs()
    # 固定其他参数
    pop_size = 200
    generations = 500
    cx = 0.9
    mu = 0.1
    k = 3
    elites = 2

    crossover_ops = ['pmx','ox']
    mutation_ops = ['swap','inversion','insertion']

    headers = ['crossover_op','mutation_op','run_seed','best_length','time_sec']
    rows = []
    records = []
    total = len(crossover_ops)*len(mutation_ops)
    idx = 0
    for co in crossover_ops:
        for mo in mutation_ops:
            idx += 1
            print(f'Operator combo {idx}/{total}: {co} + {mo}')
            bests = []
            times = []
            for r in range(repeats):
                # 生成合法的 32-bit 整数随机种子，确保在 [0, 2**32-1] 范围内
                # 使用 secrets.randbelow 避免 numpy.randint 在某些平台上对 int32 上界的限制
                seed = int(secrets.randbelow(2**32))
                start = time.time()
                res = run_ga(np.random.RandomState(int(seed)).random((20,2))*100.0,
                             pop_size=pop_size, generations=generations, crossover_prob=cx, mutation_prob=mu,
                             tournament_k=k, elites=elites, crossover_op=co, mutation_op=mo,
                             seed=int(seed), out_dir=None, verbose=False)
                elapsed = time.time() - start
                bests.append(res['best_length'])
                times.append(elapsed)
                rows.append([co,mo,int(seed),res['best_length'],elapsed])
            records.append({'crossover_op':co,'mutation_op':mo,'mean':float(np.mean(bests)),'std':float(np.std(bests)),'mean_time':float(np.mean(times))})
    fname = os.path.join(results_dir, f'benchmark_operators.csv')
    save_csv(rows, headers, fname)
    print('Saved raw results to', fname)
    # 绘图示例：箱型图比较
    import pandas as pd
    df = pd.DataFrame(rows, columns=headers)
    plt.figure(figsize=(8,4))
    df.boxplot(by=['crossover_op','mutation_op'], column=['best_length'], rot=45)
    plt.title('Operator comparison (best_length)')
    plt.suptitle('')
    plt.tight_layout()
    out_png = os.path.join(figures_dir, 'operators_boxplot.png')
    plt.savefig(out_png, dpi=150)
    plt.close()
    return rows, records


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['sensitivity','operators'], required=True)
    p.add_argument('--repeats', type=int, default=20)
    return p.parse_args()


def main():
    args = parse_args()
    if args.mode == 'sensitivity':
        run_sensitivity(repeats=args.repeats)
    else:
        run_operators(repeats=args.repeats)


if __name__ == '__main__':
    main()
