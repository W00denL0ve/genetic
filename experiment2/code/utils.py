# -*- coding: utf-8 -*-
"""
辅助工具：保存/加载数据、绘图辅助等
"""
import numpy as np
import matplotlib.pyplot as plt
import os


def save_city_map(city_map: np.ndarray, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savetxt(path, city_map, delimiter=',')


def load_city_map(path: str) -> np.ndarray:
    return np.loadtxt(path, delimiter=',')


def plot_path_simple(city_map: np.ndarray, path_idx: np.ndarray, out_path: str):
    coords = city_map[path_idx]
    coords = np.vstack([coords, coords[0]])
    plt.figure(figsize=(6,6))
    plt.plot(coords[:,0], coords[:,1], '-o')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
