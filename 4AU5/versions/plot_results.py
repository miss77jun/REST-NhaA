#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
REST2实验结果可视化后处理脚本

使用已保存的rest2_results.npz数据生成科学图表

用法:
    # 在服务器上(安装了matplotlib后)
    python plot_results.py --input /path/to/outputs/experiment_xxx/rest2_results.npz
    
    # 在本地(把数据下载后)
    python plot_results.py --input /local/path/to/rest2_results.npz
"""

import argparse
import numpy as np
import json
import os

def parse_args():
    parser = argparse.ArgumentParser(description='REST2 Results Visualization')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to rest2_results.npz file')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for figures')
    return parser.parse_args()

def load_results(npz_file):
    """加载rest2_results.npz数据"""
    data = np.load(npz_file, allow_pickle=True)
    
    results = {
        'energies': data['energies'],
        'exchange_attempts': data['attempts'],
        'exchange_accepts': data['accepts'],
        'temperatures': data['temps'],
        'lam_values': data['lams'],
        'total_time': float(data['total_time']),
        'version': str(data['version']),
        'lambda_hot': float(data['lambda_hot']),
        'periodic_boundary': bool(data['periodic_boundary']),
        'cubic_box': bool(data['cubic_box']),
        'box_size': float(data['box_size']),
        'delta_history': {},
    }
    
    # 加载delta_history
    for key in data.keys():
        if key.startswith('delta_'):
            idx = int(key.split('_')[1])
            results['delta_history'][idx] = list(data[key])
    
    # 加载box_dims
    if 'box_dims' in data:
        results['box_dims'] = list(data['box_dims'])
    else:
        results['box_dims'] = [82.407, 82.407, 109.362]
    
    return results

def generate_scientific_visualizations(results, output_dir):
    """生成科学可视化图表"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    E = results['energies']
    attempts = results['exchange_attempts']
    accepts = results['exchange_accepts']
    T = results['temperatures']
    deltas = results['delta_history']
    
    n_rep = len(T)
    rates = [accepts[k]/attempts[k]*100 if attempts[k] > 0 else 0 for k in range(n_rep-1)]
    avg_rate = np.mean(rates)
    
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # (a) 势能
    ax = axes[0,0]
    for i in range(min(8, n_rep)):
        ax.plot(E[:,i], label=f'Rep{i} ({T[i]:.0f}K)', alpha=0.7, lw=0.8)
    ax.set_xlabel('Iteration'); ax.set_ylabel('E (kJ/mol)')
    ax.set_title('(a) Potential Energy', fontweight='bold')
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
    
    # (b) 交换接受率
    ax = axes[0,1]
    colors = ['#27ae60' if 20<=r<=40 else '#e74c3c' for r in rates]
    ax.bar(range(n_rep-1), rates, color=colors)
    ax.axhline(20, color='k', ls='--'); ax.axhline(40, color='k', ls='--')
    ax.set_xlabel('Replica Pair'); ax.set_ylabel('Exchange Rate (%)')
    ax.set_title('(b) Exchange Acceptance', fontweight='bold')
    ax.grid(alpha=0.3)
    
    # (c) Delta分布
    ax = axes[0,2]
    for k in range(min(8, n_rep-1)):
        if k in deltas and deltas[k]:
            ax.hist(deltas[k], bins=30, alpha=0.5, label=f'Pair {k}-{k+1}', density=True)
    ax.axvline(0, color='k'); ax.set_xlabel('Delta'); ax.set_ylabel('P(Delta)')
    ax.set_title('(c) Delta Distribution', fontweight='bold')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    
    # (d) 能量-温度关系
    ax = axes[1,0]
    E_mean = E.mean(axis=0); E_std = E.std(axis=0)
    ax.errorbar(T, E_mean, yerr=E_std, fmt='o-', capsize=3)
    ax.set_xlabel('T (K)'); ax.set_ylabel('Average E (kJ/mol)')
    ax.set_title('(d) Energy vs Temperature', fontweight='bold')
    ax.grid(alpha=0.3)
    
    # (e) 收敛性
    ax = axes[1,1]
    for k in range(min(8, n_rep-1)):
        if k in deltas and deltas[k]:
            cum = np.cumsum([d<0 for d in deltas[k]])/np.arange(1,len(deltas[k])+1)*100
            ax.plot(cum, label=f'Pair {k}-{k+1}', alpha=0.7)
    ax.axhline(avg_rate, color='k', ls='--', label=f'Avg: {avg_rate:.1f}%')
    ax.set_xlabel('Exchange Attempt'); ax.set_ylabel('Cumulative Rate (%)')
    ax.set_title('(e) Convergence', fontweight='bold')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    
    # (f) Lambda缩放
    ax = axes[1,2]
    lams = results['lam_values']
    ax.plot(T, lams, 'o-', label='λ_global', color='blue')
    ax.set_xlabel('T (K)'); ax.set_ylabel('Lambda')
    ax.set_title('(f) Lambda Scaling', fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'fig1_analysis.png'), dpi=150)
    plt.close()
    
    print(f"  ✓ 科学分析图保存: {os.path.join(figures_dir, 'fig1_analysis.png')}")

def generate_summary_report(results, output_dir):
    """生成汇总报告"""
    E = results['energies']
    attempts = results['exchange_attempts']
    accepts = results['exchange_accepts']
    T = results['temperatures']
    
    n_rep = len(T)
    rates = [accepts[k]/attempts[k]*100 if attempts[k] > 0 else 0 for k in range(n_rep-1)]
    avg_rate = np.mean(rates)
    
    report = {
        'version': results['version'],
        'n_replicas': int(n_rep),
        'temperatures_K': [float(t) for t in T],
        'avg_exchange_rate': float(round(avg_rate, 2)),
        'n_iterations': int(len(E)),
        'total_time_s': float(round(results['total_time'], 1)),
        'periodic_boundary': results['periodic_boundary'],
        'cubic_box': results['cubic_box'],
        'box_dimensions_angstrom': [float(d) for d in results.get('box_dims', [82.407, 82.407, 109.362])],
        'box_size_max_angstrom': float(results['box_size']),
        'lambda_hot': results['lambda_hot'],
        'avg_energy_kJ/mol': {
            'mean': float(np.mean(E)),
            'std': float(np.std(E)),
        },
        'exchange_rates_%': [float(r) for r in rates],
    }
    
    report_file = os.path.join(output_dir, 'scientific_report.json')
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"  ✓ 科学报告保存: {report_file}")
    
    # 打印摘要
    print(f"\n{'='*60}")
    print(f"REST2实验结果摘要")
    print(f"{'='*60}")
    print(f"版本: {results['version']}")
    print(f"副本数: {n_rep}")
    print(f"温度范围: {T[0]:.0f}-{T[-1]:.0f}K")
    print(f"迭代次数: {len(E)}")
    print(f"总模拟时间: {results['total_time']/3600:.1f}小时")
    print(f"平均交换率: {avg_rate:.1f}%")
    print(f"平均势能: {np.mean(E):.0f} ± {np.std(E):.0f} kJ/mol")
    print(f"周期性边界: {'启用' if results['periodic_boundary'] else '未启用'}")
    print(f"盒子尺寸: {results['box_dims'][0]:.2f} × {results['box_dims'][1]:.2f} × {results['box_dims'][2]:.2f} Å")
    print(f"{'='*60}")

def main():
    args = parse_args()
    
    input_file = args.input
    if not os.path.exists(input_file):
        print(f"错误: 文件不存在 - {input_file}")
        exit(1)
    
    if args.output:
        output_dir = args.output
    else:
        output_dir = os.path.dirname(input_file)
    
    print(f"加载结果数据: {input_file}")
    print(f"输出目录: {output_dir}")
    
    results = load_results(input_file)
    print(f"数据加载成功!")
    
    print(f"\n=== 生成科学可视化图表 ===")
    generate_scientific_visualizations(results, output_dir)
    
    print(f"\n=== 生成汇总报告 ===")
    generate_summary_report(results, output_dir)
    
    print(f"\n✓ 所有图表和报告已生成完成!")

if __name__ == "__main__":
    main()
