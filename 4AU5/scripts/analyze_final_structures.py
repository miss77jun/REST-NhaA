#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析REST2模拟的最终结构
使用MDAnalysis进行结构分析和可视化
"""

import os
import sys
import numpy as np

# 检查是否安装了必要的包
try:
    import MDAnalysis as mda
    from MDAnalysis.analysis import rms, contacts
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
except ImportError as e:
    print(f"需要安装依赖: {e}")
    print("请运行: pip install MDAnalysis matplotlib")
    sys.exit(1)

def analyze_final_structures(experiment_dir):
    """分析实验目录中的最终结构"""
    
    # 查找所有最终结构文件
    pdb_files = sorted([f for f in os.listdir(experiment_dir) 
                       if f.startswith('replica_') and f.endswith('_final.pdb')])
    
    if not pdb_files:
        print(f"在 {experiment_dir} 中没有找到最终结构文件")
        return
    
    print(f"找到 {len(pdb_files)} 个最终结构文件")
    
    # 使用第一个副本作为参考
    ref_pdb = os.path.join(experiment_dir, pdb_files[0])
    ref_u = mda.Universe(ref_pdb)
    
    # 选择蛋白质原子
    protein = ref_u.select_atoms('protein')
    print(f"蛋白质包含 {len(protein)} 个原子")
    
    # RMSD分析
    rmsd_values = []
    replica_ids = []
    
    print("\n分析各副本相对于最低温度副本的RMSD:")
    print("Replica   RMSD (Å)")
    print("-" * 20)
    
    for pdb_file in pdb_files:
        replica_id = int(pdb_file.split('_')[1])
        replica_ids.append(replica_id)
        
        u = mda.Universe(os.path.join(experiment_dir, pdb_file))
        protein_mobile = u.select_atoms('protein')
        
        # 计算RMSD
        rmsd = rms.rmsd(protein_mobile.positions, protein.positions, 
                       superposition=True, center=True)
        rmsd_values.append(rmsd)
        
        print(f"  {replica_id:2d}      {rmsd:6.2f}")
    
    # 绘制RMSD图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 子图1: RMSD vs 副本编号
    ax = axes[0]
    ax.bar(replica_ids, rmsd_values, color='steelblue')
    ax.set_xlabel('Replica ID')
    ax.set_ylabel('RMSD (Å)')
    ax.set_title('RMSD relative to Replica 0 (300K)')
    ax.grid(alpha=0.3)
    
    # 子图2: 最终结构对比（选择几个代表性副本）
    ax = axes[1]
    selected_replicas = [0, 7, 15, 23, 31] if len(replica_ids) > 16 else [0, len(replica_ids)//2, len(replica_ids)-1]
    selected_rmsd = [rmsd_values[replica_ids.index(i)] for i in selected_replicas if i in replica_ids]
    selected_ids = [i for i in selected_replicas if i in replica_ids]
    
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(selected_ids)))
    bars = ax.bar(range(len(selected_ids)), selected_rmsd, color=colors)
    ax.set_xticks(range(len(selected_ids)))
    ax.set_xticklabels([f'Rep {i}' for i in selected_ids])
    ax.set_ylabel('RMSD (Å)')
    ax.set_title('Selected Replicas RMSD')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    output_file = os.path.join(experiment_dir, 'figures', 'rmsd_analysis.png')
    plt.savefig(output_file, dpi=150)
    plt.close()
    
    print(f"\nRMSD分析图已保存到: {output_file}")
    
    # 分析热残基位置变化
    print("\n分析热残基位置变化:")
    hot_residues = ['72', '167', '168', '344', '345']
    
    # 选择第一个和最后一个副本
    u_low = mda.Universe(os.path.join(experiment_dir, pdb_files[0]))
    u_high = mda.Universe(os.path.join(experiment_dir, pdb_files[-1]))
    
    for res_id in hot_residues:
        try:
            res_low = u_low.select_atoms(f'resid {res_id} and protein')
            res_high = u_high.select_atoms(f'resid {res_id} and protein')
            
            if len(res_low) > 0 and len(res_high) > 0:
                # 计算该残基的RMSD
                rmsd_res = rms.rmsd(res_high.positions, res_low.positions, 
                                   superposition=True, center=True)
                print(f"  残基 {res_id}: RMSD = {rmsd_res:.2f} Å")
        except Exception as e:
            print(f"  残基 {res_id}: 无法计算 ({e})")
    
    return rmsd_values, replica_ids


def create_trajectory_animation(experiment_dir):
    """
    如果有检查点文件，尝试创建轨迹动画
    注意：当前版本只保存了能量，没有保存轨迹
    """
    checkpoint_file = os.path.join(experiment_dir, 'rest2_results.npz')
    
    if not os.path.exists(checkpoint_file):
        print("没有找到轨迹数据文件")
        return
    
    data = np.load(checkpoint_file)
    energies = data['energies']
    temps = data['temps']
    
    print(f"\n能量数据形状: {energies.shape}")
    print(f"温度范围: {temps[0]:.1f}K - {temps[-1]:.1f}K")
    
    # 绘制能量随迭代变化的图
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i in range(min(8, len(temps))):
        ax.plot(energies[:, i], label=f'Replica {i} ({temps[i]:.0f}K)', 
               alpha=0.7, lw=0.8)
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Potential Energy (kJ/mol)')
    ax.set_title('Energy Evolution During REST2 Simulation')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    
    output_file = os.path.join(experiment_dir, 'figures', 'energy_trajectory.png')
    plt.savefig(output_file, dpi=150)
    plt.close()
    
    print(f"能量轨迹图已保存到: {output_file}")


def main():
    # 获取最新的实验目录
    outputs_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    experiment_dirs = sorted([d for d in os.listdir(outputs_dir) 
                            if d.startswith('experiment_')], reverse=True)
    
    if not experiment_dirs:
        print("没有找到实验目录")
        return
    
    latest_experiment = os.path.join(outputs_dir, experiment_dirs[0])
    print(f"分析实验: {experiment_dirs[0]}")
    print("=" * 60)
    
    # 分析最终结构
    analyze_final_structures(latest_experiment)
    
    # 创建能量轨迹图
    create_trajectory_animation(latest_experiment)
    
    print("\n分析完成!")


if __name__ == "__main__":
    main()