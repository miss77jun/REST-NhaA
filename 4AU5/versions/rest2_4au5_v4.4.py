#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
REST2 (Replica Exchange with Solute Tempering v2) 分子动力学模拟 — v4.4
4AU5 蛋白 + CHARMM36m + TIP3P + 0.15M NaCl

======================================================================
版本: v4.4 (2026-09-20)

v4.4 五残基 non-standard local-hot REST2 扩展与续算一致性修复:
  - 默认额外加热 72 PHE、167 ALA、168 ILE、344 PHE、345 ILE
  - 所有可缩放 Hamiltonian 项采用 endpoint_anchored_geometric_atom_scale_v2 原子尺度几何平均
  - 修复 DCD 时间元数据、交换后的速度处理及科学报告时间计算
  - 增加统一随机种子、NumPy RNG 状态续存及 OpenMM 独立派生种子
  - 完整 checkpoint 增加 Hamiltonian/输入文件哈希并严格验证 v4.4 续算
  - 优化 make-whole 邻接表/BFS，并移除 HTTP 调试和不可恢复 checkpoint

以下为 v4.3 历史说明 (2026-09-18):
v4.3 walker 追踪与采样质量诊断增强:
  - 完整保留 v4.2 的标准 REST2 算法、PBC/DCD 修复和正式生产参数
  - 明确 walker 标签语义并修复 checkpoint 续算，新增可靠 round-trip 分析
  - 最终报告全部相邻交换率、质量提示和 round-trip 摘要

以下为 v4.2 历史说明 (2026-09-14):
  - 完整保留 v4.1 修复后的标准 REST2 算法，非标准局部加热默认关闭
  - 16 副本，300–350 K 几何有效温标
  - 每副本 50 ns 生产采样，每 2 ps 尝试交换，每 10 ps 保存轨迹
  - 共同盒膜 NPT 平衡 5 ns，各副本在各自 Hamiltonian 下以 300 K NVT 平衡 0.5 ns
  - NPT/NVT 平衡默认施加蛋白 CA 位置约束；释放后再无约束平衡 0.5 ns
  - 每 1 ns 保存完整 checkpoint，RMSD 每 100 ps 监控一次
  - 输出全部相邻交换率及 min/mean/max，并保存 walker 标签历史

以下为 v4.1 历史说明 (2026-09-11):
  - 修复并采用标准 REST2 交换算法与生产流程

以下为 v4.0 历史说明 (2026-07-30):
  [关键修复] 局部加热强度
  1. 历史版本曾将 lambda_hot 默认值从 0.5 提升到 0.75
     - 原因: lambda_hot=0.5 在 T_max=350K 时等效温度达 700K,
       导致蛋白 RMSD 偏离 ~5Å (结构过度变形)
     - 新值 0.75 使最高等效温度降至 ~467K, 在增强采样与结构稳定间取平衡
     - 添加 --lambda_hot CLI 参数, 可灵活调整
     - 添加安全检查: lambda_hot < 0.5 时发出警告

  [新增] 残基编号映射
  2. 记录并输出残基编号映射关系
     - OpenMM 加载 PDB 时保留原始残基 ID (10-390)
     - 但写出 PDB 时自动重编号为 1-381
     - v4.0 在日志、config.json、final PDB REMARK 中记录偏移量
     - 便于后续 VMD 分析时正确选择残基

  [新增] RMSD 实时监控
  3. 模拟过程中每 100 次迭代计算蛋白 CA 原子 RMSD
     - 与初始结构对比 (Kabsch 对齐后)
     - 记录到 rmsd_history.npy 和日志
     - 可视化为 fig_rmsd_evolution.png
     - 便于判断蛋白是否过度变形

  [优化] make_whole 性能
  4. 向量化 make_whole_positions 算法
     - 仅对大分子 (蛋白、脂质) 执行 make_whole
     - 水分子和离子跳过 (它们是单残基, 不存在跨边界问题)
     - 性能提升约 3-5 倍

  [新增] 可选位置约束平衡
  5. 添加 --equilibration_restraints 选项
     - 在平衡阶段对蛋白骨架施加弱位置约束 (100 kJ/mol/nm²)
     - 防止平衡阶段蛋白过度漂移
     - 生产阶段自动释放约束

  [改进] CLI 参数
  6. 新增命令行参数:
     --lambda_hot              局部加热 lambda 端点倍率 (当前默认 0.80)
     --no_local_heating        禁用局部加热
     --equilibration_restraints 在平衡阶段启用位置约束

  [改进] 日志与输出
  7. 增强日志可读性
     - 记录残基映射、RMSD 演化、有效温度
     - final PDB 添加残基映射 REMARK 记录
     - scientific_report.json 添加 RMSD 统计

v3.3 更新内容 (2026-07-28):
  1. Force-switch vdW (r_on=1.0nm, r_off=1.2nm)
  2. LangevinMiddleIntegrator
  3. 能量最小化步骤
  4. 离子检测功能
  5. 力场详细日志
  6. 周期性盒子向量和CRYST1记录

参考文献:
  Wang et al., PNAS 108(37): 15107-15112 (2011)  [REST2原始论文]
  Liu et al., JCP 139: 034115 (2013)             [局部REST/局部加热]
======================================================================
"""

__version__ = "4.4.1"
__date__ = "2026-09-20"

import os
os.environ['CUDA_CACHE_PATH'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cuda_cache')
os.environ['CUDA_CACHE_MAXSIZE'] = '2147483648'
os.makedirs(os.environ['CUDA_CACHE_PATH'], exist_ok=True)

import sys, time, copy, logging, datetime, traceback, json, hashlib, argparse
from collections import deque
import numpy as np
import openmm
from openmm import unit
from openmm import app

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(SCRIPT_DIR) in ('versions', 'scripts'):
    SCRIPT_DIR = os.path.dirname(SCRIPT_DIR)
CHARMMGUI_DIR = os.path.join(SCRIPT_DIR, "charmm-gui", "4au5-aa-k300n")
OPENMM_DIR = os.path.join(CHARMMGUI_DIR, "openmm")

CONFIG = {
    "psf_file":     os.path.join(OPENMM_DIR, "step5_charmm2omm.psf"),
    "pdb_file":     os.path.join(OPENMM_DIR, "step5_charmm2omm.pdb"),
    "toppar_str":   os.path.join(CHARMMGUI_DIR, "toppar.str"),
    "boxinfo_str":  os.path.join(OPENMM_DIR, "step5_charmm2omm.str"),
    "base_output_dir": os.path.join(SCRIPT_DIR, "outputs"),

    "n_replicas":       16,
    "T_ref":            300.0,
    "T_max":            350.0,
    "n_steps_per_move": 1000,
    "n_iterations":     50000,
    "checkpoint_interval": 500,
    "npt_equilibration_steps": 2500000,
    "nvt_equilibration_steps": 250000,
    "unrestrained_equilibration_steps": 250000,

    "timestep":         2.0,
    "friction":         1.0,
    "random_seed":      20260920,

    "platform_preference": ["CUDA", "OpenCL", "CPU"],

    "version":          __version__,

    "cmap_scaling_mode": "scaled_maps",

    # v4.4 默认启用五残基 non-standard local-hot REST2 扩展；--no_local_heating 回到标准 REST2
    "local_heating_enabled": True,
    "local_scaling_scheme": "endpoint_anchored_geometric_atom_scale_v2",
    "hot_residue_ids": ["72", "167", "168", "344", "345"],
    "hot_residue_names": {
        "72": "PHE",
        "167": "ALA",
        "168": "ILE",
        "344": "PHE",
        "345": "ILE"
    },
    # lambda_hot 是最高 REST2 端点的额外局部倍率；参考端严格无额外偏置。
    # T_max=350K、lambda_hot=0.80 时，最高端局部有效温标为 437.5K。
    "lambda_hot": 0.80,

    "save_trajectory": True,
    "trajectory_interval": 5,
    "trajectory_format": "dcd",

    "rg_equilibration_fraction": 0.2,

    # v3.3: 周期性边界和盒子参数
    "periodic_boundary": True,
    "use_cubic_box": False,

    # v4.3: RMSD 监控
    "rmsd_monitor_interval": 50,   # 每 100 ps 计算一次 RMSD
    "rmsd_monitor_replicas": [0, 7, 15],  # 监控低/中/高有效温标副本

    # v4.3: NPT 和 NVT 平衡阶段默认启用位置约束
    "equilibration_restraints": True,
    "restraint_force_constant": 100.0,  # kJ/mol/nm² (弱约束)
}

SOLVENT_RESNAMES = {
    'HOH', 'WAT', 'TIP', 'SOL', 'TIP3', 'TIP4', 'TIP5',
    'CL', 'NA', 'K', 'CA', 'MG', 'ZN', 'SOD', 'CLA', 'CL-', 'NA+',
    'K+', 'CA2', 'MG2', 'ZN2', 'BR', 'BR-', 'I', 'I-', 'F', 'F-',
    'POPC', 'POPE', 'POPS', 'POPG', 'POPA', 'POPC2',
    'DPPC', 'DPPE', 'DPPS', 'DPPG', 'DSPA', 'DSPG',
    'DMPC', 'DMPE', 'DMPS', 'DMPG',
    'DSPC', 'DSPE', 'DSPS', 'DSPG',
    'DPOP', 'DPOPG',
    'CHL1', 'CHOL', 'CHL',
    'SM', 'SPHM',
    'POPA', 'POPA',
    'PALM', 'OLEO', 'STEA', 'MYRI',
    'GLYC', 'GLCN', 'SUG',
}

SOLVENT_SEGMENTS = {'MEMB', 'TIP3', 'HOH', 'WAT', 'SOLV', 'SOL'}

ION_RESNAME_PATTERNS = {
    'NA', 'CL', 'K', 'CA', 'MG', 'ZN', 'BR', 'I', 'F',
    'SOD', 'CLA', 'POT', 'CAL', 'MGL', 'ZNL',
    'NA+', 'CL-', 'K+', 'CA2', 'MG2', 'ZN2',
    'NAI', 'CLI', 'KI', 'CAI', 'MGI', 'ZNI',
}

AA_RESNAMES = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
    'HSD', 'HSE', 'HSP', 'HSC', 'ALAD', 'ACE', 'NME', 'CT3', 'NTER', 'CTER',
    'CYX', 'CYS2',
}

OF_GATE_RESIDUES = ['72', '167', '168', '344', '345']

OF_GATE_SELECTION = "(resid 344 or resid 345 or resid 167 or resid 168 or resid 72) and not backbone and not type H* and segid PROA"


# ======================================================================
# v4.0: RMSD 计算工具 (Kabsch 对齐)
# ======================================================================

def kabsch_align(P, Q):
    """返回将移动坐标 Q（行向量）最优旋转到参考坐标 P 的矩阵。"""
    P = np.asarray(P, dtype=float) - np.mean(P, axis=0)
    Q = np.asarray(Q, dtype=float) - np.mean(Q, axis=0)
    covariance = Q.T @ P
    U, _, Vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = 1.0 if np.linalg.det(U @ Vt) >= 0 else -1.0
    return U @ correction @ Vt


def calculate_rmsd(ref_coords, cur_coords):
    """
    计算 Kabsch 对齐后的 RMSD

    参数:
      ref_coords: (N, 3) 参考坐标
      cur_coords: (N, 3) 当前坐标

    返回:
      rmsd: float, RMSD 值 (Å)
    """
    P = np.array(ref_coords) - np.mean(ref_coords, axis=0)
    Q = np.array(cur_coords) - np.mean(cur_coords, axis=0)
    R = kabsch_align(ref_coords, cur_coords)
    Q_aligned = Q @ R
    rmsd_nm = np.sqrt(np.mean(np.sum((P - Q_aligned) ** 2, axis=1)))
    return rmsd_nm * 10.0


def extract_ca_positions(positions, ca_atom_indices):
    """
    从 OpenMM positions 中提取 CA 原子坐标

    参数:
      positions: OpenMM positions (Quantity 或 Vec3 列表)
      ca_atom_indices: CA 原子索引列表

    返回:
      (N, 3) numpy 数组 (Å)
    """
    pos = np.array([[p.x, p.y, p.z] for p in positions])
    return pos[ca_atom_indices]


def build_molecule_topology(system, topology, logger):
    """
    构建分子拓扑信息，用于make_whole操作

    返回:
      molecule_adjacency: 每个分子的原子邻接表
      molecule_atoms: 每个分子包含的原子
    """
    n_atoms = system.getNumParticles()

    bonds = []
    for bond in topology.bonds():
        p1 = bond.atom1.index
        p2 = bond.atom2.index
        bonds.append((p1, p2))

    if not bonds:
        for i in range(system.getNumForces()):
            force = system.getForce(i)
            if isinstance(force, openmm.HarmonicBondForce):
                for j in range(force.getNumBonds()):
                    p1, p2, r, k = force.getBondParameters(j)
                    bonds.append((p1, p2))
                if bonds:
                    logger.info(f"[v4.4] Found {len(bonds)} bonds from HarmonicBondForce")
                    break

    logger.info(f"[v4.4] Building molecule topology from {len(bonds)} bonds...")

    parent = list(range(n_atoms))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for p1, p2 in bonds:
        union(p1, p2)

    molecule_atoms = {}
    for atom_idx in range(n_atoms):
        mol_id = find(atom_idx)
        if mol_id not in molecule_atoms:
            molecule_atoms[mol_id] = []
        molecule_atoms[mol_id].append(atom_idx)

    atom_to_molecule = {}
    for mol_id, atoms in molecule_atoms.items():
        for atom in atoms:
            atom_to_molecule[atom] = mol_id

    molecule_adjacency = {mol_id: {} for mol_id in molecule_atoms}
    molecule_bond_counts = {mol_id: 0 for mol_id in molecule_atoms}
    for p1, p2 in bonds:
        mol_id = atom_to_molecule.get(p1)
        if mol_id is not None:
            adjacency = molecule_adjacency[mol_id]
            adjacency.setdefault(p1, []).append(p2)
            adjacency.setdefault(p2, []).append(p1)
            molecule_bond_counts[mol_id] += 1

    logger.info(f"[v4.4] Molecule topology: {len(molecule_atoms)} molecules found")
    for mol_id, atoms in sorted(molecule_atoms.items(), key=lambda x: -len(x[1]))[:5]:
        logger.info(f"  Molecule {mol_id}: {len(atoms)} atoms, {molecule_bond_counts[mol_id]} bonds")

    return molecule_adjacency, molecule_atoms


def make_whole_positions(positions, molecule_adjacency, molecule_atoms,
                        box_lengths, box_angles=None, skip_small_molecules=True,
                        small_molecule_threshold=10):
    """
    v4.0: 优化版 make_whole 操作

    改进:
      1. 跳过小分子 (水、离子等单残基分子), 仅对大分子 (蛋白、脂质) 执行
      2. P2-1修复: 使用预构建邻接表替代遍历所有键, 复杂度从O(N_bonds²)降到O(N_atoms)

    参数:
      positions: Vec3数组 (OpenMM positions)
      molecule_adjacency: 初始化时预计算的每分子原子邻接表
      molecule_atoms: 每个分子包含的原子
      box_lengths: 盒子三边长 [Lx, Ly, Lz] (nm)
      box_angles: 盒子角度 (默认90度)
      skip_small_molecules: 是否跳过小分子 (v4.0 新增)
      small_molecule_threshold: 原子数小于此值的分子视为小分子

    返回:
      修正后的positions数组
    """
    pos = np.array([[p.x, p.y, p.z] for p in positions])

    box_array = np.asarray(box_lengths, dtype=float)
    if box_array.shape == (3,):
        if np.any(box_array <= 0):
            raise ValueError("周期盒边长必须为正数")
        box_matrix = np.diag(box_array)
    elif box_array.shape == (3, 3):
        box_matrix = box_array
    else:
        raise ValueError("box_lengths 必须是三个盒长或 3x3 盒向量矩阵")
    if abs(np.linalg.det(box_matrix)) < 1e-12:
        raise ValueError("周期盒矩阵不可逆")
    inverse_box = np.linalg.inv(box_matrix)

    n_skipped = 0
    n_processed = 0

    for mol_id, atom_indices in molecule_atoms.items():
        if not atom_indices:
            continue

        # v4.0: 跳过小分子 (水、离子)
        if skip_small_molecules and len(atom_indices) < small_molecule_threshold:
            n_skipped += 1
            continue

        n_processed += 1

        adjacency = molecule_adjacency.get(mol_id, {})
        anchor_idx = atom_indices[0]

        visited = {anchor_idx}
        queue = deque([anchor_idx])

        while queue:
            current = queue.popleft()
            current_pos = pos[current]

            # 使用邻接表直接获取邻居
            for neighbor in adjacency.get(current, []):
                if neighbor not in visited:
                    # 在完整盒矩阵的分数坐标中恢复最近周期镜像，兼容正交和三斜盒
                    displacement = current_pos - pos[neighbor]
                    fractional = displacement @ inverse_box
                    lattice_shift = np.rint(fractional) @ box_matrix
                    pos[neighbor] = pos[neighbor] + lattice_shift
                    visited.add(neighbor)
                    queue.append(neighbor)

    result = [openmm.Vec3(p[0], p[1], p[2]) for p in pos]
    return result


def parse_command_line():
    parser = argparse.ArgumentParser(
        description='REST2 v4.4 with a default non-standard five-residue local-hot extension')

    parser.add_argument('--n_iterations', type=int, default=CONFIG['n_iterations'],
                        help=f'Number of iterations (default: {CONFIG["n_iterations"]})')
    parser.add_argument('--n_steps_per_move', type=int, default=CONFIG['n_steps_per_move'],
                        help=f'Steps per iteration (default: {CONFIG["n_steps_per_move"]})')
    parser.add_argument('--n_replicas', type=int, default=CONFIG['n_replicas'],
                        help=f'Number of replicas (default: {CONFIG["n_replicas"]})')
    parser.add_argument('--T_ref', type=float, default=CONFIG['T_ref'],
                        help=f'Reference temperature in K (default: {CONFIG["T_ref"]})')
    parser.add_argument('--T_max', type=float, default=CONFIG['T_max'],
                        help=f'Maximum temperature in K (default: {CONFIG["T_max"]})')

    # v4.0 新增: 局部加热控制
    parser.add_argument('--lambda_hot', type=float, default=CONFIG['lambda_hot'],
                        help=f'Highest-endpoint multiplier for local heating (default: {CONFIG["lambda_hot"]}). '
                             f'The reference endpoint is unbiased; 0.80 -> 437.5K@350K.')
    local_group = parser.add_mutually_exclusive_group()
    local_group.add_argument('--local_heating', action='store_true',
                             help='Enable non-standard five-residue local-hot REST2 extension (default)')
    local_group.add_argument('--no_local_heating', action='store_true',
                             help='Disable local-hot extension and use strict standard REST2')

    # 仅显式传参时覆盖 CONFIG，允许开启或关闭默认约束
    restraint_group = parser.add_mutually_exclusive_group()
    restraint_group.add_argument('--equilibration_restraints', dest='equilibration_restraints',
                                 action='store_true', help='Apply weak position restraints during equilibration')
    restraint_group.add_argument('--no_equilibration_restraints', dest='equilibration_restraints',
                                 action='store_false', help='Disable position restraints during equilibration')
    parser.set_defaults(equilibration_restraints=None)

    parser.add_argument('--restore', type=str, default=None,
                        help='Checkpoint file to restore from')
    parser.add_argument('--start_iteration', type=int, default=0,
                        help='Starting iteration number (for resume)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (required when restoring)')

    parser.add_argument('--checkpoint_interval', type=int, default=CONFIG['checkpoint_interval'],
                        help=f'Checkpoint saving interval (default: {CONFIG["checkpoint_interval"]})')
    parser.add_argument('--npt_equilibration_steps', type=int, default=CONFIG['npt_equilibration_steps'],
                        help=f'NPT equilibration steps (default: {CONFIG["npt_equilibration_steps"]})')
    parser.add_argument('--nvt_equilibration_steps', type=int, default=CONFIG['nvt_equilibration_steps'],
                        help=f'Restrained NVT equilibration steps (default: {CONFIG["nvt_equilibration_steps"]})')
    parser.add_argument('--unrestrained_equilibration_steps', type=int,
                        default=CONFIG['unrestrained_equilibration_steps'],
                        help=f'Unrestrained NVT equilibration steps (default: {CONFIG["unrestrained_equilibration_steps"]})')
    parser.add_argument('--trajectory_interval', type=int, default=CONFIG['trajectory_interval'],
                        help=f'Trajectory interval in iterations (default: {CONFIG["trajectory_interval"]})')
    parser.add_argument('--rmsd_monitor_interval', type=int, default=CONFIG['rmsd_monitor_interval'],
                        help=f'RMSD monitoring interval in iterations (default: {CONFIG["rmsd_monitor_interval"]})')

    return parser.parse_args()


def validate_config(cfg):
    integer_keys = ("n_replicas", "n_iterations", "n_steps_per_move", "checkpoint_interval",
                    "trajectory_interval", "rmsd_monitor_interval", "npt_equilibration_steps",
                    "nvt_equilibration_steps", "unrestrained_equilibration_steps")
    for key in integer_keys:
        value = cfg[key]
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(f"{key} 必须为正整数")
    if cfg["n_replicas"] < 2:
        raise ValueError("n_replicas 必须 >= 2")
    if "start_iteration" in cfg:
        value = cfg["start_iteration"]
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError("start_iteration 必须为非负整数")
        if value > cfg["n_iterations"]:
            raise ValueError("start_iteration 不能大于 n_iterations")
    for key in ("T_ref", "T_max", "timestep", "friction", "restraint_force_constant"):
        if not np.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"{key} 必须是有限正数")
    if cfg["T_max"] < cfg["T_ref"]:
        raise ValueError("T_max 必须 >= T_ref")
    if not np.isfinite(cfg["lambda_hot"]) or not (0 < cfg["lambda_hot"] <= 1):
        raise ValueError("lambda_hot 必须有限且满足 0 < lambda_hot <= 1")
    if cfg.get("local_scaling_scheme") != "endpoint_anchored_geometric_atom_scale_v2":
        raise ValueError("v4.4 仅支持 local_scaling_scheme='endpoint_anchored_geometric_atom_scale_v2'")
    if not np.isfinite(cfg["rg_equilibration_fraction"]) or not (0 <= cfg["rg_equilibration_fraction"] < 1):
        raise ValueError("rg_equilibration_fraction 必须有限且满足 0 <= value < 1")
    if cfg.get("trajectory_format", "dcd").lower() != "dcd":
        raise ValueError("trajectory_format 仅支持 'dcd'")
    seed = cfg.get("random_seed")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or not (0 <= seed <= 0x7FFFFFFF):
        raise ValueError("random_seed 必须是 0 到 2147483647 的整数")


class ExperimentManager:
    """实验管理器"""

    def __init__(self):
        self.experiment_dir = None
        self.figures_dir = None
        self.mbar_dir = None
        self.log_file = None
        self.logger = None
        self.cfg = None
        self.experiment_id = None

    def setup_experiment(self, output_dir=None):
        if output_dir:
            self.experiment_dir = output_dir
            self.experiment_id = os.path.basename(output_dir)
        else:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_id = f"4au5_rest2_v4.4_{ts}"
            self.experiment_dir = os.path.join(CONFIG["base_output_dir"], f"experiment_{ts}")

        self.figures_dir = os.path.join(self.experiment_dir, "figures")
        self.mbar_dir = os.path.join(self.experiment_dir, "mbar_analysis")
        self.trajectories_dir = os.path.join(self.experiment_dir, "trajectories")
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)
        os.makedirs(self.mbar_dir, exist_ok=True)
        os.makedirs(self.trajectories_dir, exist_ok=True)
        self.cfg = copy.deepcopy(CONFIG)
        self.cfg["output_dir"] = self.experiment_dir
        self.cfg["figures_dir"] = self.figures_dir
        self.cfg["mbar_dir"] = self.mbar_dir
        self.cfg["trajectories_dir"] = self.trajectories_dir
        self.cfg["experiment_id"] = self.experiment_id
        self.setup_logging()
        return self.experiment_dir

    def setup_logging(self):
        self.log_file = os.path.join(self.experiment_dir, "experiment.log")
        self.logger = logging.getLogger(f"REST2_v4.4_4au5_{self.experiment_id}")
        self.logger.setLevel(logging.DEBUG)
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
        self.logger.handlers.clear()
        fh = logging.FileHandler(self.log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        return self.logger

    def save_config(self):
        cfg_copy = {}
        for k, v in self.cfg.items():
            if isinstance(v, (str, int, float, bool, list, dict)):
                cfg_copy[k] = v
            elif isinstance(v, np.ndarray):
                cfg_copy[k] = v.tolist()
            elif isinstance(v, (np.integer, np.floating)):
                cfg_copy[k] = float(v)
            elif isinstance(v, np.bool_):
                cfg_copy[k] = bool(v)
            else:
                cfg_copy[k] = str(v)
        cfg_file = os.path.join(self.experiment_dir, "config.json")
        with open(cfg_file, 'w', encoding='utf-8') as f:
            json.dump(cfg_copy, f, indent=2, ensure_ascii=False)
        return cfg_file


def select_platform(preference, logger):
    for pref in preference:
        try:
            platform = openmm.Platform.getPlatformByName(pref)
            test_sys = openmm.System()
            test_sys.addParticle(1.0)
            integ = openmm.LangevinMiddleIntegrator(300*unit.kelvin, 1/unit.picosecond, 2*unit.femtoseconds)
            ctx = openmm.Context(test_sys, integ, platform)
            del ctx, integ, test_sys
            if pref in ("OpenCL", "CUDA"):
                platform.setPropertyDefaultValue("Precision", "mixed")
            logger.info(f"Platform: {pref}")
            return platform, pref
        except Exception as e:
            logger.debug(f"Platform {pref} test failed: {e}")
            continue
    logger.warning("No GPU platform found, using CPU")
    return openmm.Platform.getPlatformByName("CPU"), "CPU"


def cuda_device_props(index, logger):
    """[v4.4.1 multi-GPU] 按 CUDA_VISIBLE_DEVICES 轮转分配 CudaDeviceIndex。

    v4.4 原版所有 Context 共享 platform 默认设备 (多卡申请也只用 device 0)。
    本函数把副本轮转分布到全部可见设备上 (CudaDeviceIndex 是相对索引,
    OpenMM 枚举设备时已遵循 CUDA_VISIBLE_DEVICES 掩码)。
    单卡 / 无 CUDA 环境返回 None, 行为与 v4.4 完全一致。"""
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    devices = [d.strip() for d in vis.split(",") if d.strip() != ""] if vis.strip() else []
    if len(devices) <= 1:
        return None
    dev = str(index % len(devices))
    logger.info(f"[v4.4.1 multi-GPU] Context #{index} -> CUDA device {dev}")
    return {"CudaDeviceIndex": dev}


def read_charmm_gui_box_info(boxinfo_str, logger):
    """
    从CHARMM-GUI的step5_charmm2omm.str读取盒子信息
    """
    try:
        with open(boxinfo_str, 'r') as f:
            lines = f.readlines()

        box_dims = [0.0, 0.0, 0.0]
        for line in lines:
            if 'BOXLX' in line:
                parts = line.split('=')
                if len(parts) == 2:
                    box_dims[0] = float(parts[1].strip())
            elif 'BOXLY' in line:
                parts = line.split('=')
                if len(parts) == 2:
                    box_dims[1] = float(parts[1].strip())
            elif 'BOXLZ' in line:
                parts = line.split('=')
                if len(parts) == 2:
                    box_dims[2] = float(parts[1].strip())

        box_angles = [90.0, 90.0, 90.0]

        logger.info(f"[v4.4] Box from step5_charmm2omm.str: {box_dims[0]:.6f} x {box_dims[1]:.6f} x {box_dims[2]:.6f} Å")
        return box_dims, box_angles

    except Exception as e:
        logger.warning(f"[v4.4] Could not read boxinfo file: {e}")

    logger.warning("[v4.4] Using default box size")
    return [82.407, 82.407, 109.362], [90.0, 90.0, 90.0]


def setup_periodic_box_vectors(system, box_dims, box_angles, use_cubic=False, logger=None):
    """
    设置周期性盒子向量
    """
    a, b, c = box_dims
    alpha, beta, gamma = box_angles

    if use_cubic:
        max_dim = max(a, b, c)
        box_size = max_dim
        L = box_size / 10.0

        vectors = [
            openmm.Vec3(L, 0, 0),
            openmm.Vec3(0, L, 0),
            openmm.Vec3(0, 0, L)
        ] * unit.nanometer

        if logger:
            logger.info(f"[v4.4] Setting cubic periodic box: {box_size:.2f} x {box_size:.2f} x {box_size:.2f} Å")
    else:
        if alpha == 90.0 and beta == 90.0 and gamma == 90.0:
            a_nm, b_nm, c_nm = a/10.0, b/10.0, c/10.0

            vectors = [
                openmm.Vec3(a_nm, 0, 0),
                openmm.Vec3(0, b_nm, 0),
                openmm.Vec3(0, 0, c_nm)
            ] * unit.nanometer

            box_size = max(a, b, c)
            if logger:
                logger.info(f"[v4.4] Setting orthogonal periodic box: {a:.2f} x {b:.2f} x {c:.2f} Å")
        else:
            alpha_rad = np.radians(alpha)
            beta_rad = np.radians(beta)
            gamma_rad = np.radians(gamma)

            a_nm, b_nm, c_nm = a/10.0, b/10.0, c/10.0

            lx = a_nm
            ly = b_nm * np.sin(gamma_rad)
            xy = b_nm * np.cos(gamma_rad)
            xz = c_nm * np.cos(beta_rad)
            yz = (b_nm * c_nm * np.cos(alpha_rad) - xy * xz) / ly
            z_squared = c_nm**2 - xz**2 - yz**2
            if z_squared <= 0:
                raise ValueError("无效三斜盒参数：c²-xz²-yz² 必须为正")
            lz = np.sqrt(z_squared)

            vectors = [
                openmm.Vec3(lx, 0, 0),
                openmm.Vec3(xy, ly, 0),
                openmm.Vec3(xz, yz, lz)
            ] * unit.nanometer

            box_size = max(a, b, c)
            if logger:
                logger.info(f"[v4.4] Setting triclinic periodic box: {a:.2f} x {b:.2f} x {c:.2f} Å")

    system.setDefaultPeriodicBoxVectors(*vectors)

    return box_size


def build_base_system(cfg, logger):
    """
    构建CHARMM36m基础系统(v4.0: 与v3.3一致 + 残基映射记录)
    """
    logger.info("=== Building CHARMM36m Force Field System (v4.4) ===")
    logger.info("[v4.4] Periodic Boundary Conditions: ENABLED")
    logger.info(f"[v4.4] Cubic Box: {cfg.get('use_cubic_box', False)}")

    psf = app.CharmmPsfFile(cfg["psf_file"])
    pdb = app.PDBFile(cfg["pdb_file"])

    # v4.0: 记录残基编号映射
    protein_residues = []
    for res in pdb.topology.residues():
        if res.name in AA_RESNAMES:
            protein_residues.append((res.id, res.name, res.index))

    if protein_residues:
        first_id = int(protein_residues[0][0])
        last_id = int(protein_residues[-1][0])
        n_res = len(protein_residues)
        residue_offset = first_id - 1  # 输出 PDB 中残基 1 对应输入的 first_id
        logger.info(f"\n[v4.4] Residue Mapping:")
        logger.info(f"  Input PDB residues: {first_id}-{last_id} ({n_res} residues)")
        logger.info(f"  Output PDB residues: 1-{n_res} (OpenMM auto-renumber)")
        logger.info(f"  Mapping: output_resid = input_resid - {residue_offset}")
        logger.info(f"  Hot residues (input -> output):")
        for hid in cfg["hot_residue_ids"]:
            out_id = int(hid) - residue_offset
            logger.info(f"    {hid} ({cfg['hot_residue_names'][hid]}) -> {out_id}")
        cfg["residue_offset"] = residue_offset
        cfg["input_first_residue_id"] = first_id
        cfg["input_last_residue_id"] = last_id

    box_dims, box_angles = read_charmm_gui_box_info(cfg["boxinfo_str"], logger)

    original_box = box_dims.copy()

    psf.setBox(box_dims[0]*unit.angstrom, box_dims[1]*unit.angstrom, box_dims[2]*unit.angstrom)
    logger.info(f"[v4.4] PSF box set: {box_dims[0]:.6f} x {box_dims[1]:.6f} x {box_dims[2]:.6f} Å")

    par_files = []
    charmmgui_dir = os.path.dirname(cfg["toppar_str"])
    toppar_dir = os.path.join(charmmgui_dir, "toppar")

    with open(cfg["toppar_str"], 'r') as f:
        for line in f:
            line = line.split('!')[0].strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) >= 6 and parts[0] == 'open' and 'name' in parts:
                name_idx = parts.index('name')
                if name_idx + 1 < len(parts):
                    par_path = parts[name_idx + 1]
                    full = os.path.normpath(os.path.join(charmmgui_dir, par_path))
                    if os.path.exists(full):
                        par_files.append(full)
            elif len(parts) >= 2 and parts[0] == 'stream':
                par_path = parts[1]
                full = os.path.normpath(os.path.join(charmmgui_dir, par_path))
                if os.path.exists(full):
                    par_files.append(full)

    logger.info(f"Loaded {len(par_files)} parameter files")

    params = app.CharmmParameterSet(*par_files)

    system = psf.createSystem(
        params,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.2*unit.nanometer,
        constraints=app.HBonds,
        rigidWater=True,
        ewaldErrorTolerance=0.0005
    )

    # Force-switch vdW (r_on=1.0nm, r_off=1.2nm)
    n_forces = system.getNumForces()
    vdw_switching_enabled = False
    for i in range(n_forces):
        force = system.getForce(i)
        if isinstance(force, openmm.NonbondedForce):
            force.setUseSwitchingFunction(True)
            force.setSwitchingDistance(1.0 * unit.nanometer)
            vdw_switching_enabled = True
            logger.info(f"[v4.4] Force-switch vdW enabled: r_on=1.0nm, r_off=1.2nm (Force {i})")

    if not vdw_switching_enabled:
        logger.warning("[v4.4] No NonbondedForce found for Force-switch vdW!")

    if cfg.get("periodic_boundary", True):
        box_size = setup_periodic_box_vectors(
            system, box_dims, box_angles,
            use_cubic=cfg.get("use_cubic_box", False),
            logger=logger
        )
    else:
        box_size = max(box_dims)
        logger.warning("[v4.4] Periodic boundary vectors NOT explicitly set (legacy mode)")

    n_atoms = system.getNumParticles()
    n_forces = system.getNumForces()

    # PDB 无 CRYST1 时 topology 无盒子, DCDFile 会丢弃 unit-cell 记录
    pdb.topology.setPeriodicBoxVectors(system.getDefaultPeriodicBoxVectors())
    logger.info(f"[v4.4] Topology periodic box vectors set (required for DCD unit-cell records)")

    logger.info(f"System: {n_atoms} atoms, {n_forces} forces")
    logger.info(f"[v4.4] Periodic box: {original_box[0]:.6f} x {original_box[1]:.6f} x {original_box[2]:.6f} Å")

    logger.info(f"\n[v4.4] Force Field Details:")
    logger.info(f"  Nonbonded method: PME")
    logger.info(f"  Nonbonded cutoff: 1.2 nm")
    logger.info(f"  Force-switch vdW: {'YES (r_on=1.0nm)' if vdw_switching_enabled else 'NO'}")
    logger.info(f"  Constraints: HBonds")
    logger.info(f"  Rigid water: YES")
    logger.info(f"  PME ewald tolerance: 0.0005")

    cmap_count = 0
    for i in range(n_forces):
        force = system.getForce(i)
        ftype = type(force).__name__
        if isinstance(force, openmm.CMAPTorsionForce):
            cmap_count = force.getNumTorsions()
            logger.info(f"  Force {i}: {ftype} ({cmap_count} torsions) [CMAP ENABLED]")
        elif isinstance(force, openmm.NonbondedForce):
            switch_info = " [SWITCHING ON]" if force.getUseSwitchingFunction() else ""
            logger.info(f"  Force {i}: {ftype} ({force.getNumParticles()} particles){switch_info}")
        elif isinstance(force, openmm.HarmonicBondForce):
            logger.info(f"  Force {i}: {ftype} ({force.getNumBonds()} bonds)")
        elif isinstance(force, openmm.HarmonicAngleForce):
            logger.info(f"  Force {i}: {ftype} ({force.getNumAngles()} angles)")
        elif isinstance(force, openmm.PeriodicTorsionForce):
            logger.info(f"  Force {i}: {ftype} ({force.getNumTorsions()} torsions)")
        elif isinstance(force, openmm.CustomTorsionForce):
            logger.info(f"  Force {i}: {ftype} ({force.getNumTorsions()} torsions)")
        else:
            logger.info(f"  Force {i}: {ftype}")

    return system, psf, pdb, box_size, original_box


def detect_ions(topology, logger):
    """检测系统中的离子种类"""
    detected_ions = set()
    for atom in topology.atoms():
        resname = atom.residue.name.upper()
        if resname in ION_RESNAME_PATTERNS:
            detected_ions.add(resname)

    if detected_ions:
        logger.info(f"[v4.4] Detected ions: {detected_ions}")
    else:
        logger.info("[v4.4] No ions detected in system")

    return detected_ions


def identify_solute_atoms(topology, logger):
    solute_atoms = set()
    solute_residues = []
    solvent_atoms = set()

    segment_attr = None
    for atom in topology.atoms():
        if hasattr(atom.residue, 'segment_id'):
            segment_attr = 'segment_id'
            break
        elif hasattr(atom.residue, 'segment'):
            segment_attr = 'segment'
            break
        elif hasattr(atom.residue, 'chain_id'):
            segment_attr = 'chain_id'
            break

    if segment_attr:
        logger.info(f"[v4.4] Using segment-based solute identification (attribute: {segment_attr})")
        for atom in topology.atoms():
            resname = atom.residue.name.upper()
            seg_id = getattr(atom.residue, segment_attr, '').upper()
            if seg_id in SOLVENT_SEGMENTS or resname in SOLVENT_RESNAMES:
                solvent_atoms.add(atom.index)
            else:
                solute_atoms.add(atom.index)
                if atom.residue.id not in solute_residues:
                    solute_residues.append(atom.residue.id)
    else:
        logger.info("[v4.4] Using residue-name-based solute identification")
        for atom in topology.atoms():
            resname = atom.residue.name.upper()
            if resname not in SOLVENT_RESNAMES:
                solute_atoms.add(atom.index)
                if atom.residue.id not in solute_residues:
                    solute_residues.append(atom.residue.id)
            else:
                solvent_atoms.add(atom.index)

    n_total = topology.getNumAtoms()
    n_solute = len(solute_atoms)
    n_solvent = len(solvent_atoms)
    logger.info(f"[v4.4] Solute: {n_solute} atoms ({len(solute_residues)} residues), "
               f"Solvent: {n_solvent} atoms, Ratio: {n_solute/n_total*100:.1f}%")

    return solute_atoms


def identify_hot_residue_atoms(topology, solute_atoms, cfg, logger):
    target_ids = [str(res_id) for res_id in cfg["hot_residue_ids"]]
    expected_names = {str(k): str(v).upper() for k, v in cfg["hot_residue_names"].items()}
    if len(target_ids) != 5 or len(set(target_ids)) != 5:
        raise ValueError("local-hot REST2 必须配置恰好 5 个互不重复的目标残基")
    if set(expected_names) != set(target_ids):
        raise ValueError("hot_residue_names 必须与 5 个 hot_residue_ids 精确对应")

    candidates = {res_id: [] for res_id in target_ids}
    for residue in topology.residues():
        if str(residue.id) in candidates and residue.name.upper() in AA_RESNAMES:
            candidates[str(residue.id)].append(residue)

    hot_atoms = {}
    for res_id in target_ids:
        matches = candidates[res_id]
        if len(matches) != 1:
            raise ValueError(
                f"Hot residue {res_id} 必须在蛋白拓扑中唯一存在，实际匹配 {len(matches)} 个")
        residue = matches[0]
        actual_name = residue.name.upper()
        expected_name = expected_names[res_id]
        if actual_name != expected_name:
            raise ValueError(
                f"Hot residue {res_id} 名称不匹配: expected={expected_name}, actual={actual_name}")
        atoms = {atom.index for atom in residue.atoms()}
        if not atoms:
            raise ValueError(f"Hot residue {res_id} 不含原子")
        if not atoms.issubset(solute_atoms):
            bad_atoms = sorted(atoms - solute_atoms)
            raise ValueError(f"Hot residue {res_id} 含非溶质原子: {bad_atoms}")
        hot_atoms[res_id] = atoms
        logger.info(f"Hot residue {res_id} ({actual_name}): {len(atoms)} atoms identified")

    all_hot_atoms = set().union(*hot_atoms.values())
    if len(hot_atoms) != 5:
        raise ValueError(f"Local-hot residue audit failed: expected 5, got {len(hot_atoms)}")

    logger.info("[v4.4] NON-STANDARD local-hot REST2 extension enabled")
    logger.info(f"[v4.4] Local-hot audit passed: exactly 5 unique residues; all hot atoms are solute")
    logger.info(f"[v4.4] Total hot atoms: {len(all_hot_atoms)}")
    logger.info(f"[v4.4] Local scaling scheme: {cfg['local_scaling_scheme']}")
    logger.info(f"[v4.4] Lambda_hot: {cfg['lambda_hot']}")

    T_max = cfg["T_max"]
    T_ref = cfg["T_ref"]
    lam_hot = cfg["lambda_hot"]
    T_eff_ref = T_ref
    T_eff_max = T_max / lam_hot
    logger.info(f"[v4.4] Endpoint-anchored effective temperatures for hot residues:")
    logger.info(f"  At reference slot T_eff_solute={T_ref}K: local_multiplier=1.0000, hot T_eff={T_eff_ref:.1f}K")
    logger.info(f"  At highest slot T_eff_solute={T_max}K: local_multiplier={lam_hot:.4f}, hot T_eff={T_eff_max:.1f}K")

    if lam_hot < 0.5:
        logger.warning(f"[v4.4] WARNING: lambda_hot={lam_hot} is very aggressive!")
        logger.warning(f"  T_eff at T_max = {T_eff_max:.0f}K may cause protein unfolding!")
        logger.warning(f"  Recommended: lambda_hot >= 0.6")
    elif lam_hot < 0.6:
        logger.info(f"[v4.4] Note: lambda_hot={lam_hot} is moderately aggressive. T_eff_max={T_eff_max:.0f}K")

    return all_hot_atoms, hot_atoms


def identify_ca_atoms(topology, logger):
    """
    v4.0: 识别蛋白 CA 原子索引, 用于 RMSD 监控
    """
    ca_indices = []
    for atom in topology.atoms():
        if atom.name == 'CA' and atom.residue.name in AA_RESNAMES:
            ca_indices.append(atom.index)

    logger.info(f"[v4.4] Identified {len(ca_indices)} CA atoms for RMSD monitoring")
    return ca_indices


def identify_hot_heavy_atoms(topology, hot_atoms, logger):
    hot_heavy_indices = []
    for atom in topology.atoms():
        if atom.index not in hot_atoms:
            continue
        symbol = getattr(getattr(atom, "element", None), "symbol", "")
        atom_name = atom.name.strip().upper().lstrip("0123456789")
        if str(symbol).upper() != "H" and not atom_name.startswith("H"):
            hot_heavy_indices.append(atom.index)
    if hot_atoms and not hot_heavy_indices:
        raise ValueError("未能从 local-hot 五残基中识别重原子")
    logger.info(f"[v4.4] Identified {len(hot_heavy_indices)} local-hot heavy atoms for RMSD monitoring")
    return hot_heavy_indices


def local_multiplier_for_lambda(lam_global, lambda_min, lambda_hot):
    if np.isclose(lambda_min, 1.0):
        progress = 0.0
    else:
        progress = np.log(lam_global) / np.log(lambda_min)
    progress = float(np.clip(progress, 0.0, 1.0))
    return float(lambda_hot ** progress)


def atom_scale(atom, solute_atoms, hot_atoms, lam_global, local_multiplier):
    if atom in hot_atoms:
        return lam_global * local_multiplier
    if atom in solute_atoms:
        return lam_global
    return 1.0


def geometric_term_scale(atoms, solute_atoms, hot_atoms, lam_global, local_multiplier):
    scales = [atom_scale(a, solute_atoms, hot_atoms, lam_global, local_multiplier) for a in atoms]
    return float(np.prod(scales) ** (1.0 / len(scales)))


def classify_term(atoms, solute_atoms, hot_atoms):
    hot_count = sum(a in hot_atoms for a in atoms)
    if hot_count == len(atoms):
        return "hot"
    if hot_count:
        return "mixed"
    return "normal"


def create_scaled_cmap_force(original_force, solute_atoms, hot_atoms, lam_global, local_multiplier, logger):
    n_torsions = original_force.getNumTorsions()
    logger.info(f"  [CMAP] Creating per-scale cached maps for {n_torsions} torsions")
    new_force = openmm.CMAPTorsionForce()
    if hasattr(original_force, "getName") and hasattr(new_force, "setName"):
        new_force.setName(original_force.getName())
    if hasattr(original_force, "getForceGroup") and hasattr(new_force, "setForceGroup"):
        new_force.setForceGroup(original_force.getForceGroup())

    map_cache = {}
    counts = {"normal": 0, "hot": 0, "mixed": 0}
    for torsion_idx in range(n_torsions):
        params = original_force.getTorsionParameters(torsion_idx)
        orig_map_idx = int(params[0])
        atoms = tuple(int(a) for a in params[1:9])
        scale = geometric_term_scale(atoms, solute_atoms, hot_atoms, lam_global, local_multiplier)
        cache_key = (orig_map_idx, scale)
        if cache_key not in map_cache:
            size, energy_grid = original_force.getMapParameters(orig_map_idx)
            map_cache[cache_key] = new_force.addMap(size, energy_grid * scale)
        new_force.addTorsion(map_cache[cache_key], *atoms)
        counts[classify_term(atoms, solute_atoms, hot_atoms)] += 1

    logger.info(f"  [CMAP] maps cached by (source map, actual scale): {len(map_cache)}")
    logger.info(f"  [CMAP] normal/hot/mixed torsions: "
                f"{counts['normal']}/{counts['hot']}/{counts['mixed']}")
    return new_force, counts


def custom_torsion_amplitude_indices(force, logger):
    names = [force.getPerTorsionParameterName(i)
             for i in range(force.getNumPerTorsionParameters())]
    expression = "".join(force.getEnergyFunction().lower().split())
    # 本 CHARMM36m 体系的 improper 项为 k*min(dtheta,2*pi-dtheta)^2；k 是唯一能量幅度。
    if names == ["k", "theta0"] and expression.startswith("k*min(dtheta,2*pi-dtheta)^2;"):
        logger.info("    CustomTorsionForce: recognized CHARMM36m improper amplitude parameter ['k']")
        return [0]
    logger.warning("    CustomTorsionForce expression cannot be scaled reliably; left unchanged: "
                   f"parameters={names}, expression={force.getEnergyFunction()!r}")
    return []


def create_rest2_system_with_local_heating(base_system, solute_atoms, hot_atoms, lam_global,
                                            lambda_min, lambda_hot, cfg, logger):
    system = copy.deepcopy(base_system)
    local_multiplier = local_multiplier_for_lambda(lam_global, lambda_min, lambda_hot)
    hot_atom_scale = lam_global * local_multiplier
    logger.info("  Creating REST2 Hamiltonian with endpoint-anchored geometric per-atom term scaling:")
    logger.info(f"    λ_global={lam_global:.4f}, local_multiplier={local_multiplier:.4f}, "
                f"hot_atom_scale={hot_atom_scale:.4f}")
    audit = {name: {"normal": 0, "hot": 0, "mixed": 0} for name in
             ("bond", "angle", "periodic torsion", "custom torsion", "CMAP",
              "nonbonded particle", "nonbonded exception")}
    cmap_forces_to_replace = []

    for i in range(system.getNumForces()):
        force = system.getForce(i)
        if isinstance(force, openmm.CustomNonbondedForce):
            expression = force.getEnergyFunction()
            expression_lower = expression.lower()
            if "acoef" in expression_lower and "bcoef" in expression_lower:
                original_parameter_count = force.getNumPerParticleParameters()
                force.addPerParticleParameter("restScale")
                for j in range(force.getNumParticles()):
                    parameters = list(force.getParticleParameters(j))
                    if len(parameters) != original_parameter_count:
                        raise ValueError(f"CustomNonbondedForce 粒子 {j} 参数数不一致")
                    parameters.append(atom_scale(j, solute_atoms, hot_atoms, lam_global, local_multiplier))
                    force.setParticleParameters(j, parameters)
                terms = expression.split(";", 1)
                force.setEnergyFunction(
                    f"sqrt(restScale1*restScale2)*({terms[0]})" +
                    ((";" + terms[1]) if len(terms) == 2 else ""))
                logger.info("    CustomNonbondedForce CHARMM LJ uses sqrt(s_i*s_j)")

        elif isinstance(force, openmm.NonbondedForce):
            particle_scales = np.ones(force.getNumParticles())
            for j in range(force.getNumParticles()):
                q, sig, eps = force.getParticleParameters(j)
                scale = atom_scale(j, solute_atoms, hot_atoms, lam_global, local_multiplier)
                particle_scales[j] = scale
                force.setParticleParameters(j, q * np.sqrt(scale), sig, eps * scale)
                category = "hot" if j in hot_atoms else "normal"
                audit["nonbonded particle"][category] += 1
            for j in range(force.getNumExceptions()):
                p1, p2, q12, sig, eps = force.getExceptionParameters(j)
                pair_scale = np.sqrt(particle_scales[p1] * particle_scales[p2])
                force.setExceptionParameters(j, p1, p2, q12 * pair_scale, sig, eps * pair_scale)
                audit["nonbonded exception"][classify_term((p1, p2), solute_atoms, hot_atoms)] += 1

        elif isinstance(force, openmm.HarmonicBondForce):
            for j in range(force.getNumBonds()):
                p1, p2, length, k = force.getBondParameters(j)
                atoms = (p1, p2)
                scale = geometric_term_scale(atoms, solute_atoms, hot_atoms, lam_global, local_multiplier)
                force.setBondParameters(j, p1, p2, length, k * scale)
                audit["bond"][classify_term(atoms, solute_atoms, hot_atoms)] += 1

        elif isinstance(force, openmm.HarmonicAngleForce):
            for j in range(force.getNumAngles()):
                p1, p2, p3, theta, k = force.getAngleParameters(j)
                atoms = (p1, p2, p3)
                scale = geometric_term_scale(atoms, solute_atoms, hot_atoms, lam_global, local_multiplier)
                force.setAngleParameters(j, p1, p2, p3, theta, k * scale)
                audit["angle"][classify_term(atoms, solute_atoms, hot_atoms)] += 1

        elif isinstance(force, openmm.PeriodicTorsionForce):
            for j in range(force.getNumTorsions()):
                p1, p2, p3, p4, per, phase, k = force.getTorsionParameters(j)
                atoms = (p1, p2, p3, p4)
                scale = geometric_term_scale(atoms, solute_atoms, hot_atoms, lam_global, local_multiplier)
                force.setTorsionParameters(j, p1, p2, p3, p4, per, phase, k * scale)
                audit["periodic torsion"][classify_term(atoms, solute_atoms, hot_atoms)] += 1

        elif isinstance(force, openmm.CustomTorsionForce):
            amplitude_indices = custom_torsion_amplitude_indices(force, logger)
            for j in range(force.getNumTorsions()):
                p1, p2, p3, p4, parameters = force.getTorsionParameters(j)
                atoms = (p1, p2, p3, p4)
                category = classify_term(atoms, solute_atoms, hot_atoms)
                audit["custom torsion"][category] += 1
                if amplitude_indices:
                    scale = geometric_term_scale(atoms, solute_atoms, hot_atoms,
                                                 lam_global, local_multiplier)
                    parameters = list(parameters)
                    for parameter_idx in amplitude_indices:
                        parameters[parameter_idx] *= scale
                    force.setTorsionParameters(j, p1, p2, p3, p4, parameters)

        elif isinstance(force, openmm.CMAPTorsionForce):
            if cfg["cmap_scaling_mode"] == "scaled_maps":
                scaled_cmap, counts = create_scaled_cmap_force(
                    force, solute_atoms, hot_atoms, lam_global, local_multiplier, logger)
                cmap_forces_to_replace.append((i, scaled_cmap))
                for category in counts:
                    audit["CMAP"][category] += counts[category]

    logger.info("  Local heating coverage audit (normal/hot/mixed):")
    for term_name, counts in audit.items():
        logger.info(f"    {term_name}: {counts['normal']}/{counts['hot']}/{counts['mixed']}")
    return system, cmap_forces_to_replace


def rebuild_system_with_scaled_cmap(base_system, cmap_replacements, logger):
    if not cmap_replacements:
        return base_system

    logger.info("  [CMAP] Rebuilding system with scaled CMAP forces")

    new_system = openmm.System()

    for i in range(base_system.getNumParticles()):
        mass = base_system.getParticleMass(i)
        new_system.addParticle(mass)

    for i in range(base_system.getNumConstraints()):
        p1, p2, length = base_system.getConstraintParameters(i)
        new_system.addConstraint(p1, p2, length)

    new_system.setDefaultPeriodicBoxVectors(*base_system.getDefaultPeriodicBoxVectors())

    cmap_replace_dict = {orig_idx: new_force for orig_idx, new_force in cmap_replacements}

    for i in range(base_system.getNumForces()):
        if i in cmap_replace_dict:
            new_system.addForce(cmap_replace_dict[i])
            logger.info(f"    Added scaled CMAPTorsionForce (replacing force {i})")
        else:
            force = base_system.getForce(i)
            new_system.addForce(copy.deepcopy(force))

    return new_system


def optimize_temperatures_geometric(n_replicas, T_ref, T_max, logger=None):
    temps = [T_ref * (T_max/T_ref)**(i/(n_replicas-1)) for i in range(n_replicas)]
    lams = [T_ref/T for T in temps]

    if logger:
        logger.info(f"\n{'='*70}")
        logger.info(f"REST2 Effective Solute Temperature Distribution ({n_replicas} replicas, {T_ref}-{T_max}K; bath {T_ref}K)")
        logger.info(f"{'─'*70}")
        logger.info(f"  Rep    T(K)    λ_global    ΔT(K)")
        logger.info(f"{'─'*70}")
        for i, (T, lam) in enumerate(zip(temps, lams)):
            dT = T - temps[i-1] if i > 0 else 0
            if i < 10 or i >= n_replicas - 3:
                logger.info(f"  {i:2d}   {T:6.1f}   {lam:.4f}    {dT:5.2f}")
            elif i == 10:
                logger.info(f"  ...  (中间副本省略显示)  ...")
        logger.info(f"{'='*70}")
        dTs = [temps[i] - temps[i-1] for i in range(1, len(temps))]
        logger.info(f"[v4.4] Temperature spacing: min={min(dTs):.2f}K, max={max(dTs):.2f}K, avg={np.mean(dTs):.2f}K")

    return temps, lams


def analyze_walker_round_trips(walker_label_history, n_replicas=None):
    """反转 slot->walker 历史并统计 walker 混合、占用和完整端点往返。"""
    history = np.asarray(walker_label_history, dtype=int)
    if history.ndim != 2 or history.shape[0] == 0:
        raise ValueError("walker_label_history 必须是非空二维数组")
    if n_replicas is None:
        n_replicas = history.shape[1]
    if history.shape[1] != n_replicas or n_replicas < 2:
        raise ValueError("walker_label_history 的槽位数与 n_replicas 不一致")
    expected = np.arange(n_replicas)
    for row in history:
        if not np.array_equal(np.sort(row), expected):
            raise ValueError("walker_label_history 每一行都必须是完整 walker 标签排列")

    walker_slots = np.empty_like(history)
    for frame, slot_labels in enumerate(history):
        walker_slots[frame, slot_labels] = expected

    counts = np.zeros((n_replicas, 2), dtype=int)
    visited_slot_counts = np.zeros(n_replicas, dtype=int)
    visited_both_ends = np.zeros(n_replicas, dtype=bool)
    occupancy_uniform_rmse = np.zeros(n_replicas, dtype=float)
    top = n_replicas - 1
    uniform = 1.0 / n_replicas
    for walker in range(n_replicas):
        slots = walker_slots[:, walker]
        visited_slot_counts[walker] = len(np.unique(slots))
        visited_both_ends[walker] = np.any(slots == 0) and np.any(slots == top)
        occupancy = np.bincount(slots, minlength=n_replicas) / len(slots)
        occupancy_uniform_rmse[walker] = np.sqrt(np.mean((occupancy - uniform) ** 2)) / uniform
        endpoint_visits = []
        for slot in slots:
            if slot in (0, top) and (not endpoint_visits or endpoint_visits[-1] != slot):
                endpoint_visits.append(int(slot))
        for first, middle, last in zip(endpoint_visits, endpoint_visits[1:], endpoint_visits[2:]):
            if (first, middle, last) == (0, top, 0):
                counts[walker, 0] += 1
            elif (first, middle, last) == (top, 0, top):
                counts[walker, 1] += 1

    per_walker = counts.sum(axis=1)
    slot_coverage = visited_slot_counts.astype(float) / n_replicas
    return {
        'walker_slots': walker_slots,
        'directional_counts': counts,
        'per_walker': per_walker,
        'total': int(per_walker.sum()),
        'walkers_with_round_trip': int(np.count_nonzero(per_walker)),
        'visited_slot_counts': visited_slot_counts,
        'slot_coverage': slot_coverage,
        'visited_both_ends': visited_both_ends,
        'walkers_visiting_both_ends': int(np.count_nonzero(visited_both_ends)),
        'occupancy_uniform_rmse': occupancy_uniform_rmse,
        'mean_occupancy_uniform_rmse': float(np.mean(occupancy_uniform_rmse)),
    }


def log_round_trip_summary(summary, logger):
    logger.info("  Walker mixing (round trips: 0->top->0 / top->0->top):")
    for walker, (up_down, down_up) in enumerate(summary['directional_counts']):
        logger.info(f"    Walker {walker:2d}: slots={summary['visited_slot_counts'][walker]:d}, "
                    f"coverage={summary['slot_coverage'][walker]:.1%}, "
                    f"both_ends={bool(summary['visited_both_ends'][walker])}, "
                    f"round_trips={up_down:d}/{down_up:d}, "
                    f"occupancy_RMSE={summary['occupancy_uniform_rmse'][walker]:.3f}")
    logger.info(f"  Walkers visiting both ends: {summary['walkers_visiting_both_ends']}/"
                f"{len(summary['per_walker'])}; mean relative uniform occupancy RMSE: "
                f"{summary['mean_occupancy_uniform_rmse']:.3f}")
    if summary['total'] == 0:
        logger.info("  Round-trip: 未观察到完整往返；需结合生产时长评价，短预实验不据此判错")
    else:
        logger.info(f"  Round-trip total: {summary['total']}; walkers with >=1: "
                    f"{summary['walkers_with_round_trip']}/{len(summary['per_walker'])}")


def evaluate_experiment_quality(cfg, results, rg_results=None):
    """用独立维度评价实验质量；不以单一总分掩盖任一风险。"""
    production_ns = (len(results.get('energies', [])) * cfg['n_steps_per_move'] *
                     cfg['timestep'] / 1.0e6)

    energies = np.asarray(results.get('energies', []), dtype=float)
    finite = bool(energies.ndim == 2 and energies.size and np.all(np.isfinite(energies)))
    drift_ratios = []
    if finite:
        window = max(1, int(np.ceil(energies.shape[0] * 0.2)))
        for slot in range(energies.shape[1]):
            values = energies[:, slot]
            std = np.std(values)
            difference = np.mean(values[-window:]) - np.mean(values[:window])
            drift_ratios.append(float(abs(difference) / std) if std > 0
                                else (0.0 if difference == 0 else float('inf')))
    max_drift = max(drift_ratios) if drift_ratios else None
    if not finite:
        numerical_status = 'fail'
        numerical_reason = '能量包含非有限值或没有可评价的能量数据'
    elif max_drift <= 1.0:
        numerical_status = 'pass'
        numerical_reason = '所有能量有限，最大首末20%均值漂移/全程标准差不超过1'
    elif max_drift <= 2.0:
        numerical_status = 'warning'
        numerical_reason = '所有能量有限，但至少一个槽位的漂移筛查比值位于1到2'
    else:
        numerical_status = 'fail'
        numerical_reason = '至少一个槽位的漂移筛查比值大于2'
    numerical = {
        'status': numerical_status,
        'reason': numerical_reason,
        'all_energies_finite': finite,
        'per_slot_drift_ratio': drift_ratios,
        'max_drift_ratio': max_drift,
        'note': '首末20%均值差/全程标准差仅为漂移筛查，不是严格平衡或收敛检验',
    }

    attempts = np.asarray(results.get('exchange_attempts', []), dtype=int)
    accepts = np.asarray(results.get('exchange_accepts', []), dtype=int)
    rates = np.full(attempts.shape, np.nan, dtype=float)
    attempted = attempts > 0
    rates[attempted] = accepts[attempted] / attempts[attempted] * 100.0
    no_breaks = bool(rates.size and np.all(np.isfinite(rates)))
    finite_rates = rates[np.isfinite(rates)]
    if not no_breaks:
        exchange_status = 'fail'
        exchange_reason = '存在无交换尝试的相邻pair断点或没有交换数据'
    elif np.min(finite_rates) >= 20.0:
        exchange_status = 'pass'
        exchange_reason = '全部相邻交换率不低于20%，无断点；40–55%属于良好范围'
    elif np.min(finite_rates) >= 10.0:
        exchange_status = 'warning'
        exchange_reason = '最低相邻交换率位于10–20%'
    else:
        exchange_status = 'fail'
        exchange_reason = '最低相邻交换率低于10%'
    lowest_pair = int(np.nanargmin(rates)) if finite_rates.size else None
    exchange = {
        'status': exchange_status,
        'reason': exchange_reason,
        'all_adjacent_rates_percent': [float(v) if np.isfinite(v) else None for v in rates],
        'min_percent': float(np.min(finite_rates)) if finite_rates.size else None,
        'mean_percent': float(np.mean(finite_rates)) if finite_rates.size else None,
        'max_percent': float(np.max(finite_rates)) if finite_rates.size else None,
        'lowest_pair': ([lowest_pair, lowest_pair + 1] if lowest_pair is not None else None),
        'has_breakpoint': not no_breaks,
    }

    ca_values, hot_values = [], []
    for record in results.get('rmsd_history', []) or []:
        if not isinstance(record, dict):
            continue
        ca_values.extend(float(v) for v in record.get('rmsd', {}).values() if np.isfinite(v))
        hot_values.extend(float(v) for v in record.get('hot_heavy_rmsd', {}).values() if np.isfinite(v))

    def rmsd_component(values, warning_limit, fail_limit, label):
        if not values:
            return {'status': 'not_assessed', 'reason': f'没有{label}历史数据', 'max_A': None}
        maximum = max(values)
        if maximum < warning_limit:
            status = 'pass'
            reason = f'全部记录的{label}均低于{warning_limit:g} Å'
        elif maximum <= fail_limit:
            status = 'warning'
            reason = f'{label}最大值位于{warning_limit:g}–{fail_limit:g} Å'
        else:
            status = 'fail'
            reason = f'{label}最大值高于{fail_limit:g} Å'
        return {'status': status, 'reason': reason, 'max_A': float(maximum)}

    ca_safety = rmsd_component(ca_values, 4.0, 6.0, 'CA RMSD')
    hot_safety = rmsd_component(hot_values, 3.0, 5.0, 'hot-heavy RMSD')
    assessed_structural = [item for item in (ca_safety, hot_safety)
                           if item['status'] != 'not_assessed']
    structural_rank = {'pass': 0, 'warning': 1, 'fail': 2}
    if not assessed_structural:
        structural_status = 'not_assessed'
        structural_reason = '没有CA或hot-heavy RMSD历史数据'
    else:
        structural_status = max(assessed_structural,
                                key=lambda item: structural_rank[item['status']])['status']
        structural_reason = '; '.join(item['reason'] for item in (ca_safety, hot_safety))
    structural = {
        'status': structural_status,
        'reason': structural_reason,
        'ca': ca_safety,
        'hot_heavy': hot_safety,
        'evaluated_over_all_recorded_frames': True,
    }

    walker_history = results.get('walker_label_history')
    if walker_history is not None and np.asarray(walker_history).size:
        mixing = analyze_walker_round_trips(walker_history, len(results['temperatures']))
        walkers_with_rt = mixing['walkers_with_round_trip']
        if walkers_with_rt:
            walker_status = 'pass'
            walker_reason = f'{walkers_with_rt}个walker观察到完整round trip'
        elif production_ns < 10.0:
            walker_status = 'insufficient_duration'
            walker_reason = '总生产时间短于10 ns，0 round-trip仅表示时长不足'
        elif production_ns < 50.0:
            walker_status = 'warning'
            walker_reason = '10–50 ns生产中尚未观察到round-trip'
        else:
            walker_status = 'fail'
            walker_reason = '至少50 ns生产后仍未观察到round-trip'
        walker = {
            'status': walker_status,
            'reason': walker_reason,
            'visited_slot_counts': mixing['visited_slot_counts'].astype(int).tolist(),
            'slot_coverage': mixing['slot_coverage'].astype(float).tolist(),
            'walkers_visiting_both_ends': mixing['walkers_visiting_both_ends'],
            'visited_both_ends': mixing['visited_both_ends'].astype(bool).tolist(),
            'round_trips_per_walker': mixing['per_walker'].astype(int).tolist(),
            'occupancy_uniform_relative_rmse': mixing['occupancy_uniform_rmse'].astype(float).tolist(),
            'mean_occupancy_uniform_relative_rmse': mixing['mean_occupancy_uniform_rmse'],
        }
    else:
        walkers_with_rt = int(results.get('walkers_with_round_trip', 0))
        walker = {'status': 'not_assessed', 'reason': '没有slot_to_walker历史，无法评价walker混合'}

    fraction_with_rt = walkers_with_rt / max(1, len(results.get('temperatures', [])))
    if production_ns < 10.0:
        maturity_status = 'pilot_only'
        maturity_reason = '生产时间短于10 ns，只能视为预实验，不能判为收敛'
    elif production_ns < 50.0:
        maturity_status = 'preliminary'
        maturity_reason = '生产时间为10–50 ns，采样结论仍属初步'
    elif fraction_with_rt >= 0.75:
        maturity_status = 'production_mixed'
        maturity_reason = '生产时间至少50 ns，且至少75%的walker有round-trip'
    elif walkers_with_rt > 0:
        maturity_status = 'production_sampled'
        maturity_reason = '生产时间至少50 ns，并观察到round-trip，但充分混合的walker不足75%'
    else:
        maturity_status = 'preliminary'
        maturity_reason = '生产时间至少50 ns，但未观察到round-trip，采样分析仍属初步'
    maturity = {'status': maturity_status, 'reason': maturity_reason,
                'production_ns_per_replica': float(production_ns),
                'fraction_walkers_with_round_trip': float(fraction_with_rt)}

    if not rg_results:
        rg = {'status': 'not_assessed', 'reason': '没有Rg分析结果', 'limitations': [
            '各Hamiltonian槽位轨迹并非独立重复实验', '时间序列存在自相关']}
    else:
        indices = sorted(rg_results)
        means = np.asarray([rg_results[i]['rg_mean'] for i in indices], dtype=float)
        stds = np.asarray([rg_results[i]['rg_std'] for i in indices], dtype=float)
        frames = [int(rg_results[i].get('n_frames', 0)) for i in indices]
        temperatures = np.asarray(results['temperatures'], dtype=float)[indices]
        endpoint_difference = float(means[-1] - means[0]) if len(means) >= 2 else None
        pooled_std = (float(np.sqrt((stds[0] ** 2 + stds[-1] ** 2) / 2.0))
                      if len(stds) >= 2 else None)
        effect_size = (endpoint_difference / pooled_std
                       if pooled_std is not None and pooled_std > 0 else None)
        if len(means) >= 2 and np.ptp(temperatures) > 0:
            slope, intercept = np.polyfit(temperatures, means, 1)
            fitted = slope * temperatures + intercept
            ss_total = np.sum((means - np.mean(means)) ** 2)
            r_squared = float(1.0 - np.sum((means - fitted) ** 2) / ss_total) if ss_total > 0 else 0.0
            slope = float(slope)
        else:
            slope, r_squared = None, None
        min_frames = min(frames) if frames else 0
        exploratory = production_ns < 10.0 or min_frames < 200
        rg = {
            'status': 'exploratory_only' if exploratory else 'descriptive',
            'reason': ('生产时间短于10 ns或至少一个槽位少于200帧，仅可作探索性描述，不得声称显著'
                       if exploratory else '时长和帧数达到描述性分析门槛；不进行p值或显著性推断'),
            'slot_count': len(indices),
            'minimum_frames_per_slot': int(min_frames),
            'endpoint_mean_difference_A': endpoint_difference,
            'endpoint_effect_size_pooled_sd': (float(effect_size) if effect_size is not None else None),
            'linear_slope_A_per_K': slope,
            'linear_r_squared': r_squared,
            'limitations': ['Rg时间序列存在时间自相关', '不同Hamiltonian槽位不是独立重复实验'],
        }

    fatal_issues = []
    if not finite:
        fatal_issues.append('non_finite_energy')
    core_pass = all(item['status'] == 'pass' for item in (numerical, exchange, structural))
    readiness_status = 'ready' if core_pass and not fatal_issues else 'not_ready'
    readiness = {
        'status': readiness_status,
        'reason': ('数值稳定性、交换质量和结构安全均通过，且没有fatal issue；短预实验中的walker/Rg未成熟不阻止进入生产'
                   if readiness_status == 'ready' else
                   '数值稳定性、交换质量、结构安全未全部通过，或存在fatal issue'),
        'fatal_issues': fatal_issues,
    }
    if maturity_status == 'pilot_only':
        overall_stage = ('pilot_completed_ready_for_production' if readiness_status == 'ready'
                         else 'pilot_completed_not_ready_for_production')
    elif maturity_status == 'production_mixed' and rg['status'] == 'descriptive':
        overall_stage = 'production_mixed_analysis_ready'
    elif maturity_status == 'production_mixed':
        overall_stage = 'production_mixed_analysis_preliminary'
    elif production_ns >= 50.0:
        overall_stage = 'production_completed_analysis_preliminary'
    else:
        overall_stage = 'preliminary_sampling_analysis_preliminary'

    return {
        'numerical_stability': numerical,
        'exchange_quality': exchange,
        'structural_safety': structural,
        'walker_mixing': walker,
        'sampling_maturity': maturity,
        'rg_evidence': rg,
        'production_readiness': readiness,
        'overall_stage': overall_stage,
    }


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def input_file_hashes(cfg):
    hashes = {}
    for key in ('psf_file', 'pdb_file', 'toppar_str', 'boxinfo_str'):
        path = cfg.get(key)
        if path and os.path.isfile(path):
            hashes[key] = sha256_file(path)
    return hashes


def checkpoint_scalar(data, key):
    value = data[key]
    return value.item() if np.asarray(value).shape == () else value.tolist()


def save_full_checkpoint(cfg, iteration, contexts, energy_history, exchange_attempts,
                         exchange_accepts, temps, lams, delta_history, logger, rng,
                         rmsd_history=None, acceptance_history=None,
                         walker_labels=None, walker_label_history=None,
                         walker_history_start_iteration=0):
    ckpt_file = os.path.join(cfg["output_dir"], f"checkpoint_{iteration}_full.npz")

    positions_list = []
    velocities_list = []
    box_vectors_list = []

    for ctx in contexts:
        state = ctx.getState(getPositions=True, getVelocities=True)
        pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        vel = state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer/unit.picosecond)
        box = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
        positions_list.append(np.asarray(pos))
        velocities_list.append(np.asarray(vel))
        box_vectors_list.append(np.asarray(box))

    lambda_min = cfg["T_ref"] / cfg["T_max"]
    save_dict = {
        'iteration': iteration,
        'energies': np.array(energy_history),
        'attempts': exchange_attempts,
        'accepts': exchange_accepts,
        'temps': temps,
        'lams': lams,
        'lambda_hot': cfg["lambda_hot"],
        'local_heating_enabled': cfg["local_heating_enabled"],
        'local_scaling_scheme': cfg["local_scaling_scheme"],
        'reference_state_unbiased': True,
        'local_multiplier_endpoint': cfg["lambda_hot"],
        'lambda_min': lambda_min,
        'hot_effective_temperature_range_K': [cfg["T_ref"], cfg["T_max"] / cfg["lambda_hot"]],
        'hot_residue_ids': cfg["hot_residue_ids"],
        'hot_residue_names': cfg["hot_residue_names"],
        'T_ref': cfg["T_ref"],
        'T_max': cfg["T_max"],
        'n_replicas': len(temps),
        'n_atoms': contexts[0].getSystem().getNumParticles(),
        'checkpoint_version': __version__,
        'psf_file': cfg["psf_file"],
        'pdb_file': cfg["pdb_file"],
        'toppar_str': cfg["toppar_str"],
        'boxinfo_str': cfg["boxinfo_str"],
        'cmap_scaling_mode': cfg["cmap_scaling_mode"],
        'periodic_boundary': cfg["periodic_boundary"],
        'use_cubic_box': cfg["use_cubic_box"],
        'timestep': cfg["timestep"],
        'friction': cfg["friction"],
        'random_seed': cfg["random_seed"],
        'numpy_rng_state_json': json.dumps(rng.bit_generator.state),
        'walker_history_start_iteration': walker_history_start_iteration,
        'walker_label_semantics': 'slot_to_walker',
    }
    for key, file_hash in input_file_hashes(cfg).items():
        save_dict[f'{key}_sha256'] = file_hash

    if acceptance_history is not None:
        for k, values in acceptance_history.items():
            save_dict[f"accepted_{k}"] = np.asarray(values, dtype=bool)

    if rmsd_history is not None:
        save_dict['rmsd_history'] = np.array(rmsd_history)
    if walker_labels is not None:
        save_dict['walker_labels'] = np.asarray(walker_labels, dtype=int)
    if walker_label_history is not None:
        save_dict['walker_label_history'] = np.asarray(walker_label_history, dtype=int)

    save_dict['coordinate_unit'] = 'nanometer'
    for i, (pos, vel, box) in enumerate(zip(positions_list, velocities_list, box_vectors_list)):
        save_dict[f'positions_{i}'] = pos
        save_dict[f'velocities_{i}'] = vel
        save_dict[f'box_vectors_{i}'] = box

    for k, v in delta_history.items():
        if v:
            save_dict[f"delta_{k}"] = np.array(v)

    np.savez_compressed(ckpt_file, **save_dict)
    logger.info(f"  Full checkpoint saved: {ckpt_file}")

    return ckpt_file


def restore_from_checkpoint(checkpoint_file, cfg, logger):
    logger.info(f"\n=== Restoring from Checkpoint ===")
    logger.info(f"  Loading: {checkpoint_file}")

    data = np.load(checkpoint_file, allow_pickle=True)
    required_common = {'iteration', 'energies', 'attempts', 'accepts', 'temps', 'lams'}
    missing = sorted(required_common - set(data.files))
    if missing:
        raise ValueError("仅支持 full checkpoint 恢复；checkpoint 缺少必需键: " + ", ".join(missing))

    start_iteration = int(data['iteration'])
    energy_history = list(data['energies'])
    exchange_attempts = np.asarray(data['attempts'], dtype=int)
    exchange_accepts = np.asarray(data['accepts'], dtype=int)
    temps = np.asarray(data['temps'], dtype=float)
    lams = np.asarray(data['lams'], dtype=float)
    n_replicas = len(temps)
    required_state = [f'{kind}_{i}' for i in range(n_replicas)
                      for kind in ('positions', 'velocities', 'box_vectors')]
    missing_state = [key for key in required_state if key not in data]
    if missing_state:
        raise ValueError(
            "仅支持 full checkpoint 恢复；缺少位置/速度/周期盒状态键: "
            + ", ".join(missing_state[:6])
            + ("..." if len(missing_state) > 6 else "")
        )

    if len(lams) != n_replicas or len(exchange_attempts) != n_replicas - 1:
        raise ValueError("checkpoint 的副本温度、lambda 或交换数组长度不一致")

    delta_history = {k: [] for k in range(n_replicas - 1)}
    acceptance_history = {k: [] for k in range(n_replicas - 1)}
    for k in range(n_replicas - 1):
        if f'delta_{k}' in data:
            delta_history[k] = list(data[f'delta_{k}'])
        if f'accepted_{k}' in data:
            acceptance_history[k] = [bool(v) for v in data[f'accepted_{k}']]
    if not any(acceptance_history.values()):
        logger.warning("旧 checkpoint 无逐次 accepted 历史；累积接受率曲线将从恢复后开始")

    rmsd_history = list(data['rmsd_history']) if 'rmsd_history' in data else []
    history_array = (np.asarray(data['walker_label_history'], dtype=int)
                     if 'walker_label_history' in data else None)
    if 'walker_labels' in data:
        walker_labels = np.asarray(data['walker_labels'], dtype=int)
    elif history_array is not None and history_array.ndim == 2 and len(history_array):
        walker_labels = history_array[-1].copy()
        logger.warning("旧 checkpoint 无 walker_labels；已从 walker_label_history 最后一帧恢复")
    else:
        walker_labels = np.arange(n_replicas, dtype=int)
        logger.warning("旧 checkpoint 无 walker 标签字段；从恢复点按当前槽位编号重新初始化")
    if walker_labels.shape != (n_replicas,) or sorted(walker_labels.tolist()) != list(range(n_replicas)):
        raise ValueError("checkpoint 的 walker_labels 不是有效副本标签排列")

    if history_array is not None:
        if history_array.ndim != 2 or history_array.shape[1] != n_replicas:
            raise ValueError("checkpoint 的 walker_label_history 形状无效")
        for row in history_array:
            if sorted(row.tolist()) != list(range(n_replicas)):
                raise ValueError("checkpoint 的 walker_label_history 含无效标签排列")
        walker_label_history = [row.copy() for row in history_array]
        if 'walker_history_start_iteration' in data:
            walker_history_start_iteration = int(data['walker_history_start_iteration'])
        else:
            walker_history_start_iteration = start_iteration - len(walker_label_history) + 1
            logger.warning("旧 checkpoint 无 walker_history_start_iteration；已按历史长度推断为 %d",
                           walker_history_start_iteration)
        expected_length = start_iteration - walker_history_start_iteration + 1
        if expected_length != len(walker_label_history):
            raise ValueError("checkpoint 的 walker 历史长度与 iteration/start 元数据不自洽")
        if not np.array_equal(walker_label_history[-1], walker_labels):
            raise ValueError("checkpoint 的 walker_labels 与历史最后一帧不一致")
        if 'walker_label_semantics' not in data:
            logger.warning("旧 checkpoint 无 walker_label_semantics；按槽位->walker 标签语义解释")
    else:
        walker_label_history = [walker_labels.copy()]
        walker_history_start_iteration = start_iteration
        logger.warning("旧 checkpoint 无 walker_label_history；从恢复迭代重新开始 round-trip 历史")
    positions_list = [data[f'positions_{i}'] for i in range(n_replicas)]
    velocities_list = [data[f'velocities_{i}'] for i in range(n_replicas)]
    box_vectors_list = [data[f'box_vectors_{i}'] for i in range(n_replicas)]
    atom_counts = {len(p) for p in positions_list}
    if len(atom_counts) != 1:
        raise ValueError("checkpoint 各副本原子数不一致")

    metadata_keys = (
        'checkpoint_version', 'local_heating_enabled', 'lambda_hot', 'local_scaling_scheme',
        'reference_state_unbiased', 'local_multiplier_endpoint', 'lambda_min',
        'hot_effective_temperature_range_K', 'hot_residue_ids', 'hot_residue_names',
        'T_ref', 'T_max', 'n_replicas', 'n_atoms', 'psf_file', 'pdb_file',
        'toppar_str', 'boxinfo_str', 'cmap_scaling_mode', 'periodic_boundary',
        'use_cubic_box', 'timestep', 'friction', 'random_seed'
    )
    checkpoint_cfg = {key: checkpoint_scalar(data, key) for key in metadata_keys if key in data}
    is_v44_checkpoint = str(checkpoint_cfg.get('checkpoint_version', '')).startswith('4.4')
    required_v44 = {
        'local_heating_enabled', 'lambda_hot', 'local_scaling_scheme',
        'reference_state_unbiased', 'local_multiplier_endpoint', 'lambda_min',
        'hot_effective_temperature_range_K', 'hot_residue_ids', 'hot_residue_names', 'T_ref', 'T_max',
        'n_replicas', 'n_atoms', 'toppar_str', 'cmap_scaling_mode',
        'periodic_boundary', 'use_cubic_box', 'timestep', 'friction', 'random_seed'
    }
    hash_keys = ('psf_file', 'pdb_file', 'toppar_str', 'boxinfo_str')
    configured_inputs = {key: cfg.get(key) for key in hash_keys if cfg.get(key)}
    missing_current_inputs = sorted(
        key for key, path in configured_inputs.items() if not os.path.isfile(path)
    )
    required_hashes = {f'{key}_sha256' for key in configured_inputs}
    missing_metadata = sorted(required_v44 - set(checkpoint_cfg))
    missing_hashes = sorted(required_hashes - set(data.files))
    if is_v44_checkpoint and missing_current_inputs:
        raise ValueError(
            "v4.4 checkpoint 恢复所需的当前输入文件不存在: " +
            ", ".join(missing_current_inputs)
        )
    if is_v44_checkpoint and (missing_metadata or missing_hashes):
        raise ValueError("v4.4 checkpoint 缺少严格恢复元数据: " +
                         ", ".join(missing_metadata + missing_hashes))
    if not is_v44_checkpoint:
        logger.warning("旧 checkpoint 缺少 v4.4 严格一致性元数据时仅校验已有字段；OpenMM NPZ 恢复并非 bitwise 可复现")

    critical_cfg = (
        'local_heating_enabled', 'lambda_hot', 'local_scaling_scheme',
        'hot_residue_ids', 'hot_residue_names', 'T_ref', 'T_max',
        'n_replicas', 'cmap_scaling_mode', 'periodic_boundary', 'use_cubic_box',
        'timestep', 'friction', 'random_seed'
    )
    if is_v44_checkpoint:
        for key in critical_cfg:
            current = cfg.get(key)
            saved = checkpoint_cfg[key]
            if isinstance(current, (list, tuple, np.ndarray)):
                matches = np.array_equal(np.asarray(current), np.asarray(saved))
            else:
                matches = current == saved
            if not matches:
                raise ValueError(f"v4.4 checkpoint 与当前关键配置不一致: {key}: checkpoint={saved!r}, current={current!r}")
        expected_semantics = {
            'reference_state_unbiased': True,
            'local_multiplier_endpoint': cfg['lambda_hot'],
            'lambda_min': cfg['T_ref'] / cfg['T_max'],
            'hot_effective_temperature_range_K': [cfg['T_ref'], cfg['T_max'] / cfg['lambda_hot']],
        }
        for key, expected in expected_semantics.items():
            saved = checkpoint_cfg[key]
            if isinstance(expected, list):
                matches = np.allclose(np.asarray(saved, dtype=float), np.asarray(expected, dtype=float))
            elif isinstance(expected, float):
                matches = np.isclose(float(saved), expected)
            else:
                matches = saved == expected
            if not matches:
                raise ValueError(f"v4.4 checkpoint 方案语义不一致: {key}: checkpoint={saved!r}, current={expected!r}")
        if checkpoint_cfg['n_atoms'] != next(iter(atom_counts)):
            raise ValueError("v4.4 checkpoint 的 n_atoms 元数据与坐标不一致")
        current_hashes = input_file_hashes(cfg)
        for key, current_hash in current_hashes.items():
            saved_hash = str(checkpoint_scalar(data, f'{key}_sha256'))
            if saved_hash != current_hash:
                raise ValueError(f"v4.4 checkpoint 输入文件 SHA-256 不一致: {key}")

    if 'numpy_rng_state_json' in data:
        rng_state = json.loads(str(checkpoint_scalar(data, 'numpy_rng_state_json')))
    else:
        rng_state = None
        logger.warning("旧 checkpoint 无 NumPy Generator 状态；交换随机序列将从配置 random_seed 重新开始")

    coordinate_unit = str(data['coordinate_unit']) if 'coordinate_unit' in data else 'legacy_nanometer'
    logger.info(f"  Checkpoint coordinate unit: {coordinate_unit}")

    logger.info(f"  Restored iteration: {start_iteration}")
    logger.info(f"  Energy history length: {len(energy_history)}")
    logger.info(f"  Total replicas: {n_replicas}")
    if rmsd_history:
        logger.info(f"  RMSD history length: {len(rmsd_history)}")

    return {
        'start_iteration': start_iteration,
        'energy_history': energy_history,
        'exchange_attempts': exchange_attempts,
        'exchange_accepts': exchange_accepts,
        'temps': temps,
        'lams': lams,
        'delta_history': delta_history,
        'acceptance_history': acceptance_history,
        'checkpoint_cfg': checkpoint_cfg,
        'rng_state': rng_state,
        'positions_list': positions_list,
        'velocities_list': velocities_list,
        'box_vectors_list': box_vectors_list,
        'coordinate_unit': coordinate_unit,
        'rmsd_history': rmsd_history,
        'walker_labels': walker_labels,
        'walker_label_history': walker_label_history,
        'walker_history_start_iteration': walker_history_start_iteration,
    }


def verify_dcd_file(dcd_file, logger):
    """验证DCD文件是否正确生成并包含周期性盒子信息"""
    try:
        import MDAnalysis as mda

        u = mda.Universe(dcd_file)
        n_frames = len(u.trajectory)
        logger.info(f"[v4.4] DCD verification: {dcd_file}")
        logger.info(f"[v4.4]   Frames: {n_frames}")

        if hasattr(u.trajectory, 'dimensions'):
            dims = u.trajectory.dimensions
            if dims is not None:
                logger.info(f"[v4.4]   Box dimensions: {dims[:3]} Å")
                logger.info(f"[v4.4]   Box angles: {dims[3:]} degrees")
                logger.info(f"[v4.4]   Periodic boundary: VERIFIED")
            else:
                logger.warning(f"[v4.4]   Box dimensions: NOT FOUND in DCD file")

        return True

    except ImportError:
        logger.warning("[v4.4] MDAnalysis not available for DCD verification")
        return False
    except Exception as e:
        logger.error(f"[v4.4] DCD verification failed: {e}")
        return False


def run_rest2(cfg, logger, restore_data=None):
    kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
    kB_val = kB.value_in_unit(unit.kilojoule_per_mole / unit.kelvin)
    rng = np.random.default_rng(cfg["random_seed"])
    if restore_data and restore_data.get('rng_state') is not None:
        rng.bit_generator.state = restore_data['rng_state']
        logger.info("NumPy Generator 状态已从 checkpoint 恢复；OpenMM NPZ 状态恢复并非 bitwise 可复现")
    seed_sequence = np.random.SeedSequence(cfg["random_seed"])
    openmm_seeds = [int(child.generate_state(1, dtype=np.uint32)[0] % 2147483646) + 1
                    for child in seed_sequence.spawn(3 * cfg["n_replicas"] + 2)]

    logger.info(f"\n{'='*70}")
    mode = "REST2 + non-standard local heating" if cfg["local_heating_enabled"] else "standard REST2"
    exchange_ps = cfg["n_steps_per_move"] * cfg["timestep"] / 1000.0
    production_ns = cfg["n_iterations"] * exchange_ps / 1000.0
    logger.info(f"REST2 Simulation [v{cfg['version']}] - {mode}")
    logger.info(f"Replicas: {cfg['n_replicas']}")
    logger.info(f"Effective solute temperature range: {cfg['T_ref']:.1f}-{cfg['T_max']:.1f} K")
    logger.info(f"Physical thermostat temperature: {cfg['T_ref']:.1f} K for every replica")
    logger.info(f"Exchange interval: {cfg['n_steps_per_move']} steps ({exchange_ps:.3f} ps)")
    logger.info(f"Production length: {cfg['n_iterations']} iterations ({production_ns:.3f} ns per replica)")
    if cfg["local_heating_enabled"]:
        logger.warning("NON-STANDARD local-hot REST2 extension is active")
        logger.info(f"Hot residues: {cfg['hot_residue_ids']}")
        logger.info(f"Lambda_hot endpoint multiplier: {cfg['lambda_hot']}")
        logger.info(f"Local scaling scheme: {cfg['local_scaling_scheme']}")
        logger.info(f"Reference state unbiased: True")
        logger.info(f"Local-hot effective temperature range: {cfg['T_ref']:.1f}-"
                    f"{cfg['T_max']/cfg['lambda_hot']:.1f} K")
    logger.info(f"[v4.4] Periodic boundary: {cfg.get('periodic_boundary', True)}")
    logger.info(f"[v4.4] Preserve input box shape: {not cfg.get('use_cubic_box', False)}")
    logger.info(f"[v4.4] Equilibration restraints: {cfg.get('equilibration_restraints', False)}")
    logger.info(f"[v4.4] NPT/NVT/unrestrained equilibration steps: "
                f"{cfg['npt_equilibration_steps']}/{cfg['nvt_equilibration_steps']}/"
                f"{cfg['unrestrained_equilibration_steps']}")
    if cfg.get("save_trajectory", False):
        trajectory_ps = exchange_ps * cfg["trajectory_interval"]
        logger.info(f"Trajectory interval: {cfg['trajectory_interval']} iterations ({trajectory_ps:.3f} ps)")
    logger.info(f"Checkpoint interval: {cfg['checkpoint_interval']} iterations")
    logger.info(f"[v4.4] RMSD monitoring: every {cfg.get('rmsd_monitor_interval', 100)} iterations")
    logger.info(f"{'='*70}")

    base_system, psf, pdb, box_size, box_dims = build_base_system(cfg, logger)

    detected_ions = detect_ions(pdb.topology, logger)
    if detected_ions:
        global SOLVENT_RESNAMES
        SOLVENT_RESNAMES.update(detected_ions)
        logger.info(f"[v4.4] Added {len(detected_ions)} ion types to solvent residues")

    solute_atoms = identify_solute_atoms(pdb.topology, logger)

    if cfg["local_heating_enabled"]:
        hot_atoms, hot_atoms_by_residue = identify_hot_residue_atoms(
            pdb.topology, solute_atoms, cfg, logger)
    else:
        hot_atoms = set()
        hot_atoms_by_residue = {}
        logger.info("[v4.4] Local heating DISABLED")

    # 识别全蛋白 CA 与 local-hot 五残基重原子，用于 Hamiltonian slot RMSD 监控。
    ca_atom_indices = identify_ca_atoms(pdb.topology, logger)
    hot_heavy_atom_indices = identify_hot_heavy_atoms(pdb.topology, hot_atoms, logger)

    initial_ca_coords = extract_ca_positions(pdb.positions, ca_atom_indices)
    initial_hot_heavy_coords = (extract_ca_positions(pdb.positions, hot_heavy_atom_indices)
                                if hot_heavy_atom_indices else np.empty((0, 3)))
    logger.info(f"[v4.4] Saved initial CA coordinates for RMSD reference ({len(initial_ca_coords)} atoms)")
    logger.info(f"[v4.4] Saved initial local-hot heavy-atom coordinates for RMSD reference "
                f"({len(initial_hot_heavy_coords)} atoms)")

    n_rep = cfg["n_replicas"]
    T_ref = cfg["T_ref"]
    T_max = cfg["T_max"]
    lam_hot = cfg["lambda_hot"]
    lambda_min = T_ref / T_max

    if restore_data:
        temps = list(restore_data['temps'])
        lams = list(restore_data['lams'])
        start_iteration = restore_data['start_iteration']
        energy_history = restore_data['energy_history']
        exchange_attempts = restore_data['exchange_attempts']
        exchange_accepts = restore_data['exchange_accepts']
        delta_history = restore_data['delta_history']
        acceptance_history = restore_data['acceptance_history']
        rmsd_history = restore_data.get('rmsd_history', [])
        walker_labels = restore_data['walker_labels'].copy()
        walker_label_history = [np.asarray(labels, dtype=int).copy()
                                for labels in restore_data['walker_label_history']]
        walker_history_start_iteration = restore_data['walker_history_start_iteration']
    else:
        temps, lams = optimize_temperatures_geometric(n_rep, T_ref, T_max, logger)
        start_iteration = 0
        energy_history = []
        exchange_attempts = np.zeros(n_rep-1, dtype=int)
        exchange_accepts = np.zeros(n_rep-1, dtype=int)
        delta_history = {k: [] for k in range(n_rep-1)}
        acceptance_history = {k: [] for k in range(n_rep-1)}
        rmsd_history = []
        # walker_labels[slot] 是当前 Hamiltonian 槽位中的 walker 标签。
        walker_labels = np.arange(n_rep, dtype=int)
        walker_label_history = [walker_labels.copy()]
        walker_history_start_iteration = 0

    mode_name = "REST2 + non-standard local-hot extension" if cfg["local_heating_enabled"] else "standard REST2"
    logger.info(f"\n=== Creating Replica Systems: {mode_name} ===")
    replica_systems = []

    # P0-3修复: 在Context创建前将位置约束力添加到System中
    restraint_force_indices = []  # 记录约束力在每个system中的index
    if cfg.get("equilibration_restraints", False) and not restore_data:
        logger.info(f"[v4.4] Position restraints enabled (k={cfg['restraint_force_constant']} kJ/mol/nm²)")
        logger.info(f"[v4.4] Restraints will be applied to protein CA atoms during equilibration, then removed")

    for i, (T, lam_global) in enumerate(zip(temps, lams)):
        if not restore_data or i < 5:
            logger.info(f"Replica {i}: T={T:.1f}K, λ_global={lam_global:.4f}")

        if cfg["local_heating_enabled"]:
            rep_sys, cmap_replacements = create_rest2_system_with_local_heating(
                base_system, solute_atoms, hot_atoms, lam_global, lambda_min, lam_hot,
                cfg, logger
            )
        else:
            rep_sys, cmap_replacements = create_rest2_system_with_local_heating(
                base_system, solute_atoms, set(), lam_global, lambda_min, 1.0,
                cfg, logger
            )

        if cmap_replacements:
            rep_sys = rebuild_system_with_scaled_cmap(rep_sys, cmap_replacements, logger)

        # P0-3修复: 在Context创建前添加位置约束力到System
        if cfg.get("equilibration_restraints", False) and not restore_data:
            restraint = openmm.CustomExternalForce("0.5*k*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
            restraint.addGlobalParameter("k", cfg["restraint_force_constant"] * unit.kilojoule_per_mole/unit.nanometer**2)
            restraint.addPerParticleParameter("x0")
            restraint.addPerParticleParameter("y0")
            restraint.addPerParticleParameter("z0")
            # 对蛋白CA原子添加约束
            for atom in pdb.topology.atoms():
                if atom.name == 'CA' and atom.residue.name in AA_RESNAMES:
                    p = pdb.positions[atom.index]
                    restraint.addParticle(atom.index, [p.x, p.y, p.z])
            restraint_idx = rep_sys.addForce(restraint)
            restraint_force_indices.append(restraint_idx)
            if i == 0:
                logger.info(f"[v4.4] Added position restraint force ({restraint.getNumParticles()} CA atoms) to replica systems")
        else:
            restraint_force_indices.append(-1)

        replica_systems.append(rep_sys)

    platform, plat_name = select_platform(cfg["platform_preference"], logger)

    logger.info(f"\n=== Initializing Replica Contexts ===")
    contexts = []

    for i, (rep_sys, T) in enumerate(zip(replica_systems, temps)):
        # REST2 的动力学热浴对所有副本固定在参考温度；T 仅是有效溶质温度标签。
        integ = openmm.LangevinMiddleIntegrator(
            T_ref*unit.kelvin, cfg["friction"]/unit.picosecond, cfg["timestep"]*unit.femtoseconds
        )
        integ.setRandomNumberSeed(openmm_seeds[i])
        props = cuda_device_props(i, logger)
        if props is None:
            ctx = openmm.Context(rep_sys, integ, platform)
        else:
            ctx = openmm.Context(rep_sys, integ, platform, props)

        if restore_data:
            pos = restore_data['positions_list'][i]
            vel = restore_data['velocities_list'][i]
            pos_openmm = [openmm.Vec3(p[0], p[1], p[2]) for p in pos] * unit.nanometer
            vel_openmm = [openmm.Vec3(v[0], v[1], v[2]) for v in vel] * unit.nanometer/unit.picosecond
            ctx.setPositions(pos_openmm)
            ctx.setVelocities(vel_openmm)
            saved_box = restore_data.get('box_vectors_list', [None] * n_rep)[i]
            if saved_box is not None:
                box_openmm = [openmm.Vec3(*v) for v in saved_box] * unit.nanometer
                ctx.setPeriodicBoxVectors(*box_openmm)
            else:
                ctx.setPeriodicBoxVectors(*base_system.getDefaultPeriodicBoxVectors())
        else:
            ctx.setPositions(pdb.positions)
            ctx.setPeriodicBoxVectors(*base_system.getDefaultPeriodicBoxVectors())
            if i < 5:
                logger.info(f"  Replica {i}: Energy minimization...")
                state_before = ctx.getState(getEnergy=True)
                E_before = state_before.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                logger.info(f"    Before minimization: E_pot = {E_before:.1f} kJ/mol")
            openmm.LocalEnergyMinimizer.minimize(ctx, 10*unit.kilojoule_per_mole/unit.nanometer)
            if i < 5:
                state_after = ctx.getState(getEnergy=True)
                E_after = state_after.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                logger.info(f"    After minimization: E_pot = {E_after:.1f} kJ/mol")
            ctx.setVelocitiesToTemperature(T_ref*unit.kelvin, openmm_seeds[2 * n_rep + i])

        contexts.append(ctx)

    if not restore_data:
        logger.info(f"\n=== Equilibration ===")

        # replica_systems[0] 已包含配置启用的位置约束；复制后用于共同盒膜 NPT。
        # 将 NPT 构型和盒子分发给所有 replica，保证后续 NVT 交换使用同一周期盒。
        npt_steps = cfg["npt_equilibration_steps"]
        npt_ns = npt_steps * cfg["timestep"] / 1.0e6
        logger.info(f"[v4.4] Phase 1: common-box NPT equilibration ({npt_steps} steps, {npt_ns:.3f} ns)...")
        state = contexts[0].getState(getPositions=True, getVelocities=True, enforcePeriodicBox=True)
        npt_sys = copy.deepcopy(replica_systems[0])
        try:
            barostat = openmm.MonteCarloMembraneBarostat(
                1*unit.bar, 0*unit.bar*unit.nanometer, T_ref*unit.kelvin,
                openmm.MonteCarloMembraneBarostat.XYIsotropic,
                openmm.MonteCarloMembraneBarostat.ZFree
            )
            logger.info("[v4.4] Using semi-isotropic MonteCarloMembraneBarostat")
        except (AttributeError, TypeError):
            barostat = openmm.MonteCarloBarostat(1*unit.bar, T_ref*unit.kelvin)
            logger.warning("[v4.4] Compatible MonteCarloMembraneBarostat unavailable; using isotropic barostat")
        barostat.setRandomNumberSeed(openmm_seeds[3 * n_rep])
        npt_sys.addForce(barostat)
        npt_integ = openmm.LangevinMiddleIntegrator(
            T_ref*unit.kelvin, cfg["friction"]/unit.picosecond, cfg["timestep"]*unit.femtoseconds
        )
        npt_integ.setRandomNumberSeed(openmm_seeds[3 * n_rep + 1])
        npt_props = cuda_device_props(0, logger)
        if npt_props is None:
            npt_ctx = openmm.Context(npt_sys, npt_integ, platform)
        else:
            npt_ctx = openmm.Context(npt_sys, npt_integ, platform, npt_props)
        npt_ctx.setPositions(state.getPositions())
        npt_ctx.setVelocities(state.getVelocities())
        npt_ctx.setPeriodicBoxVectors(*state.getPeriodicBoxVectors())
        npt_integ.step(npt_steps)
        npt_state = npt_ctx.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=True)
        common_pos = npt_state.getPositions()
        common_vel = npt_state.getVelocities()
        common_box = npt_state.getPeriodicBoxVectors()
        for ctx in contexts:
            ctx.setPeriodicBoxVectors(*common_box)
            ctx.setPositions(common_pos)
            ctx.setVelocities(common_vel)
        box_a = common_box[0][0].value_in_unit(unit.angstrom)
        box_b = common_box[1][1].value_in_unit(unit.angstrom)
        box_c = common_box[2][2].value_in_unit(unit.angstrom)
        logger.info(f"[v4.4] Common NPT box = {box_a:.2f} x {box_b:.2f} x {box_c:.2f} Å")
        del npt_ctx, npt_integ, npt_sys
        logger.info(f"[v4.4] NPT equilibration complete. Starting per-replica NVT equilibration.")

        nvt_steps = cfg["nvt_equilibration_steps"]
        nvt_ns = nvt_steps * cfg["timestep"] / 1.0e6
        logger.info(f"[v4.4] Phase 2: each replica at physical temperature {T_ref:.1f} K "
                    f"under its own Hamiltonian ({nvt_steps} steps, {nvt_ns:.3f} ns per replica)...")

        if cfg.get("equilibration_restraints", False):
            logger.info(f"[v4.4] Position restraints ACTIVE during NPT and NVT equilibration")
            logger.info(f"[v4.4] Restraints will be removed after equilibration by rebuilding contexts")

        for i, (ctx, T) in enumerate(zip(contexts, temps)):
            integ = ctx.getIntegrator()
            integ.setTemperature(T_ref*unit.kelvin)
            integ.setStepSize(cfg["timestep"]*unit.femtoseconds)
            integ.step(nvt_steps)

            if i < 5:
                state = ctx.getState(getEnergy=True)
                E = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                local_multiplier = local_multiplier_for_lambda(lams[i], lambda_min, lam_hot)
                hot_effective_T = T / local_multiplier
                logger.info(f"  Hamiltonian slot {i} (effective solute T={T:.1f}K, "
                            f"local-hot effective T={hot_effective_T:.1f}K, bath={T_ref:.1f}K): "
                            f"Equilibrated E={E:.1f} kJ/mol")

        # P0-3修复: 平衡后移除位置约束，重建Context
        if cfg.get("equilibration_restraints", False):
            logger.info(f"\n[v4.4] Removing position restraints and rebuilding contexts...")
            new_contexts = []
            for i, (ctx, rep_sys, T) in enumerate(zip(contexts, replica_systems, temps)):
                # 保存平衡后的位置和速度
                state = ctx.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=True)
                pos = state.getPositions()
                vel = state.getVelocities()
                box = state.getPeriodicBoxVectors()

                # 从系统中移除约束力
                ridx = restraint_force_indices[i]
                if ridx >= 0:
                    rep_sys.removeForce(ridx)

                # 重建Context（无约束）
                integ = openmm.LangevinMiddleIntegrator(
                    T_ref*unit.kelvin, cfg["friction"]/unit.picosecond, cfg["timestep"]*unit.femtoseconds
                )
                integ.setRandomNumberSeed(openmm_seeds[n_rep + i])
                new_props = cuda_device_props(i, logger)
                if new_props is None:
                    new_ctx = openmm.Context(rep_sys, integ, platform)
                else:
                    new_ctx = openmm.Context(rep_sys, integ, platform, new_props)
                new_ctx.setPositions(pos)
                new_ctx.setVelocities(vel)
                new_ctx.setPeriodicBoxVectors(*box)
                new_contexts.append(new_ctx)
                del ctx

            contexts = new_contexts
            release_steps = cfg["unrestrained_equilibration_steps"]
            release_ns = release_steps * cfg["timestep"] / 1.0e6
            logger.info(f"[v4.4] Position restraints removed. Running unrestrained NVT equilibration "
                        f"({release_steps} steps, {release_ns:.3f} ns per replica)...")
            for ctx in contexts:
                ctx.getIntegrator().step(release_steps)
            logger.info(f"[v4.4] Unrestrained NVT equilibration complete; production contexts are ready.")
    else:
        logger.info(f"\n=== Skipping Equilibration (Restored from checkpoint) ===")

    # 初始化DCD轨迹写入器
    dcd_writers = []
    dcd_files = []
    if cfg.get("save_trajectory", False):
        logger.info(f"\n=== Initializing Trajectory Writers ===")
        trajectories_dir = cfg.get("trajectories_dir", cfg["output_dir"])
        cfg['_trajectory_paths'] = []
        for i in range(n_rep):
            dcd_name = (f"replica_{i}_cont_from_{start_iteration}.dcd"
                        if restore_data else f"replica_{i}.dcd")
            dcd_file_path = os.path.join(trajectories_dir, dcd_name)
            integration_dt = (cfg["timestep"] / 1000.0) * unit.picoseconds
            dcd_interval = cfg["trajectory_interval"] * cfg["n_steps_per_move"]
            dcd_file_obj = open(dcd_file_path, 'wb')
            dcd_writer = app.DCDFile(
                dcd_file_obj, pdb.topology, integration_dt,
                firstStep=start_iteration * cfg["n_steps_per_move"],
                interval=dcd_interval
            )
            dcd_writers.append(dcd_writer)
            dcd_files.append(dcd_file_obj)
            cfg.setdefault('_trajectory_paths', []).append(dcd_file_path)
            if i < 5:
                logger.info(f"  Replica {i}: {dcd_file_path}")
        logger.info(f"  Trajectory interval: Every {cfg['trajectory_interval']} iterations")
        logger.info(f"  Trajectory mode: Write (new file)")
        logger.info(f"  PBC mode: enforcePeriodicBox + make_whole (v4.4)")

    # 构建分子拓扑，用于make_whole操作
    logger.info(f"\n=== Building Molecule Topology for PBC ===")
    molecule_adjacency, molecule_atoms = build_molecule_topology(
        base_system, pdb.topology, logger
    )
    n_molecules = len(molecule_atoms)
    logger.info(f"[v4.4] Make-whole ready: {n_molecules} molecules")

    def save_dcd_frame(ctx, writer, mol_adjacency, mol_atoms):
        """保存DCD帧，包含PBC包裹和make_whole"""
        state = ctx.getState(getPositions=True, enforcePeriodicBox=True)
        raw_pos = state.getPositions()
        box_vecs = state.getPeriodicBoxVectors()

        # NPT 平衡后盒子可能已改变；使用当前完整盒矩阵兼容正交和三斜盒
        box_matrix = np.array([
            [v.x, v.y, v.z] for v in box_vecs.value_in_unit(unit.nanometer)
        ])

        # v4.0: 优化版 make_whole (跳过小分子)
        if mol_adjacency:
            processed_pos = make_whole_positions(
                raw_pos, mol_adjacency, mol_atoms, box_matrix,
                skip_small_molecules=True, small_molecule_threshold=10
            )
        else:
            processed_pos = raw_pos

        writer.writeModel(processed_pos, periodicBoxVectors=box_vecs)

    rmsd_monitor_replicas = cfg.get("rmsd_monitor_replicas", [0, 7, 15])
    rmsd_monitor_interval = cfg.get("rmsd_monitor_interval", 100)

    def monitor_rmsd(contexts):
        """按 Hamiltonian slot 计算全蛋白 CA 和 local-hot 重原子 RMSD。"""
        ca_values = {}
        hot_heavy_values = {}
        for slot_idx in rmsd_monitor_replicas:
            if slot_idx < len(contexts):
                state = contexts[slot_idx].getState(getPositions=True, enforcePeriodicBox=True)
                positions = state.getPositions()
                ca_values[slot_idx] = calculate_rmsd(
                    initial_ca_coords, extract_ca_positions(positions, ca_atom_indices))
                if hot_heavy_atom_indices:
                    hot_heavy_values[slot_idx] = calculate_rmsd(
                        initial_hot_heavy_coords,
                        extract_ca_positions(positions, hot_heavy_atom_indices))
        return ca_values, hot_heavy_values

    n_steps = cfg["n_steps_per_move"]
    n_iter = cfg["n_iterations"]
    dt = cfg["timestep"]
    total_ps = n_iter * n_steps * dt / 1000
    total_ns = total_ps / 1000

    logger.info(f"\n{'='*70}")
    logger.info(f"REST2 Production Simulation ({mode_name})")
    logger.info(f"  Replicas: {n_rep}, effective solute T: {T_ref:.0f}-{T_max:.0f}K")
    logger.info(f"  Physical thermostat T: {T_ref:.0f}K for every replica")
    logger.info(f"  Steps/iter: {n_steps}")
    logger.info(f"  Iteration range: {start_iteration} - {n_iter}")
    logger.info(f"  Remaining iterations: {n_iter - start_iteration}")
    logger.info(f"  Platform: {plat_name}")
    logger.info(f"[v4.4] Periodic boundary conditions: ACTIVE")
    logger.info(f"[v4.4] Using original box shape from input file")
    logger.info(f"[v4.4] CA + local-hot heavy RMSD monitoring by Hamiltonian slot: "
                f"slots {rmsd_monitor_replicas}, every {rmsd_monitor_interval} iters")
    logger.info(f"{'='*70}\n")

    t0 = time.time()

    # P1-1修复: 写入初始帧（iteration=0）作为DCD的第一帧
    if cfg.get("save_trajectory", False) and dcd_writers and start_iteration == 0:
        logger.info(f"[v4.4] Writing initial frame (iteration 0) to DCD files...")
        for i, ctx in enumerate(contexts):
            save_dcd_frame(
                ctx, dcd_writers[i], molecule_adjacency, molecule_atoms
            )

    try:
        for iteration in range(start_iteration, n_iter):
            for ctx in contexts:
                ctx.getIntegrator().step(n_steps)

            pairs = range(0, n_rep-1, 2) if iteration % 2 == 0 else range(1, n_rep-1, 2)

            positions = []
            box_vectors = []
            energies = []
            for ctx in contexts:
                state = ctx.getState(getEnergy=True, getPositions=True, enforcePeriodicBox=True)
                positions.append(state.getPositions())
                box_vectors.append(state.getPeriodicBoxVectors())
                energies.append(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))

            for m in pairs:
                n = m + 1
                exchange_attempts[m] += 1

                pos_m, pos_n = positions[m], positions[n]
                box_m, box_n = box_vectors[m], box_vectors[n]
                E_m_xm, E_n_xn = energies[m], energies[n]

                # 构型包含位置和周期盒；共同盒 NVT 下二者通常相同，此处理也保证恢复旧数据时一致。
                contexts[m].setPeriodicBoxVectors(*box_n)
                contexts[m].setPositions(pos_n)
                E_m_xn = contexts[m].getState(getEnergy=True, enforcePeriodicBox=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

                contexts[n].setPeriodicBoxVectors(*box_m)
                contexts[n].setPositions(pos_m)
                E_n_xm = contexts[n].getState(getEnergy=True, enforcePeriodicBox=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

                beta_ref = 1/(kB_val*T_ref)
                delta = beta_ref * (E_m_xn + E_n_xm - E_m_xm - E_n_xn)
                delta_history[m].append(delta)

                accept = delta <= 0 or rng.random() < np.exp(-delta)
                acceptance_history[m].append(bool(accept))

                if accept:
                    exchange_accepts[m] += 1
                    # 所有 Hamiltonian 槽位使用同一 T_ref 热浴，交换仅移动构型；
                    # 各 slot 保持自己的原速度，无需重采样，也不交换速度。
                    positions[m], positions[n] = pos_n, pos_m
                    box_vectors[m], box_vectors[n] = box_n, box_m
                    energies[m], energies[n] = E_m_xn, E_n_xm
                    walker_labels[m], walker_labels[n] = walker_labels[n], walker_labels[m]
                else:
                    contexts[m].setPeriodicBoxVectors(*box_m)
                    contexts[m].setPositions(pos_m)
                    contexts[n].setPeriodicBoxVectors(*box_n)
                    contexts[n].setPositions(pos_n)

            energy_history.append(energies)
            walker_label_history.append(walker_labels.copy())

            if (iteration + 1) % rmsd_monitor_interval == 0:
                ca_rmsd_vals, hot_heavy_rmsd_vals = monitor_rmsd(contexts)
                rmsd_history.append({
                    'iteration': iteration + 1,
                    'rmsd': ca_rmsd_vals,
                    'hot_heavy_rmsd': hot_heavy_rmsd_vals,
                    'slot_semantics': 'hamiltonian_slot'
                })
                ca_text = ", ".join(
                    f"slot {slot}: {value:.2f}Å" for slot, value in ca_rmsd_vals.items())
                hot_text = ", ".join(
                    f"slot {slot}: {value:.2f}Å" for slot, value in hot_heavy_rmsd_vals.items())
                logger.info(f"  [v4.4] Protein CA RMSD by Hamiltonian slot @ Iter {iteration+1}: {ca_text}")
                if hot_text:
                    logger.info(f"  [v4.4] Local-hot heavy-atom RMSD by Hamiltonian slot @ Iter "
                                f"{iteration+1}: {hot_text}")
                for slot, value in hot_heavy_rmsd_vals.items():
                    if value > 5.0:
                        logger.warning(f"[v4.4] SEVERE WARNING: Hamiltonian slot {slot} local-hot "
                                       f"heavy-atom RMSD={value:.2f} Å (>5.0 Å)")
                    elif value > 3.0:
                        logger.warning(f"[v4.4] WARNING: Hamiltonian slot {slot} local-hot "
                                       f"heavy-atom RMSD={value:.2f} Å (>3.0 Å)")

            if cfg.get("save_trajectory", False) and dcd_writers:
                if (iteration + 1) % cfg["trajectory_interval"] == 0:
                    for i, ctx in enumerate(contexts):
                        save_dcd_frame(
                            ctx, dcd_writers[i], molecule_adjacency, molecule_atoms
                        )

            if (iteration+1) % 20 == 0:
                numeric_rates = [exchange_accepts[k]/exchange_attempts[k]*100
                                 if exchange_attempts[k] > 0 else np.nan
                                 for k in range(n_rep-1)]
                rates_text = ", ".join(
                    f"{k}-{k+1}: {rate:.1f}%" if np.isfinite(rate) else f"{k}-{k+1}: -"
                    for k, rate in enumerate(numeric_rates)
                )
                finite_rates = np.asarray([rate for rate in numeric_rates if np.isfinite(rate)])
                logger.info(f"  Iter {iteration+1:5d}/{n_iter}: E(first 4)={[int(e) for e in energies[:4]]}")
                logger.info(f"    Adjacent exchange rates: {rates_text}")
                logger.info(f"    Exchange rate min/mean/max: {finite_rates.min():.1f}% / "
                            f"{finite_rates.mean():.1f}% / {finite_rates.max():.1f}%")

            if (iteration+1) % cfg["checkpoint_interval"] == 0:
                save_full_checkpoint(cfg, iteration+1, contexts, energy_history, exchange_attempts,
                                    exchange_accepts, temps, lams, delta_history, logger, rng,
                                    rmsd_history=rmsd_history,
                                    acceptance_history=acceptance_history,
                                    walker_labels=walker_labels,
                                    walker_label_history=walker_label_history,
                                    walker_history_start_iteration=walker_history_start_iteration)
    finally:
        if dcd_writers:
            logger.info(f"\n=== Closing Trajectory Files ===")
            for i, dcd_f in enumerate(dcd_files):
                dcd_f.close()
                if i < 5:
                    logger.info(f"  Replica {i}: DCD file closed")

    if rmsd_history:
        rmsd_file = os.path.join(cfg["output_dir"], "rmsd_history.npy")
        np.save(rmsd_file, np.asarray(rmsd_history, dtype=object), allow_pickle=True)
        logger.info(f"[v4.4] CA and local-hot heavy-atom RMSD history saved: {rmsd_file}")

    total_time = time.time() - t0

    final_rates = np.asarray([
        exchange_accepts[k]/exchange_attempts[k]*100
        for k in range(n_rep-1) if exchange_attempts[k] > 0
    ])

    logger.info(f"\n{'='*70}")
    logger.info(f"REST2 Simulation Complete! [v{cfg['version']}]")
    logger.info(f"  Total time: {total_time:.1f}s")
    logger.info("  Final adjacent exchange rates:")
    for k in range(n_rep-1):
        if exchange_attempts[k] > 0:
            rate = exchange_accepts[k]/exchange_attempts[k]*100
            logger.info(f"    Pair {k:2d}-{k+1:2d}: {rate:6.2f}% "
                        f"({exchange_accepts[k]}/{exchange_attempts[k]})")
            if rate < 10.0:
                logger.warning(f"    Exchange failure: pair {k}-{k+1} = {rate:.2f}% (<10%)")
            elif rate < 20.0:
                logger.warning(f"    Low exchange warning: pair {k}-{k+1} = {rate:.2f}% (10-20%)")
            elif rate > 70.0:
                logger.info(f"    Pair {k}-{k+1} = {rate:.2f}% (>70%): Hamiltonian 梯度可能过密")
        else:
            logger.info(f"    Pair {k:2d}-{k+1:2d}: no attempts")
    if final_rates.size:
        logger.info(f"  Exchange rate min/mean/max: {final_rates.min():.2f}% / "
                    f"{final_rates.mean():.2f}% / {final_rates.max():.2f}%")
    logger.info(f"  Effective solute temperature: {T_ref:.1f}-{T_max:.1f} K; "
                f"bath temperature: {T_ref:.1f} K for every replica")
    round_trip_summary = analyze_walker_round_trips(walker_label_history, n_rep)
    log_round_trip_summary(round_trip_summary, logger)

    if rmsd_history:
        final_rmsd = rmsd_history[-1]['rmsd']
        final_hot_rmsd = rmsd_history[-1].get('hot_heavy_rmsd', {})
        logger.info("  Final RMSD vs initial structure by Hamiltonian slot:")
        for slot_idx, rmsd_val in final_rmsd.items():
            T = temps[slot_idx]
            logger.info(f"    Slot {slot_idx} (effective solute T={T:.1f}K): CA RMSD={rmsd_val:.2f} Å")
        for slot_idx, rmsd_val in final_hot_rmsd.items():
            local_multiplier = local_multiplier_for_lambda(lams[slot_idx], lambda_min, lam_hot)
            hot_effective_T = temps[slot_idx] / local_multiplier
            logger.info(f"    Slot {slot_idx} (local-hot effective T={hot_effective_T:.1f}K): "
                        f"hot-heavy RMSD={rmsd_val:.2f} Å")

    logger.info(f"{'='*70}")

    results = {
        'energies': np.array(energy_history),
        'exchange_attempts': exchange_attempts,
        'exchange_accepts': exchange_accepts,
        'temperatures': np.array(temps),
        'lam_values': np.array(lams),
        'total_time': total_time,
        'delta_history': delta_history,
        'acceptance_history': acceptance_history,
        'walker_labels': walker_labels,
        'walker_label_history': np.asarray(walker_label_history, dtype=int),
        'walker_history_start_iteration': walker_history_start_iteration,
        'walker_round_trip_directions': round_trip_summary['directional_counts'],
        'walker_round_trips': round_trip_summary['per_walker'],
        'round_trip_total': round_trip_summary['total'],
        'walkers_with_round_trip': round_trip_summary['walkers_with_round_trip'],
        'walker_visited_slot_counts': round_trip_summary['visited_slot_counts'],
        'walker_slot_coverage': round_trip_summary['slot_coverage'],
        'walkers_visiting_both_ends': round_trip_summary['walkers_visiting_both_ends'],
        'walker_visited_both_ends': round_trip_summary['visited_both_ends'],
        'walker_occupancy_uniform_rmse': round_trip_summary['occupancy_uniform_rmse'],
        'version': cfg['version'],
        'cmap_scaling_mode': cfg['cmap_scaling_mode'],
        'local_heating': cfg['local_heating_enabled'],
        'hot_residue_ids': cfg['hot_residue_ids'],
        'hot_residue_names': cfg['hot_residue_names'],
        'lambda_hot': cfg['lambda_hot'],
        'local_scaling_scheme': cfg['local_scaling_scheme'],
        'reference_state_unbiased': True,
        'local_multiplier_endpoint': cfg['lambda_hot'],
        'hot_effective_temperatures': np.array([
            T / local_multiplier_for_lambda(lam, lambda_min, lam_hot)
            for T, lam in zip(temps, lams)
        ]),
        'periodic_boundary': cfg.get('periodic_boundary', True),
        'cubic_box': cfg.get('use_cubic_box', False),
        'box_dims': box_dims,
        'box_size': box_size,
        'rmsd_history': rmsd_history,
        'residue_offset': cfg.get('residue_offset', 9),
    }

    try:
        generate_scientific_visualizations(results, cfg["output_dir"], logger)
    except ImportError as e:
        logger.warning(f"[v4.4] matplotlib not installed, skipping visualizations: {e}")
    walker_history_file = os.path.join(cfg["output_dir"], "walker_label_history.npy")
    np.save(walker_history_file, results['walker_label_history'])
    logger.info(f"[v4.4] Walker label history saved: {walker_history_file}")

    # 使用MDAnalysis计算OF gate Rg
    logger.info(f"\n{'='*70}")
    logger.info(f"[v4.4] Calculating OF Gate Rg using MDAnalysis")
    logger.info(f"{'='*70}")
    logger.info(f"  OF gate selection: {OF_GATE_SELECTION}")
    logger.info(f"  Equilibration fraction: {cfg['rg_equilibration_fraction']*100:.0f}%")

    rg_results = calculate_rg_with_mdanalysis(cfg, logger, temps)

    if rg_results:
        generate_rg_visualizations(rg_results, temps, cfg["lambda_hot"], cfg["output_dir"], logger)

    evaluation = evaluate_experiment_quality(cfg, results, rg_results)
    results['rg_results'] = rg_results
    results['experiment_evaluation'] = evaluation
    save_final_results(cfg, results, contexts, pdb)
    save_scientific_report(cfg, results, logger)
    evaluation_file = os.path.join(cfg["output_dir"], "experiment_evaluation.json")
    with open(evaluation_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation, f, indent=2, ensure_ascii=False)
    logger.info(f"  Experiment evaluation saved to {evaluation_file}")
    logger.info(f"  Production readiness: {evaluation['production_readiness']['status']}")
    logger.info(f"  Overall stage: {evaluation['overall_stage']}")
    for dimension in ('numerical_stability', 'exchange_quality', 'structural_safety',
                      'walker_mixing', 'sampling_maturity', 'rg_evidence'):
        item = evaluation[dimension]
        logger.info(f"  {dimension}: {item['status']} - {item['reason']}")

    # v4.0: 生成 RMSD 演化图
    if rmsd_history:
        try:
            generate_rmsd_visualizations(rmsd_history, temps, cfg["lambda_hot"], cfg["output_dir"], logger)
        except Exception as e:
            logger.warning(f"[v4.4] Could not generate RMSD visualization: {e}")

    # 验证DCD文件
    if cfg.get("save_trajectory", False):
        logger.info(f"\n[v4.4] Verifying DCD trajectory files...")
        trajectories_dir = cfg.get("trajectories_dir", cfg["output_dir"])
        trajectory_paths = cfg.get('_trajectory_paths')
        for i in range(min(3, n_rep)):
            dcd_file = (trajectory_paths[i] if trajectory_paths
                        else os.path.join(trajectories_dir, f"replica_{i}.dcd"))
            if os.path.exists(dcd_file):
                verify_dcd_file(dcd_file, logger)

    return results


def save_final_results(cfg, results, contexts, pdb):
    final_file = os.path.join(cfg["output_dir"], "rest2_results.npz")
    np.savez_compressed(final_file,
        energies=results['energies'],
        attempts=results['exchange_attempts'],
        accepts=results['exchange_accepts'],
        temps=results['temperatures'],
        lams=results['lam_values'],
        total_time=results['total_time'],
        version=results['version'],
        cmap_mode=results['cmap_scaling_mode'],
        lambda_hot=results['lambda_hot'],
        local_scaling_scheme=results['local_scaling_scheme'],
        reference_state_unbiased=results['reference_state_unbiased'],
        local_multiplier_endpoint=results['local_multiplier_endpoint'],
        hot_effective_temperatures_K=results['hot_effective_temperatures'],
        rmsd_history=np.asarray(results['rmsd_history'], dtype=object),
        hot_residue_ids=results['hot_residue_ids'],
        periodic_boundary=results['periodic_boundary'],
        cubic_box=results['cubic_box'],
        box_size=results['box_size'],
        box_dims=results.get('box_dims', []),
        residue_offset=results.get('residue_offset', 9),
        walker_labels=results['walker_labels'],
        walker_label_history=results['walker_label_history'],
        walker_history_start_iteration=results['walker_history_start_iteration'],
        walker_label_semantics='slot_to_walker',
        walker_round_trip_directions=results['walker_round_trip_directions'],
        walker_round_trips=results['walker_round_trips'],
        round_trip_total=results['round_trip_total'],
        walkers_with_round_trip=results['walkers_with_round_trip'],
        walker_visited_slot_counts=results['walker_visited_slot_counts'],
        walker_slot_coverage=results['walker_slot_coverage'],
        walkers_visiting_both_ends=results['walkers_visiting_both_ends'],
        walker_visited_both_ends=results['walker_visited_both_ends'],
        walker_occupancy_uniform_rmse=results['walker_occupancy_uniform_rmse'],
        experiment_evaluation_json=json.dumps(results['experiment_evaluation'], ensure_ascii=False),
        **{f"delta_{k}": np.array(v) for k, v in results['delta_history'].items() if v},
        **{f"accepted_{k}": np.asarray(v, dtype=bool)
           for k, v in results['acceptance_history'].items() if v})

    for i, ctx in enumerate(contexts):
        state = ctx.getState(getPositions=True, enforcePeriodicBox=True)
        pos = state.getPositions()
        pdb.topology.setPeriodicBoxVectors(state.getPeriodicBoxVectors())
        pdb_file = os.path.join(cfg["output_dir"], f"replica_{i}_final.pdb")
        with open(pdb_file, 'w') as f:
            app.PDBFile.writeFile(pdb.topology, pos, f)

    energy_file = os.path.join(cfg["mbar_dir"], "own_state_energies.npz")
    np.savez_compressed(
        energy_file,
        own_state_energies_kj_per_mol=results['energies'],
        temperatures_K=results['temperatures'],
        lambda_hot=results['lambda_hot'],
        metadata="每帧仅含其自身 Hamiltonian 的势能；未计算交叉态能量，不能直接作为 MBAR u_kn/N_k"
    )


def generate_scientific_visualizations(results, output_dir, logger):
    logger.info(f"\n=== Generating Scientific Visualizations ===")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figures_dir = os.path.join(output_dir, 'figures')

    E = results['energies']
    attempts = results['exchange_attempts']
    accepts = results['exchange_accepts']
    T = results['temperatures']
    deltas = results['delta_history']
    acceptance_history = results.get('acceptance_history', {})

    n_rep = len(T)
    rates = [accepts[k]/attempts[k]*100 if attempts[k] > 0 else 0 for k in range(n_rep-1)]
    avg_rate = np.mean(rates)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    ax = axes[0,0]
    for i in range(min(8, n_rep)):
        ax.plot(E[:,i], label=f'Rep{i} ({T[i]:.0f}K)', alpha=0.7, lw=0.8)
    ax.set_xlabel('Iteration'); ax.set_ylabel('E (kJ/mol)')
    ax.set_title('(a) Potential Energy', fontweight='bold')
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)

    ax = axes[0,1]
    colors = ['#27ae60' if 20 <= r <= 60 else '#f39c12' if 10 <= r <= 70 else '#e74c3c'
              for r in rates]
    ax.bar(range(n_rep-1), rates, color=colors)
    ax.axhline(20, color='k', ls='--'); ax.axhline(60, color='k', ls='--')
    ax.set_xlabel('Replica Pair'); ax.set_ylabel('Exchange Rate (%)')
    ax.set_title('(b) Exchange Acceptance', fontweight='bold')
    ax.grid(alpha=0.3)

    ax = axes[0,2]
    for k in range(min(8, n_rep-1)):
        if deltas[k]:
            ax.hist(deltas[k], bins=30, alpha=0.5, label=f'Pair {k}-{k+1}', density=True)
    ax.axvline(0, color='k'); ax.set_xlabel('Delta'); ax.set_ylabel('P(Delta)')
    ax.set_title('(c) Delta Distribution', fontweight='bold')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[1,0]
    E_mean = E.mean(axis=0); E_std = E.std(axis=0)
    ax.errorbar(T, E_mean, yerr=E_std, fmt='o-', capsize=3)
    ax.set_xlabel('Effective solute temperature (K)'); ax.set_ylabel('Average E (kJ/mol)')
    ax.set_title('(d) Energy vs Effective Solute Temperature', fontweight='bold')
    ax.grid(alpha=0.3)

    ax = axes[1,1]
    for k in range(min(8, n_rep-1)):
        accepted = acceptance_history.get(k, [])
        if accepted:
            cum = np.cumsum(np.asarray(accepted, dtype=bool)) / np.arange(1, len(accepted)+1) * 100
            ax.plot(cum, label=f'Pair {k}-{k+1}', alpha=0.7)
    ax.axhline(avg_rate, color='k', ls='--', label=f'Avg: {avg_rate:.1f}%')
    ax.set_xlabel('Exchange Attempt'); ax.set_ylabel('Cumulative Rate (%)')
    ax.set_title('(e) Convergence', fontweight='bold')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[1,2]
    lams = results['lam_values']
    ax.plot(T, lams, 'o-', label='λ_global', color='blue')
    ax.set_xlabel('Effective solute temperature (K)'); ax.set_ylabel('Lambda')
    ax.set_title('(f) Lambda Scaling', fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'fig1_analysis.png'), dpi=150)
    plt.close()

    logger.info(f"  Figures saved to {figures_dir}")


def generate_rmsd_visualizations(rmsd_history, temps, lambda_hot, output_dir, logger):
    """
    v4.0: 生成 RMSD 演化图
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    figures_dir = os.path.join(output_dir, 'figures')

    logger.info(f"\n[v4.4] Generating RMSD evolution visualization...")

    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    ax = axes[0]

    # 提取数据
    iterations = [r['iteration'] for r in rmsd_history]

    # 每个监控副本画一条线
    all_reps = set()
    for r in rmsd_history:
        all_reps.update(r['rmsd'].keys())

    colors = plt.cm.coolwarm(np.linspace(0, 1, len(all_reps)))

    for rep_idx, color in zip(sorted(all_reps), colors):
        rmsd_vals = [r['rmsd'].get(rep_idx, np.nan) for r in rmsd_history]
        T = temps[rep_idx] if rep_idx < len(temps) else 0
        ax.plot(iterations, rmsd_vals, 'o-', color=color, markersize=4,
                label=f'Hamiltonian slot {rep_idx} (effective solute T={T:.0f}K)', linewidth=1.5)

    ax.set_ylabel('Protein CA RMSD (Å)', fontsize=12)
    ax.set_title('Protein CA RMSD vs Initial Structure', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    lambda_min = temps[0] / temps[-1]
    for rep_idx, color in zip(sorted(all_reps), colors):
        rmsd_vals = [r.get('hot_heavy_rmsd', {}).get(rep_idx, np.nan) for r in rmsd_history]
        lam_global = temps[0] / temps[rep_idx]
        hot_T = temps[rep_idx] / local_multiplier_for_lambda(lam_global, lambda_min, lambda_hot)
        ax.plot(iterations, rmsd_vals, 'o-', color=color, markersize=4,
                label=f'Hamiltonian slot {rep_idx} (local-hot effective T={hot_T:.1f}K)', linewidth=1.5)
    ax.axhline(3.0, color='orange', ls='--', alpha=0.7, label='warning 3 Å')
    ax.axhline(5.0, color='red', ls='--', alpha=0.7, label='severe warning 5 Å')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Local-hot heavy RMSD (Å)', fontsize=12)
    ax.set_title('Local-hot Heavy-atom RMSD vs Initial Structure', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'fig_rmsd_evolution.png'), dpi=150)
    plt.close()

    logger.info(f"  RMSD evolution figure saved: {os.path.join(figures_dir, 'fig_rmsd_evolution.png')}")


def save_scientific_report(cfg, results, logger):
    E = results['energies']
    attempts = results['exchange_attempts']
    accepts = results['exchange_accepts']
    T = results['temperatures']

    n_rep = len(T)
    rates = [accepts[k]/attempts[k]*100 if attempts[k] > 0 else 0 for k in range(n_rep-1)]
    avg_rate = np.mean(rates)

    rmsd_stats = {}
    if results.get('rmsd_history'):
        final_rmsd = results['rmsd_history'][-1]['rmsd']
        final_hot_rmsd = results['rmsd_history'][-1].get('hot_heavy_rmsd', {})
        rmsd_stats = {
            'slot_semantics': 'hamiltonian_slot',
            'final_ca_rmsd_A': {str(k): float(v) for k, v in final_rmsd.items()},
            'final_hot_heavy_rmsd_A': {str(k): float(v) for k, v in final_hot_rmsd.items()},
            'hot_heavy_warning_threshold_A': 3.0,
            'hot_heavy_severe_warning_threshold_A': 5.0,
        }
        if len(results['rmsd_history']) > 4:
            late_rmsd = results['rmsd_history'][len(results['rmsd_history'])//2:]
            for slot_idx in final_rmsd.keys():
                vals = [r['rmsd'].get(slot_idx, np.nan) for r in late_rmsd]
                rmsd_stats[f'avg_ca_rmsd_late_slot{slot_idx}_A'] = float(np.nanmean(vals))
            for slot_idx in final_hot_rmsd.keys():
                vals = [r.get('hot_heavy_rmsd', {}).get(slot_idx, np.nan) for r in late_rmsd]
                rmsd_stats[f'avg_hot_heavy_rmsd_late_slot{slot_idx}_A'] = float(np.nanmean(vals))

    report = {
        'title': 'REST2 v4.4 scientific report',
        'version': results['version'],
        'n_replicas': int(n_rep),
        'temperature_interpretation': f"effective solute temperature; all replicas use a {cfg['T_ref']:g} K bath",
        'effective_solute_temperatures_K': [float(t) for t in T],
        'bath_temperature_K': float(cfg['T_ref']),
        'exchange_rates_percent': [float(round(rate, 2)) for rate in rates],
        'min_exchange_rate': float(round(np.min(rates), 2)),
        'avg_exchange_rate': float(round(avg_rate, 2)),
        'max_exchange_rate': float(round(np.max(rates), 2)),
        'n_iterations': int(len(E)),
        'total_simulation_ns': float(len(E) * cfg['n_steps_per_move'] * cfg['timestep'] / 1000000),
        'total_time_s': float(round(results['total_time'], 1)),
        'periodic_boundary': results['periodic_boundary'],
        'cubic_box': results['cubic_box'],
        'box_dimensions_angstrom': [float(d) for d in results.get('box_dims', [82.407, 82.407, 109.362])],
        'box_size_max_angstrom': float(results['box_size']),
        'lambda_hot': float(results['lambda_hot']),
        'local_scaling_scheme': results['local_scaling_scheme'],
        'reference_state_unbiased': bool(results['reference_state_unbiased']),
        'local_multiplier_endpoint': float(results['local_multiplier_endpoint']),
        'hot_effective_temperature_range_K': ([float(results['hot_effective_temperatures'][0]),
                                                float(results['hot_effective_temperatures'][-1])]
                                               if results['local_heating'] else None),
        'hot_effective_temperatures_K': ([float(t) for t in results['hot_effective_temperatures']]
                                         if results['local_heating'] else None),
        'residue_offset': int(results.get('residue_offset', 9)),
        'round_trip': {
            'per_walker': [int(v) for v in results['walker_round_trips']],
            'directional_0_top_0__top_0_top': results['walker_round_trip_directions'].astype(int).tolist(),
            'total': int(results['round_trip_total']),
            'walkers_with_at_least_one': int(results['walkers_with_round_trip']),
            'visited_slot_counts': results['walker_visited_slot_counts'].astype(int).tolist(),
            'slot_coverage': results['walker_slot_coverage'].astype(float).tolist(),
            'walkers_visiting_both_ends': int(results['walkers_visiting_both_ends']),
            'visited_both_ends': results['walker_visited_both_ends'].astype(bool).tolist(),
            'occupancy_uniform_relative_rmse': results['walker_occupancy_uniform_rmse'].astype(float).tolist(),
            'note': ('未观察到完整往返；应按生产时长评价，短预实验不据此判错'
                     if results['round_trip_total'] == 0 else '已观察到完整端点往返'),
        },
        'rmsd_stats': rmsd_stats,
        'rg_evidence': results['experiment_evaluation']['rg_evidence'],
        'experiment_evaluation': results['experiment_evaluation'],
    }

    report_file = os.path.join(cfg["output_dir"], "scientific_report.json")
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info(f"  Scientific report saved to {report_file}")


# ======================================================================
# MDAnalysis Rg计算 (与com_rg_dist.py完全一致)
# ======================================================================

def radgyr(atomgroup, masses, total_mass=None):
    """
    回转半径计算函数 (参考com_rg_dist.py)
    """
    coordinates = atomgroup.positions
    center_of_mass = atomgroup.center_of_mass()

    ri_sq = (coordinates - center_of_mass) ** 2
    sq = np.sum(ri_sq, axis=1)

    sq_rs = np.array([sq])

    if total_mass is None:
        total_mass = np.sum(masses)
    rog_sq = np.sum(masses * sq_rs, axis=1) / total_mass

    return np.sqrt(rog_sq)


def calculate_rg_with_mdanalysis(cfg, logger, temps):
    """
    使用MDAnalysis计算OF gate回转半径
    """
    try:
        import MDAnalysis as mda
        from MDAnalysis.analysis.base import analysis_class
    except ImportError:
        logger.error("MDAnalysis not installed! Cannot calculate OF gate Rg.")
        logger.error("Install: pip install MDAnalysis")
        return None

    logger.info(f"\n[v4.4] Loading MDAnalysis...")
    logger.info(f"  MDAnalysis version: {mda.__version__}")

    trajectories_dir = cfg.get("trajectories_dir", cfg["output_dir"])
    n_replicas = cfg["n_replicas"]
    psf_file = cfg["psf_file"]
    equil_fraction = cfg.get("rg_equilibration_fraction", 0.2)

    rg_results = {}
    trajectory_paths = cfg.get('_trajectory_paths')
    for rep_idx in range(n_replicas):
        dcd_file = (trajectory_paths[rep_idx] if trajectory_paths
                    else os.path.join(trajectories_dir, f"replica_{rep_idx}.dcd"))

        if not os.path.exists(dcd_file):
            logger.warning(f"  Replica {rep_idx}: DCD file not found - {dcd_file}")
            continue

        try:
            logger.info(f"  Replica {rep_idx}: Loading trajectory...")
            u = mda.Universe(psf_file, dcd_file)

            atom_of = u.select_atoms(OF_GATE_SELECTION)

            if atom_of.n_atoms == 0:
                logger.warning(f"  Replica {rep_idx}: No OF gate atoms selected!")
                continue

            logger.info(f"  Replica {rep_idx}: OF gate atoms selected: {atom_of.n_atoms}")

            RadiusOfGyration = analysis_class(radgyr)

            total_mass = np.sum(atom_of.masses)
            rog_u_of = RadiusOfGyration(u.trajectory, atom_of, atom_of.masses, total_mass=total_mass)
            rog_u_of.run()

            rg_values = rog_u_of.results['timeseries'][:, 0]

            n_frames = len(rg_values)
            equil_cutoff = int(n_frames * equil_fraction)
            rg_values_equilibrated = rg_values[equil_cutoff:]

            if len(rg_values_equilibrated) > 0:
                rg_results[rep_idx] = {
                    'n_frames': len(rg_values_equilibrated),
                    'rg_mean': np.mean(rg_values_equilibrated),
                    'rg_std': np.std(rg_values_equilibrated),
                    'rg_min': np.min(rg_values_equilibrated),
                    'rg_max': np.max(rg_values_equilibrated),
                    'rg_values': rg_values_equilibrated,
                }

                lambda_min = temps[0] / temps[-1]
                lam_global = temps[0] / temps[rep_idx]
                local_multiplier = local_multiplier_for_lambda(
                    lam_global, lambda_min, cfg['lambda_hot'])
                hot_effective_T = temps[rep_idx] / local_multiplier
                logger.info(f"  Hamiltonian slot {rep_idx}: OF gate Rg = "
                            f"{np.mean(rg_values_equilibrated):.2f} ± {np.std(rg_values_equilibrated):.2f} Å "
                            f"(effective solute T={temps[rep_idx]:.1f}K, "
                            f"local-hot effective T={hot_effective_T:.1f}K, "
                            f"{len(rg_values_equilibrated)} frames)")

        except Exception as e:
            logger.warning(f"  Replica {rep_idx}: Error calculating Rg - {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())

    return rg_results


def generate_rg_visualizations(rg_results, temps, lambda_hot, output_dir, logger):
    """生成OF gate Rg可视化图表"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as e:
        logger.warning(f"[v4.4] matplotlib not installed, skipping OF gate Rg visualizations: {e}")
        return

    figures_dir = os.path.join(output_dir, 'figures')

    if not rg_results:
        logger.warning("No OF gate Rg data to visualize")
        return

    logger.info(f"\n[v4.4] Generating OF gate Rg visualizations...")

    rep_indices = sorted(rg_results.keys())
    rg_means = [rg_results[i]['rg_mean'] for i in rep_indices]
    rg_stds = [rg_results[i]['rg_std'] for i in rep_indices]
    temp_values = [temps[i] for i in rep_indices]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.errorbar(temp_values, rg_means, yerr=rg_stds, fmt='o-', capsize=3,
                color='#2c3e50', linewidth=2, markersize=6)
    ax.set_xlabel('Effective solute temperature (K)', fontsize=12)
    ax.set_ylabel('OF Gate Radius of Gyration (Å)', fontsize=12)
    ax.set_title('(a) OF Gate Rg vs Effective Solute Temperature', fontweight='bold')
    ax.grid(alpha=0.3)

    try:
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(temp_values, rg_means)
        ax.plot(temp_values, [slope*t + intercept for t in temp_values],
                '--', color='#e74c3c', label=f'Trend: Rg = {slope:.4f}T + {intercept:.2f}')
        ax.legend(fontsize=10)
    except ImportError:
        logger.warning("scipy not available, skipping trend line")

    ax = axes[1]
    n_temps = len(temp_values)
    colors = plt.cm.viridis(np.linspace(0, 1, n_temps))

    for i, (idx, color) in enumerate(zip(rep_indices[::4], colors[::4])):
        rg_vals = rg_results[idx]['rg_values']
        ax.hist(rg_vals, bins=20, alpha=0.5, color=color,
                label=f'{temps[idx]:.0f}K', density=True)

    ax.set_xlabel('OF Gate Radius of Gyration (Å)', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.set_title('(b) OF Gate Rg Distribution', fontweight='bold')
    ax.legend(fontsize=9, ncol=2)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'fig_rg_analysis.png'), dpi=150)
    plt.close()

    logger.info(f"  OF gate Rg analysis figure saved: {os.path.join(figures_dir, 'fig_rg_analysis.png')}")

    rg_data_file = os.path.join(output_dir, "rg_analysis.json")
    rg_save_data = {
        'analysis_type': 'OF_gate_Rg',
        'selection_string': OF_GATE_SELECTION,
        'replica_indices': [int(i) for i in rep_indices],
        'effective_solute_temperatures_K': [float(t) for t in temp_values],
        'local_hot_effective_temperatures_K': [
            float(temps[i] / local_multiplier_for_lambda(
                temps[0] / temps[i], temps[0] / temps[-1], lambda_hot))
            for i in rep_indices
        ],
        'rg_mean_A': [float(r) for r in rg_means],
        'rg_std_A': [float(r) for r in rg_stds],
    }

    try:
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(temp_values, rg_means)
        rg_save_data['trend_slope'] = float(slope)
        rg_save_data['trend_intercept'] = float(intercept)
        rg_save_data['trend_r_squared'] = float(r_value**2)
    except:
        pass

    with open(rg_data_file, 'w', encoding='utf-8') as f:
        json.dump(rg_save_data, f, indent=2)

    logger.info(f"  OF gate Rg data saved: {rg_data_file}")


if __name__ == "__main__":
    args = parse_command_line()

    manager = ExperimentManager()

    restore_data = None
    if args.restore:
        if not args.output_dir:
            print("ERROR: --output_dir is required when using --restore")
            sys.exit(1)
        exp_dir = manager.setup_experiment(output_dir=args.output_dir)
    else:
        exp_dir = manager.setup_experiment()

    logger = manager.logger
    cfg = manager.cfg

    cfg['n_iterations'] = args.n_iterations
    cfg['n_steps_per_move'] = args.n_steps_per_move
    cfg['n_replicas'] = args.n_replicas
    cfg['T_ref'] = args.T_ref
    cfg['T_max'] = args.T_max
    cfg['checkpoint_interval'] = args.checkpoint_interval
    cfg['npt_equilibration_steps'] = args.npt_equilibration_steps
    cfg['nvt_equilibration_steps'] = args.nvt_equilibration_steps
    cfg['unrestrained_equilibration_steps'] = args.unrestrained_equilibration_steps
    cfg['trajectory_interval'] = args.trajectory_interval
    cfg['rmsd_monitor_interval'] = args.rmsd_monitor_interval

    # v4.0: 应用局部加热参数
    cfg['lambda_hot'] = args.lambda_hot
    if args.local_heating:
        cfg['local_heating_enabled'] = True
        logger.warning("[v4.4] Non-standard local-hot extension ENABLED by command line")
    elif args.no_local_heating:
        cfg['local_heating_enabled'] = False
        logger.info("[v4.4] Local-hot extension disabled; using standard REST2")

    # 仅当用户显式指定时覆盖默认位置约束设置
    if args.equilibration_restraints is not None:
        cfg['equilibration_restraints'] = args.equilibration_restraints

    cfg['start_iteration'] = args.start_iteration

    try:
        if args.restore:
            restore_data = restore_from_checkpoint(args.restore, cfg, logger)
            saved_cfg = restore_data['checkpoint_cfg']
            for key in ('local_heating_enabled', 'lambda_hot', 'local_scaling_scheme',
                        'hot_residue_ids', 'hot_residue_names', 'T_ref',
                        'T_max', 'timestep', 'friction'):
                if key in saved_cfg:
                    current = cfg.get(key)
                    saved = saved_cfg[key]
                    if isinstance(current, np.ndarray):
                        differs = not np.array_equal(current, saved)
                    else:
                        differs = current != saved
                    if differs:
                        logger.warning("恢复配置覆盖当前值: %s=%r -> checkpoint值 %r", key, current, saved)
                    cfg[key] = saved
            cfg['n_replicas'] = len(restore_data['temps'])
            cfg['start_iteration'] = restore_data['start_iteration']
            logger.info("恢复副本数采用 checkpoint: n_replicas=%d", cfg['n_replicas'])
            if args.start_iteration not in (0, restore_data['start_iteration']):
                logger.warning("忽略 --start_iteration=%d；恢复迭代由 checkpoint 决定为 %d",
                               args.start_iteration, restore_data['start_iteration'])
        validate_config(cfg)
        # 命令行及 checkpoint 覆盖完成后再写配置，确保 config.json 与实际运行参数一致
        manager.save_config()

        logger.info(f"\n{'='*70}")
        mode = "REST2 + non-standard local heating" if cfg['local_heating_enabled'] else "standard REST2"
        exchange_ps = cfg['n_steps_per_move'] * cfg['timestep'] / 1000.0
        production_ns = cfg['n_iterations'] * exchange_ps / 1000.0
        logger.info(f"REST2 Simulation - 4AU5 Protein + CHARMM36m")
        logger.info(f"Version: {cfg['version']} ({mode})")
        logger.info(f"Replicas: {cfg['n_replicas']}")
        logger.info(f"Effective solute temperature: {cfg['T_ref']:.1f}-{cfg['T_max']:.1f} K")
        logger.info(f"Physical thermostat temperature: {cfg['T_ref']:.1f} K")
        logger.info(f"Exchange interval: {cfg['n_steps_per_move']} steps ({exchange_ps:.3f} ps)")
        logger.info(f"Production: {cfg['n_iterations']} iterations ({production_ns:.3f} ns per replica)")
        logger.info(f"Trajectory interval: {cfg['trajectory_interval']} iterations "
                    f"({cfg['trajectory_interval'] * exchange_ps:.3f} ps)")
        logger.info(f"Checkpoint interval: {cfg['checkpoint_interval']} iterations "
                    f"({cfg['checkpoint_interval'] * exchange_ps / 1000.0:.3f} ns)")
        logger.info(f"[v4.4] Periodic boundary conditions: {cfg['periodic_boundary']}")
        logger.info(f"[v4.4] Preserve input box shape: {not cfg['use_cubic_box']}")
        logger.info(f"[v4.4] Non-standard local-hot REST2 extension: {cfg['local_heating_enabled']}")
        if cfg['local_heating_enabled']:
            logger.warning("[v4.4] This is a non-standard local-hot REST2 Hamiltonian")
            logger.info(f"[v4.4] Hot residues: {cfg['hot_residue_ids']}")
            logger.info(f"[v4.4] Local scaling scheme: {cfg['local_scaling_scheme']}")
            logger.info(f"[v4.4] Lambda_hot endpoint multiplier: {cfg['lambda_hot']} "
                        f"(local-hot effective T range = {cfg['T_ref']:.1f}-"
                        f"{cfg['T_max']/cfg['lambda_hot']:.1f} K; reference state unbiased)")
        logger.info(f"[v4.4] Equilibration restraints: {cfg.get('equilibration_restraints', False)}")
        logger.info(f"[v4.4] Equilibration steps (NPT/restrained NVT/unrestrained NVT): "
                    f"{cfg['npt_equilibration_steps']}/{cfg['nvt_equilibration_steps']}/"
                    f"{cfg['unrestrained_equilibration_steps']}")
        if args.restore:
            logger.info(f"Restored from: {args.restore}")
            logger.info(f"Starting from checkpoint iteration: {restore_data['start_iteration']}")
        logger.info(f"Output: {exp_dir}")
        logger.info(f"{'='*70}")

        results = run_rest2(cfg, logger, restore_data)

        logger.info(f"\n[SUCCESS] Experiment completed!")
        logger.info(f"Results saved to: {exp_dir}")

    except Exception as e:
        logger.error(f"\n[FATAL ERROR] {str(e)}")
        logger.error(traceback.format_exc())
        raise
