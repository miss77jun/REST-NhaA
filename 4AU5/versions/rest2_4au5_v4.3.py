#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
REST2 (Replica Exchange with Solute Tempering v2) 分子动力学模拟 — v4.3
4AU5 蛋白 + CHARMM36m + TIP3P + 0.15M NaCl

======================================================================
版本: v4.3 (2026-09-18)

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
  1. lambda_hot 默认值从 0.5 提升到 0.75
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
     --lambda_hot              局部加热 lambda 值 (默认 0.75)
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

__version__ = "4.3"
__date__ = "2026-09-18"

import os
os.environ['CUDA_CACHE_PATH'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cuda_cache')
os.environ['CUDA_CACHE_MAXSIZE'] = '2147483648'
os.makedirs(os.environ['CUDA_CACHE_PATH'], exist_ok=True)

import sys, time, copy, logging, datetime, traceback, json, hashlib, argparse
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
    "n_iterations":     25000,
    "checkpoint_interval": 500,
    "npt_equilibration_steps": 2500000,
    "nvt_equilibration_steps": 250000,
    "unrestrained_equilibration_steps": 250000,

    "timestep":         2.0,
    "friction":         1.0,

    "platform_preference": ["CUDA", "OpenCL", "CPU"],
    "use_barostat":     False,

    "version":          __version__,

    "cmap_scaling_mode": "scaled_maps",

    # 标准 REST2 默认只缩放完整溶质；局部热区属于非标准扩展，默认关闭
    "local_heating_enabled": False,
    "hot_residue_ids": ["72", "167", "168", "344", "345"],
    "hot_residue_names": {
        "72": "PHE",
        "167": "ALA",
        "168": "ILE",
        "344": "PHE",
        "345": "ILE"
    },
    # v4.0: lambda_hot 从 0.5 提升到 0.75
    #   T_max=350K 时, 热残基等效温度:
    #     lambda_hot=0.5  -> T_eff = 350/0.5 = 700K  (过于激进)
    #     lambda_hot=0.75 -> T_eff = 350/0.75 = 467K (适中)
    "lambda_hot": 0.75,

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
      molecule_bonds: 每个分子的键连接列表
      atom_to_molecule: 原子索引到分子ID的映射
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
                    logger.info(f"[v4.3] Found {len(bonds)} bonds from HarmonicBondForce")
                    break

    logger.info(f"[v4.3] Building molecule topology from {len(bonds)} bonds...")

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

    molecule_bonds = {mol_id: [] for mol_id in molecule_atoms}
    for p1, p2 in bonds:
        mol_id = atom_to_molecule.get(p1)
        if mol_id is not None:
            molecule_bonds[mol_id].append((p1, p2))

    logger.info(f"[v4.3] Molecule topology: {len(molecule_atoms)} molecules found")
    for mol_id, atoms in sorted(molecule_atoms.items(), key=lambda x: -len(x[1]))[:5]:
        logger.info(f"  Molecule {mol_id}: {len(atoms)} atoms, {len(molecule_bonds[mol_id])} bonds")

    return molecule_bonds, atom_to_molecule, molecule_atoms


def make_whole_positions(positions, molecule_bonds, molecule_atoms,
                        box_lengths, box_angles=None, skip_small_molecules=True,
                        small_molecule_threshold=10):
    """
    v4.0: 优化版 make_whole 操作

    改进:
      1. 跳过小分子 (水、离子等单残基分子), 仅对大分子 (蛋白、脂质) 执行
      2. P2-1修复: 使用预构建邻接表替代遍历所有键, 复杂度从O(N_bonds²)降到O(N_atoms)

    参数:
      positions: Vec3数组 (OpenMM positions)
      molecule_bonds: 每个分子的键连接
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

        # P2-1修复: 预构建邻接表，避免遍历所有键
        adjacency = {}
        for p1, p2 in molecule_bonds.get(mol_id, []):
            if p1 not in adjacency:
                adjacency[p1] = []
            if p2 not in adjacency:
                adjacency[p2] = []
            adjacency[p1].append(p2)
            adjacency[p2].append(p1)

        anchor_idx = atom_indices[0]

        visited = {anchor_idx}
        queue = [anchor_idx]

        while queue:
            current = queue.pop(0)
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
    parser = argparse.ArgumentParser(description='REST2 Molecular Dynamics Simulation v4.3')

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
                        help=f'Lambda for local heating (default: {CONFIG["lambda_hot"]}). '
                             f'Lower = more aggressive. T_eff = T/lambda. '
                             f'0.75 -> 467K@350K; 0.5 -> 700K@350K (too aggressive)')
    local_group = parser.add_mutually_exclusive_group()
    local_group.add_argument('--local_heating', action='store_true',
                             help='Enable non-standard local-hot extension')
    local_group.add_argument('--no_local_heating', action='store_true',
                             help='Disable local-hot extension (standard REST2; default)')

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
    if not np.isfinite(cfg["rg_equilibration_fraction"]) or not (0 <= cfg["rg_equilibration_fraction"] < 1):
        raise ValueError("rg_equilibration_fraction 必须有限且满足 0 <= value < 1")


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
            self.experiment_id = f"4au5_rest2_v4.3_{ts}"
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
        self.logger = logging.getLogger(f"REST2_v4.3_4au5_{self.experiment_id}")
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

        logger.info(f"[v4.3] Box from step5_charmm2omm.str: {box_dims[0]:.6f} x {box_dims[1]:.6f} x {box_dims[2]:.6f} Å")
        return box_dims, box_angles

    except Exception as e:
        logger.warning(f"[v4.3] Could not read boxinfo file: {e}")

    logger.warning("[v4.3] Using default box size")
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
            logger.info(f"[v4.3] Setting cubic periodic box: {box_size:.2f} x {box_size:.2f} x {box_size:.2f} Å")
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
                logger.info(f"[v4.3] Setting orthogonal periodic box: {a:.2f} x {b:.2f} x {c:.2f} Å")
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
                logger.info(f"[v4.3] Setting triclinic periodic box: {a:.2f} x {b:.2f} x {c:.2f} Å")

    system.setDefaultPeriodicBoxVectors(*vectors)

    return box_size


def build_base_system(cfg, logger):
    """
    构建CHARMM36m基础系统(v4.0: 与v3.3一致 + 残基映射记录)
    """
    logger.info("=== Building CHARMM36m Force Field System (v4.3) ===")
    logger.info("[v4.3] Periodic Boundary Conditions: ENABLED")
    logger.info(f"[v4.3] Cubic Box: {cfg.get('use_cubic_box', False)}")

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
        logger.info(f"\n[v4.3] Residue Mapping:")
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
    logger.info(f"[v4.3] PSF box set: {box_dims[0]:.6f} x {box_dims[1]:.6f} x {box_dims[2]:.6f} Å")

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
            logger.info(f"[v4.3] Force-switch vdW enabled: r_on=1.0nm, r_off=1.2nm (Force {i})")

    if not vdw_switching_enabled:
        logger.warning("[v4.3] No NonbondedForce found for Force-switch vdW!")

    if cfg.get("periodic_boundary", True):
        box_size = setup_periodic_box_vectors(
            system, box_dims, box_angles,
            use_cubic=cfg.get("use_cubic_box", False),
            logger=logger
        )
    else:
        box_size = max(box_dims)
        logger.warning("[v4.3] Periodic boundary vectors NOT explicitly set (legacy mode)")

    n_atoms = system.getNumParticles()
    n_forces = system.getNumForces()

    # PDB 无 CRYST1 时 topology 无盒子, DCDFile 会丢弃 unit-cell 记录
    pdb.topology.setPeriodicBoxVectors(system.getDefaultPeriodicBoxVectors())
    logger.info(f"[v4.3] Topology periodic box vectors set (required for DCD unit-cell records)")

    logger.info(f"System: {n_atoms} atoms, {n_forces} forces")
    logger.info(f"[v4.3] Periodic box: {original_box[0]:.6f} x {original_box[1]:.6f} x {original_box[2]:.6f} Å")

    logger.info(f"\n[v4.3] Force Field Details:")
    logger.info(f"  Nonbonded method: PME")
    logger.info(f"  Nonbonded cutoff: 1.2 nm")
    logger.info(f"  Force-switch vdW: {'YES (r_on=1.0nm)' if vdw_switching_enabled else 'NO'}")
    logger.info(f"  Constraints: HBonds")
    logger.info(f"  Rigid water: YES")
    logger.info(f"  PME ewald tolerance: 0.0005")

    cmap_count = 0
    cmap_idx = None
    for i in range(n_forces):
        force = system.getForce(i)
        ftype = type(force).__name__
        if isinstance(force, openmm.CMAPTorsionForce):
            cmap_count = force.getNumTorsions()
            cmap_idx = i
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

    return system, psf, pdb, box_size, params, cmap_idx, original_box


def detect_ions(topology, logger):
    """检测系统中的离子种类"""
    detected_ions = set()
    for atom in topology.atoms():
        resname = atom.residue.name.upper()
        if resname in ION_RESNAME_PATTERNS:
            detected_ions.add(resname)

    if detected_ions:
        logger.info(f"[v4.3] Detected ions: {detected_ions}")
    else:
        logger.info("[v4.3] No ions detected in system")

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
        logger.info(f"[v4.3] Using segment-based solute identification (attribute: {segment_attr})")
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
        logger.info("[v4.3] Using residue-name-based solute identification")
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
    logger.info(f"[v4.3] Solute: {n_solute} atoms ({len(solute_residues)} residues), "
               f"Solvent: {n_solvent} atoms, Ratio: {n_solute/n_total*100:.1f}%")

    return solute_atoms


def identify_hot_residue_atoms(topology, cfg, logger):
    hot_atoms = {}

    residue_atoms = {}
    for atom in topology.atoms():
        resname = atom.residue.name.upper()
        if resname not in AA_RESNAMES:
            continue
        res_id = atom.residue.id
        if res_id not in residue_atoms:
            residue_atoms[res_id] = []
        residue_atoms[res_id].append(atom.index)

    for res_id in cfg["hot_residue_ids"]:
        if res_id in residue_atoms:
            hot_atoms[res_id] = set(residue_atoms[res_id])
            logger.info(f"Hot residue {res_id} ({cfg['hot_residue_names'][res_id]}): "
                       f"{len(hot_atoms[res_id])} atoms identified")
        else:
            logger.warning(f"Hot residue {res_id} not found in topology")

    all_hot_atoms = set()
    for atoms in hot_atoms.values():
        all_hot_atoms.update(atoms)

    logger.info(f"[v4.3] Local heating enabled for {len(hot_atoms)} residues")
    logger.info(f"[v4.3] Total hot atoms: {len(all_hot_atoms)}")
    logger.info(f"[v4.3] Lambda_hot: {cfg['lambda_hot']}")

    # v4.0: 计算并显示有效温度
    T_max = cfg["T_max"]
    T_ref = cfg["T_ref"]
    lam_hot = cfg["lambda_hot"]
    T_eff_ref = T_ref / lam_hot
    T_eff_max = T_max / lam_hot
    logger.info(f"[v4.3] Effective temperatures for hot residues:")
    logger.info(f"  At T_ref={T_ref}K: T_eff = {T_eff_ref:.0f}K")
    logger.info(f"  At T_max={T_max}K: T_eff = {T_eff_max:.0f}K")

    # v4.0: 安全检查
    if lam_hot < 0.5:
        logger.warning(f"[v4.3] WARNING: lambda_hot={lam_hot} is very aggressive!")
        logger.warning(f"  T_eff at T_max = {T_eff_max:.0f}K may cause protein unfolding!")
        logger.warning(f"  Recommended: lambda_hot >= 0.6")
    elif lam_hot < 0.6:
        logger.info(f"[v4.3] Note: lambda_hot={lam_hot} is moderately aggressive. T_eff_max={T_eff_max:.0f}K")

    return all_hot_atoms, hot_atoms


def identify_ca_atoms(topology, logger):
    """
    v4.0: 识别蛋白 CA 原子索引, 用于 RMSD 监控
    """
    ca_indices = []
    for atom in topology.atoms():
        if atom.name == 'CA' and atom.residue.name in AA_RESNAMES:
            ca_indices.append(atom.index)

    logger.info(f"[v4.3] Identified {len(ca_indices)} CA atoms for RMSD monitoring")
    return ca_indices


def create_scaled_cmap_force(original_force, solute_atoms, hot_atoms, lam, lam_hot, logger):
    n_maps = original_force.getNumMaps()
    n_torsions = original_force.getNumTorsions()

    logger.info(f"  [CMAP] Creating scaled CMAP force with λ={lam:.4f}")
    logger.info(f"  [CMAP] Original: {n_maps} maps, {n_torsions} torsions")

    # 标准 REST2 的溶质 CMAP 乘 λ；hot map 是保留的局部扩展。
    scaled_maps = {}
    hot_maps = {}
    original_maps = {}
    new_force = openmm.CMAPTorsionForce()

    for map_idx in range(n_maps):
        size, energy_grid = original_force.getMapParameters(map_idx)
        scaled_maps[map_idx] = new_force.addMap(size, energy_grid * lam)
        hot_maps[map_idx] = new_force.addMap(size, energy_grid * lam * lam_hot)
        original_maps[map_idx] = new_force.addMap(size, energy_grid)

    n_scaled = 0
    n_hot = 0
    n_preserved = 0

    for torsion_idx in range(n_torsions):
        params = original_force.getTorsionParameters(torsion_idx)
        orig_map_idx = params[0]
        atoms = params[1:9]

        if all(a in solute_atoms for a in atoms):
            if any(a in hot_atoms for a in atoms):
                new_map_idx = hot_maps[orig_map_idx]
                n_hot += 1
            else:
                new_map_idx = scaled_maps[orig_map_idx]
                n_scaled += 1
        else:
            new_map_idx = original_maps[orig_map_idx]
            n_preserved += 1
        new_force.addTorsion(new_map_idx, *atoms)

    logger.info(f"  [CMAP] Result: {n_scaled} standard REST2, {n_hot} local-hot, {n_preserved} preserved")

    return new_force, n_scaled


def create_rest2_system_with_local_heating(base_system, solute_atoms, hot_atoms, lam_global,
                                            lam_hot, params, cmap_idx, cfg, logger):
    system = copy.deepcopy(base_system)
    sqrt_lam_global = np.sqrt(lam_global)
    lam_hot_effective = lam_global * lam_hot

    n_forces = system.getNumForces()

    logger.info(f"  Creating REST2 system with local heating:")
    logger.info(f"    λ_global={lam_global:.4f}, λ_hot={lam_hot:.4f}")
    logger.info(f"    λ_hot_effective={lam_hot_effective:.4f}")

    cmap_forces_to_replace = []

    for i in range(n_forces):
        force = system.getForce(i)

        if isinstance(force, openmm.CustomNonbondedForce):
            expression = force.getEnergyFunction()
            expression_lower = expression.lower()
            if "acoef" in expression_lower and "bcoef" in expression_lower:
                particle_scales = np.ones(force.getNumParticles())
                for j in range(force.getNumParticles()):
                    if j in hot_atoms:
                        particle_scales[j] = lam_hot_effective
                    elif j in solute_atoms:
                        particle_scales[j] = lam_global

                original_parameter_count = force.getNumPerParticleParameters()
                force.addPerParticleParameter("restScale")
                for j in range(force.getNumParticles()):
                    parameters = list(force.getParticleParameters(j))
                    if len(parameters) != original_parameter_count:
                        raise ValueError(
                            f"CustomNonbondedForce 粒子 {j} 参数数 {len(parameters)} "
                            f"与声明数 {original_parameter_count} 不一致"
                        )
                    force.setParticleParameters(j, parameters + [particle_scales[j]])

                terms = expression.split(";", 1)
                scaled_energy = f"sqrt(restScale1*restScale2)*({terms[0]})"
                if len(terms) == 2:
                    scaled_energy += ";" + terms[1]
                force.setEnergyFunction(scaled_energy)
                logger.info(
                    f"    CustomNonbondedForce CHARMM LJ: 保留 {force.getNumExclusions()} 个排除和原 tabulated functions，"
                    f"新增 restScale（原每粒子参数数={original_parameter_count}）"
                )

        elif isinstance(force, openmm.NonbondedForce):
            n_normal = 0
            n_hot = 0
            particle_scales = np.ones(force.getNumParticles())
            for j in range(force.getNumParticles()):
                q, sig, eps = force.getParticleParameters(j)
                if j in hot_atoms:
                    particle_scales[j] = lam_hot_effective
                    force.setParticleParameters(j, q * np.sqrt(lam_hot_effective), sig,
                                                eps * lam_hot_effective)
                    n_hot += 1
                elif j in solute_atoms:
                    particle_scales[j] = lam_global
                    force.setParticleParameters(j, q * sqrt_lam_global, sig, eps * lam_global)
                    n_normal += 1

            # exception 使用与粒子参数相同的几何组合尺度。普通溶质-hot
            # 交叉项因此为 sqrt(λ * λ_hot_effective)，而不是整项 hot 尺度。
            for j in range(force.getNumExceptions()):
                p1, p2, q12, sig, eps = force.getExceptionParameters(j)
                pair_scale = np.sqrt(particle_scales[p1] * particle_scales[p2])
                force.setExceptionParameters(j, p1, p2, q12 * pair_scale, sig, eps * pair_scale)

            logger.info(f"    NonbondedForce: {n_normal} normal + {n_hot} hot particles")

        elif isinstance(force, openmm.PeriodicTorsionForce):
            for j in range(force.getNumTorsions()):
                p1, p2, p3, p4, per, phase, k = force.getTorsionParameters(j)
                if all(x in solute_atoms for x in [p1, p2, p3, p4]):
                    if any(x in hot_atoms for x in [p1, p2, p3, p4]):
                        force.setTorsionParameters(j, p1, p2, p3, p4, per, phase, k * lam_hot_effective)
                    else:
                        force.setTorsionParameters(j, p1, p2, p3, p4, per, phase, k * lam_global)

        elif isinstance(force, openmm.CustomTorsionForce):
            param_names = [force.getPerTorsionParameterName(k) for k in range(force.getNumPerTorsionParameters())]
            k_idx = next((k for k, n in enumerate(param_names) if 'k' in n.lower()), None)
            if k_idx is not None:
                for j in range(force.getNumTorsions()):
                    p1, p2, p3, p4, pars = force.getTorsionParameters(j)
                    if all(x in solute_atoms for x in [p1, p2, p3, p4]):
                        pars = list(pars)
                        if any(x in hot_atoms for x in [p1, p2, p3, p4]):
                            pars[k_idx] *= lam_hot_effective
                        else:
                            pars[k_idx] *= lam_global
                        force.setTorsionParameters(j, p1, p2, p3, p4, pars)

        elif isinstance(force, openmm.CMAPTorsionForce):
            if cfg["cmap_scaling_mode"] == "scaled_maps":
                scaled_cmap, n_cmap_scaled = create_scaled_cmap_force(
                    force, solute_atoms, hot_atoms, lam_global, lam_hot, logger
                )
                cmap_forces_to_replace.append((i, scaled_cmap))

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
        logger.info(f"[v4.3] Temperature spacing: min={min(dTs):.2f}K, max={max(dTs):.2f}K, avg={np.mean(dTs):.2f}K")

    return temps, lams


def analyze_walker_round_trips(walker_label_history, n_replicas=None):
    """按 Hamiltonian 槽位中的 walker 标签历史统计两种方向的完整端点往返。"""
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
    top = n_replicas - 1
    for walker in range(n_replicas):
        endpoint_visits = []
        for slot in walker_slots[:, walker]:
            if slot in (0, top) and (not endpoint_visits or endpoint_visits[-1] != slot):
                endpoint_visits.append(int(slot))
        for first, middle, last in zip(endpoint_visits, endpoint_visits[1:], endpoint_visits[2:]):
            if (first, middle, last) == (0, top, 0):
                counts[walker, 0] += 1
            elif (first, middle, last) == (top, 0, top):
                counts[walker, 1] += 1

    per_walker = counts.sum(axis=1)
    return {
        'walker_slots': walker_slots,
        'directional_counts': counts,
        'per_walker': per_walker,
        'total': int(per_walker.sum()),
        'walkers_with_round_trip': int(np.count_nonzero(per_walker)),
    }


def log_round_trip_summary(summary, logger):
    logger.info("  Walker round trips (0->top->0 / top->0->top):")
    for walker, (up_down, down_up) in enumerate(summary['directional_counts']):
        logger.info(f"    Walker {walker:2d}: {up_down:d} / {down_up:d} "
                    f"(total {summary['per_walker'][walker]:d})")
    if summary['total'] == 0:
        logger.info("  Round-trip: 未观察到完整往返；当前采样长度不足以据此判断采样错误")
    else:
        logger.info(f"  Round-trip total: {summary['total']}; walkers with >=1: "
                    f"{summary['walkers_with_round_trip']}/{len(summary['per_walker'])}")


def save_full_checkpoint(cfg, iteration, contexts, energy_history, exchange_attempts,
                         exchange_accepts, temps, lams, delta_history, logger,
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

    save_dict = {
        'iteration': iteration,
        'energies': np.array(energy_history),
        'attempts': exchange_attempts,
        'accepts': exchange_accepts,
        'temps': temps,
        'lams': lams,
        'lambda_hot': cfg["lambda_hot"],
        'local_heating_enabled': cfg["local_heating_enabled"],
        'hot_residue_ids': cfg["hot_residue_ids"],
        'T_ref': cfg["T_ref"],
        'n_replicas': len(temps),
        'n_atoms': contexts[0].getSystem().getNumParticles(),
        'psf_file': cfg["psf_file"],
        'pdb_file': cfg["pdb_file"],
        'timestep': cfg["timestep"],
        'friction': cfg["friction"],
        'walker_history_start_iteration': walker_history_start_iteration,
        'walker_label_semantics': 'slot_to_walker',
    }

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

    checkpoint_cfg = {}
    for key in ('local_heating_enabled', 'lambda_hot', 'hot_residue_ids', 'T_ref', 'T_max',
                'n_replicas', 'n_atoms', 'psf_file', 'pdb_file', 'timestep', 'friction'):
        if key in data:
            value = data[key]
            checkpoint_cfg[key] = value.item() if np.asarray(value).shape == () else value.tolist()
    required_hamiltonian = {'local_heating_enabled', 'lambda_hot', 'hot_residue_ids', 'T_ref', 'n_atoms'}
    missing_hamiltonian = sorted(required_hamiltonian - set(checkpoint_cfg))
    if missing_hamiltonian:
        logger.warning("旧 checkpoint 缺少 Hamiltonian 元数据 (%s)；只能校验已有字段",
                       ", ".join(missing_hamiltonian))
    checkpoint_cfg['n_replicas'] = n_replicas
    checkpoint_cfg['n_atoms'] = next(iter(atom_counts))
    checkpoint_cfg['T_ref'] = float(temps[0])
    checkpoint_cfg['T_max'] = float(temps[-1])

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
        logger.info(f"[v4.3] DCD verification: {dcd_file}")
        logger.info(f"[v4.3]   Frames: {n_frames}")

        if hasattr(u.trajectory, 'dimensions'):
            dims = u.trajectory.dimensions
            if dims is not None:
                logger.info(f"[v4.3]   Box dimensions: {dims[:3]} Å")
                logger.info(f"[v4.3]   Box angles: {dims[3:]} degrees")
                logger.info(f"[v4.3]   Periodic boundary: VERIFIED")
            else:
                logger.warning(f"[v4.3]   Box dimensions: NOT FOUND in DCD file")

        return True

    except ImportError:
        logger.warning("[v4.3] MDAnalysis not available for DCD verification")
        return False
    except Exception as e:
        logger.error(f"[v4.3] DCD verification failed: {e}")
        return False


def run_rest2(cfg, logger, restore_data=None):
    kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
    kB_val = kB.value_in_unit(unit.kilojoule_per_mole / unit.kelvin)

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
        logger.info(f"Hot residues: {cfg['hot_residue_ids']}")
        logger.info(f"Lambda_hot: {cfg['lambda_hot']}")
    logger.info(f"[v4.3] Periodic boundary: {cfg.get('periodic_boundary', True)}")
    logger.info(f"[v4.3] Preserve input box shape: {not cfg.get('use_cubic_box', False)}")
    logger.info(f"[v4.3] Equilibration restraints: {cfg.get('equilibration_restraints', False)}")
    logger.info(f"[v4.3] NPT/NVT/unrestrained equilibration steps: "
                f"{cfg['npt_equilibration_steps']}/{cfg['nvt_equilibration_steps']}/"
                f"{cfg['unrestrained_equilibration_steps']}")
    if cfg.get("save_trajectory", False):
        trajectory_ps = exchange_ps * cfg["trajectory_interval"]
        logger.info(f"Trajectory interval: {cfg['trajectory_interval']} iterations ({trajectory_ps:.3f} ps)")
    logger.info(f"Checkpoint interval: {cfg['checkpoint_interval']} iterations")
    logger.info(f"[v4.3] RMSD monitoring: every {cfg.get('rmsd_monitor_interval', 100)} iterations")
    logger.info(f"{'='*70}")

    base_system, psf, pdb, box_size, params, cmap_idx, box_dims = build_base_system(cfg, logger)

    detected_ions = detect_ions(pdb.topology, logger)
    if detected_ions:
        global SOLVENT_RESNAMES
        SOLVENT_RESNAMES.update(detected_ions)
        logger.info(f"[v4.3] Added {len(detected_ions)} ion types to solvent residues")

    solute_atoms = identify_solute_atoms(pdb.topology, logger)

    if cfg["local_heating_enabled"]:
        hot_atoms, hot_atoms_by_residue = identify_hot_residue_atoms(pdb.topology, cfg, logger)
    else:
        hot_atoms = set()
        hot_atoms_by_residue = {}
        logger.info("[v4.3] Local heating DISABLED")

    # v4.0: 识别 CA 原子用于 RMSD 监控
    ca_atom_indices = identify_ca_atoms(pdb.topology, logger)

    # v4.0: 保存初始 CA 坐标作为 RMSD 参照
    initial_ca_coords = extract_ca_positions(pdb.positions, ca_atom_indices)
    logger.info(f"[v4.3] Saved initial CA coordinates for RMSD reference ({len(initial_ca_coords)} atoms)")

    n_rep = cfg["n_replicas"]
    T_ref = cfg["T_ref"]
    T_max = cfg["T_max"]
    lam_hot = cfg["lambda_hot"]

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
        logger.info(f"[v4.3] Position restraints enabled (k={cfg['restraint_force_constant']} kJ/mol/nm²)")
        logger.info(f"[v4.3] Restraints will be applied to protein CA atoms during equilibration, then removed")

    for i, (T, lam_global) in enumerate(zip(temps, lams)):
        if not restore_data or i < 5:
            logger.info(f"Replica {i}: T={T:.1f}K, λ_global={lam_global:.4f}")

        if cfg["local_heating_enabled"]:
            rep_sys, cmap_replacements = create_rest2_system_with_local_heating(
                base_system, solute_atoms, hot_atoms, lam_global, lam_hot,
                params, cmap_idx, cfg, logger
            )
        else:
            rep_sys, cmap_replacements = create_rest2_system_with_local_heating(
                base_system, solute_atoms, set(), lam_global, 1.0,
                params, cmap_idx, cfg, logger
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
                logger.info(f"[v4.3] Added position restraint force ({restraint.getNumParticles()} CA atoms) to replica systems")
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
        ctx = openmm.Context(rep_sys, integ, platform)

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
            ctx.setVelocitiesToTemperature(T_ref*unit.kelvin)

        contexts.append(ctx)

    if not restore_data:
        logger.info(f"\n=== Equilibration ===")

        # replica_systems[0] 已包含配置启用的位置约束；复制后用于共同盒膜 NPT。
        # 将 NPT 构型和盒子分发给所有 replica，保证后续 NVT 交换使用同一周期盒。
        npt_steps = cfg["npt_equilibration_steps"]
        npt_ns = npt_steps * cfg["timestep"] / 1.0e6
        logger.info(f"[v4.3] Phase 1: common-box NPT equilibration ({npt_steps} steps, {npt_ns:.3f} ns)...")
        state = contexts[0].getState(getPositions=True, getVelocities=True, enforcePeriodicBox=True)
        npt_sys = copy.deepcopy(replica_systems[0])
        try:
            barostat = openmm.MonteCarloMembraneBarostat(
                1*unit.bar, 0*unit.bar*unit.nanometer, T_ref*unit.kelvin,
                openmm.MonteCarloMembraneBarostat.XYIsotropic,
                openmm.MonteCarloMembraneBarostat.ZFree
            )
            logger.info("[v4.3] Using semi-isotropic MonteCarloMembraneBarostat")
        except (AttributeError, TypeError):
            barostat = openmm.MonteCarloBarostat(1*unit.bar, T_ref*unit.kelvin)
            logger.warning("[v4.3] Compatible MonteCarloMembraneBarostat unavailable; using isotropic barostat")
        npt_sys.addForce(barostat)
        npt_integ = openmm.LangevinMiddleIntegrator(
            T_ref*unit.kelvin, cfg["friction"]/unit.picosecond, cfg["timestep"]*unit.femtoseconds
        )
        npt_ctx = openmm.Context(npt_sys, npt_integ, platform)
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
        logger.info(f"[v4.3] Common NPT box = {box_a:.2f} x {box_b:.2f} x {box_c:.2f} Å")
        del npt_ctx, npt_integ, npt_sys
        logger.info(f"[v4.3] NPT equilibration complete. Starting per-replica NVT equilibration.")

        nvt_steps = cfg["nvt_equilibration_steps"]
        nvt_ns = nvt_steps * cfg["timestep"] / 1.0e6
        logger.info(f"[v4.3] Phase 2: each replica at physical temperature {T_ref:.1f} K "
                    f"under its own Hamiltonian ({nvt_steps} steps, {nvt_ns:.3f} ns per replica)...")

        if cfg.get("equilibration_restraints", False):
            logger.info(f"[v4.3] Position restraints ACTIVE during NPT and NVT equilibration")
            logger.info(f"[v4.3] Restraints will be removed after equilibration by rebuilding contexts")

        for i, (ctx, T) in enumerate(zip(contexts, temps)):
            integ = ctx.getIntegrator()
            integ.setTemperature(T_ref*unit.kelvin)
            integ.setStepSize(cfg["timestep"]*unit.femtoseconds)
            integ.step(nvt_steps)

            if i < 5:
                state = ctx.getState(getEnergy=True)
                E = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                logger.info(f"  Replica {i} (effective solute T={T:.1f}K, bath={T_ref:.1f}K): "
                            f"Equilibrated E={E:.1f} kJ/mol")

        # P0-3修复: 平衡后移除位置约束，重建Context
        if cfg.get("equilibration_restraints", False):
            logger.info(f"\n[v4.3] Removing position restraints and rebuilding contexts...")
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
                new_ctx = openmm.Context(rep_sys, integ, platform)
                new_ctx.setPositions(pos)
                new_ctx.setVelocities(vel)
                new_ctx.setPeriodicBoxVectors(*box)
                new_contexts.append(new_ctx)
                del ctx

            contexts = new_contexts
            release_steps = cfg["unrestrained_equilibration_steps"]
            release_ns = release_steps * cfg["timestep"] / 1.0e6
            logger.info(f"[v4.3] Position restraints removed. Running unrestrained NVT equilibration "
                        f"({release_steps} steps, {release_ns:.3f} ns per replica)...")
            for ctx in contexts:
                ctx.getIntegrator().step(release_steps)
            logger.info(f"[v4.3] Unrestrained NVT equilibration complete; production contexts are ready.")
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
            dt = cfg["timestep"] * cfg["trajectory_interval"] * cfg["n_steps_per_move"] / 1000.0
            dcd_file_obj = open(dcd_file_path, 'wb')
            dcd_writer = app.DCDFile(
                dcd_file_obj, pdb.topology, dt,
                firstStep=start_iteration * cfg["n_steps_per_move"]
            )
            dcd_writers.append(dcd_writer)
            dcd_files.append(dcd_file_obj)
            cfg.setdefault('_trajectory_paths', []).append(dcd_file_path)
            if i < 5:
                logger.info(f"  Replica {i}: {dcd_file_path}")
        logger.info(f"  Trajectory interval: Every {cfg['trajectory_interval']} iterations")
        logger.info(f"  Trajectory mode: Write (new file)")
        logger.info(f"  PBC mode: enforcePeriodicBox + make_whole (v4.3)")

    # 构建分子拓扑，用于make_whole操作
    logger.info(f"\n=== Building Molecule Topology for PBC ===")
    molecule_bonds, atom_to_molecule, molecule_atoms = build_molecule_topology(
        base_system, pdb.topology, logger
    )
    n_molecules = len(molecule_atoms)
    logger.info(f"[v4.3] Make-whole ready: {n_molecules} molecules")

    box_vecs_init = base_system.getDefaultPeriodicBoxVectors()
    box_lengths_nm = [
        box_vecs_init[0][0].value_in_unit(unit.nanometer),
        box_vecs_init[1][1].value_in_unit(unit.nanometer),
        box_vecs_init[2][2].value_in_unit(unit.nanometer)
    ]
    logger.info(f"[v4.3] Box lengths for make-whole: {box_lengths_nm[0]:.4f} x {box_lengths_nm[1]:.4f} x {box_lengths_nm[2]:.4f} nm")

    def save_dcd_frame(ctx, writer, box_lens, mol_bonds, mol_atoms):
        """保存DCD帧，包含PBC包裹和make_whole"""
        state = ctx.getState(getPositions=True, enforcePeriodicBox=True)
        raw_pos = state.getPositions()
        box_vecs = state.getPeriodicBoxVectors()

        # NPT 平衡后盒子可能已改变；使用当前完整盒矩阵兼容正交和三斜盒
        box_matrix = np.array([
            [v.x, v.y, v.z] for v in box_vecs.value_in_unit(unit.nanometer)
        ])

        # v4.0: 优化版 make_whole (跳过小分子)
        if mol_bonds:
            processed_pos = make_whole_positions(
                raw_pos, mol_bonds, mol_atoms, box_matrix,
                skip_small_molecules=True, small_molecule_threshold=10
            )
        else:
            processed_pos = raw_pos

        # #region debug-point A-E:dcd-protein-coordinates
        if not getattr(writer, '_trae_debugged', False):
            _raw = np.asarray([[p.x, p.y, p.z] for p in raw_pos]); _out = np.asarray([[p.x, p.y, p.z] for p in processed_pos]); _idx = np.asarray(sorted(solute_atoms), dtype=int); _frac = _out[_idx] @ np.linalg.inv(box_matrix); _payload = {"sessionId":"vmd-empty-protein","runId":"pre-fix","hypothesisId":"A-E","location":"rest2_4au5_v4.0.py:save_dcd_frame","msg":"[DEBUG] DCD写入前蛋白质坐标诊断","data":{"system_atoms":int(len(_out)),"topology_atoms":int(pdb.topology.getNumAtoms()),"solute_atoms":int(len(_idx)),"largest_molecule_atoms":int(max(map(len, mol_atoms.values()))),"raw_min":_raw[_idx].min(axis=0).tolist(),"raw_max":_raw[_idx].max(axis=0).tolist(),"out_min":_out[_idx].min(axis=0).tolist(),"out_max":_out[_idx].max(axis=0).tolist(),"out_center":_out[_idx].mean(axis=0).tolist(),"box_matrix":box_matrix.tolist(),"solute_fraction_inside_primary_box":float(np.mean(np.all((_frac >= 0) & (_frac < 1), axis=1))),"finite":bool(np.isfinite(_out[_idx]).all()),"unique_rounded_solute_coords":int(len(np.unique(np.round(_out[_idx], 5), axis=0)))}}; exec("try:\n import json, urllib.request\n urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:7777/event',data=json.dumps(_payload).encode(),headers={'Content-Type':'application/json'}),timeout=1).read()\nexcept Exception:\n pass"); writer._trae_debugged = True
        # #endregion

        writer.writeModel(processed_pos, periodicBoxVectors=box_vecs)

    # v4.0: RMSD 监控函数
    rmsd_monitor_replicas = cfg.get("rmsd_monitor_replicas", [0, 15, 31])
    rmsd_monitor_interval = cfg.get("rmsd_monitor_interval", 100)

    def monitor_rmsd(iteration, contexts, ca_indices, ref_coords):
        """计算并记录指定副本的 RMSD"""
        rmsd_values = {}
        for rep_idx in rmsd_monitor_replicas:
            if rep_idx < len(contexts):
                # P1-2修复: 添加enforcePeriodicBox，防止蛋白跨边界时RMSD异常跳变
                state = contexts[rep_idx].getState(getPositions=True, enforcePeriodicBox=True)
                cur_coords = extract_ca_positions(state.getPositions(), ca_indices)
                rmsd = calculate_rmsd(ref_coords, cur_coords)
                rmsd_values[rep_idx] = rmsd
        return rmsd_values

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
    logger.info(f"[v4.3] Periodic boundary conditions: ACTIVE")
    logger.info(f"[v4.3] Using original box shape from input file")
    logger.info(f"[v4.3] RMSD monitoring: replicas {rmsd_monitor_replicas}, every {rmsd_monitor_interval} iters")
    logger.info(f"{'='*70}\n")

    t0 = time.time()

    # P1-1修复: 写入初始帧（iteration=0）作为DCD的第一帧
    if cfg.get("save_trajectory", False) and dcd_writers and start_iteration == 0:
        logger.info(f"[v4.3] Writing initial frame (iteration 0) to DCD files...")
        for i, ctx in enumerate(contexts):
            save_dcd_frame(
                ctx, dcd_writers[i],
                box_lengths_nm, molecule_bonds, molecule_atoms
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

                accept = delta <= 0 or np.random.random() < np.exp(-delta)
                acceptance_history[m].append(bool(accept))

                if accept:
                    exchange_accepts[m] += 1
                    contexts[m].setVelocitiesToTemperature(T_ref*unit.kelvin)
                    contexts[n].setVelocitiesToTemperature(T_ref*unit.kelvin)
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

            # v4.0: RMSD 监控
            if (iteration + 1) % rmsd_monitor_interval == 0:
                rmsd_vals = monitor_rmsd(iteration + 1, contexts, ca_atom_indices, initial_ca_coords)
                rmsd_history.append({
                    'iteration': iteration + 1,
                    'rmsd': rmsd_vals
                })
                rmsd_str = ", ".join([f"Rep{r}: {v:.2f}Å" for r, v in rmsd_vals.items()])
                logger.info(f"  [v4.3] RMSD @ Iter {iteration+1}: {rmsd_str}")

            if cfg.get("save_trajectory", False) and dcd_writers:
                if (iteration + 1) % cfg["trajectory_interval"] == 0:
                    for i, ctx in enumerate(contexts):
                        save_dcd_frame(
                            ctx, dcd_writers[i],
                            box_lengths_nm, molecule_bonds, molecule_atoms
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
                                    exchange_accepts, temps, lams, delta_history, logger,
                                    rmsd_history=rmsd_history,
                                    acceptance_history=acceptance_history,
                                    walker_labels=walker_labels,
                                    walker_label_history=walker_label_history,
                                    walker_history_start_iteration=walker_history_start_iteration)
                save_checkpoint(cfg, iteration+1, energy_history, exchange_attempts,
                               exchange_accepts, temps, lams, delta_history)
    finally:
        if dcd_writers:
            logger.info(f"\n=== Closing Trajectory Files ===")
            for i, dcd_f in enumerate(dcd_files):
                dcd_f.close()
                if i < 5:
                    logger.info(f"  Replica {i}: DCD file closed")

    # v4.0: 保存 RMSD 历史
    if rmsd_history:
        rmsd_file = os.path.join(cfg["output_dir"], "rmsd_history.npy")
        np.save(rmsd_file, np.array([(r['iteration'], list(r['rmsd'].items()))
                                      for r in rmsd_history], dtype=object), allow_pickle=True)
        logger.info(f"[v4.3] RMSD history saved: {rmsd_file}")

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
            if rate < 15.0:
                logger.warning(f"    Low exchange bottleneck: pair {k}-{k+1} = {rate:.2f}% (<15%)")
            elif rate > 65.0:
                logger.info(f"    Pair {k}-{k+1} = {rate:.2f}% (>65%): Hamiltonian 梯度可能过密")
        else:
            logger.info(f"    Pair {k:2d}-{k+1:2d}: no attempts")
    if final_rates.size:
        logger.info(f"  Exchange rate min/mean/max: {final_rates.min():.2f}% / "
                    f"{final_rates.mean():.2f}% / {final_rates.max():.2f}%")
    logger.info(f"  Effective solute temperature: {T_ref:.1f}-{T_max:.1f} K; "
                f"bath temperature: {T_ref:.1f} K for every replica")
    round_trip_summary = analyze_walker_round_trips(walker_label_history, n_rep)
    log_round_trip_summary(round_trip_summary, logger)

    # v4.0: 最终 RMSD 摘要
    if rmsd_history:
        final_rmsd = rmsd_history[-1]['rmsd']
        logger.info(f"  Final RMSD vs initial structure:")
        for rep_idx, rmsd_val in final_rmsd.items():
            T = temps[rep_idx]
            logger.info(f"    Replica {rep_idx} (effective solute T={T:.0f}K): {rmsd_val:.2f} Å")

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
        'version': cfg['version'],
        'cmap_scaling_mode': cfg['cmap_scaling_mode'],
        'local_heating': cfg['local_heating_enabled'],
        'hot_residue_ids': cfg['hot_residue_ids'],
        'hot_residue_names': cfg['hot_residue_names'],
        'lambda_hot': cfg['lambda_hot'],
        'periodic_boundary': cfg.get('periodic_boundary', True),
        'cubic_box': cfg.get('use_cubic_box', False),
        'box_dims': box_dims,
        'box_size': box_size,
        'rmsd_history': rmsd_history,
        'residue_offset': cfg.get('residue_offset', 9),
    }

    save_final_results(cfg, results, contexts, pdb)
    try:
        generate_scientific_visualizations(results, cfg["output_dir"], logger)
    except ImportError as e:
        logger.warning(f"[v4.3] matplotlib not installed, skipping visualizations: {e}")
    save_scientific_report(cfg, results, logger)
    walker_history_file = os.path.join(cfg["output_dir"], "walker_label_history.npy")
    np.save(walker_history_file, results['walker_label_history'])
    logger.info(f"[v4.3] Walker label history saved: {walker_history_file}")

    # 使用MDAnalysis计算OF gate Rg
    logger.info(f"\n{'='*70}")
    logger.info(f"[v4.3] Calculating OF Gate Rg using MDAnalysis")
    logger.info(f"{'='*70}")
    logger.info(f"  OF gate selection: {OF_GATE_SELECTION}")
    logger.info(f"  Equilibration fraction: {cfg['rg_equilibration_fraction']*100:.0f}%")

    rg_results = calculate_rg_with_mdanalysis(cfg, logger, temps)

    if rg_results:
        generate_rg_visualizations(rg_results, temps, cfg["output_dir"], logger)

    # v4.0: 生成 RMSD 演化图
    if rmsd_history:
        try:
            generate_rmsd_visualizations(rmsd_history, temps, cfg["output_dir"], logger)
        except Exception as e:
            logger.warning(f"[v4.3] Could not generate RMSD visualization: {e}")

    # 验证DCD文件
    if cfg.get("save_trajectory", False):
        logger.info(f"\n[v4.3] Verifying DCD trajectory files...")
        trajectories_dir = cfg.get("trajectories_dir", cfg["output_dir"])
        trajectory_paths = cfg.get('_trajectory_paths')
        for i in range(min(3, n_rep)):
            dcd_file = (trajectory_paths[i] if trajectory_paths
                        else os.path.join(trajectories_dir, f"replica_{i}.dcd"))
            if os.path.exists(dcd_file):
                verify_dcd_file(dcd_file, logger)

    return results


def save_checkpoint(cfg, iteration, energy_history, attempts, accepts, temps, lams, deltas):
    ckpt_file = os.path.join(cfg["output_dir"], f"checkpoint_{iteration}.npz")
    np.savez_compressed(ckpt_file,
        energies=np.array(energy_history),
        attempts=attempts, accepts=accepts,
        temps=temps, lams=lams,
        lambda_hot=cfg["lambda_hot"],
        hot_residue_ids=cfg["hot_residue_ids"],
        **{f"delta_{k}": np.array(v) for k, v in deltas.items() if v})


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
    colors = ['#27ae60' if 20<=r<=40 else '#e74c3c' for r in rates]
    ax.bar(range(n_rep-1), rates, color=colors)
    ax.axhline(20, color='k', ls='--'); ax.axhline(40, color='k', ls='--')
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
    ax.set_xlabel('T (K)'); ax.set_ylabel('Average E (kJ/mol)')
    ax.set_title('(d) Energy vs Temperature', fontweight='bold')
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
    ax.set_xlabel('T (K)'); ax.set_ylabel('Lambda')
    ax.set_title('(f) Lambda Scaling', fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'fig1_analysis.png'), dpi=150)
    plt.close()

    logger.info(f"  Figures saved to {figures_dir}")


def generate_rmsd_visualizations(rmsd_history, temps, output_dir, logger):
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

    logger.info(f"\n[v4.3] Generating RMSD evolution visualization...")

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

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
                label=f'Replica {rep_idx} (T={T:.0f}K)', linewidth=1.5)

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Protein CA RMSD (Å)', fontsize=12)
    ax.set_title('Protein RMSD Evolution vs Initial Structure', fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # 添加参考线
    ax.axhline(3.0, color='green', ls='--', alpha=0.5, label='3Å (stable)')
    ax.axhline(5.0, color='orange', ls='--', alpha=0.5, label='5Å (moderate)')
    ax.axhline(8.0, color='red', ls='--', alpha=0.5, label='8Å (significant)')

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

    # v4.0: RMSD 统计
    rmsd_stats = {}
    if results.get('rmsd_history'):
        final_rmsd = results['rmsd_history'][-1]['rmsd']
        rmsd_stats = {
            'final_rmsd_A': {str(k): float(v) for k, v in final_rmsd.items()},
        }
        # 计算平均 RMSD (平衡后)
        if len(results['rmsd_history']) > 4:
            late_rmsd = results['rmsd_history'][len(results['rmsd_history'])//2:]
            for rep_idx in final_rmsd.keys():
                vals = [r['rmsd'].get(rep_idx, 0) for r in late_rmsd]
                rmsd_stats[f'avg_rmsd_late_rep{rep_idx}_A'] = float(np.mean(vals))

    report = {
        'title': 'REST2 v4.3 scientific report',
        'version': results['version'],
        'n_replicas': int(n_rep),
        'temperature_interpretation': 'effective solute temperature; all replicas use a 300 K bath',
        'effective_solute_temperatures_K': [float(t) for t in T],
        'bath_temperature_K': float(cfg['T_ref']),
        'exchange_rates_percent': [float(round(rate, 2)) for rate in rates],
        'min_exchange_rate': float(round(np.min(rates), 2)),
        'avg_exchange_rate': float(round(avg_rate, 2)),
        'max_exchange_rate': float(round(np.max(rates), 2)),
        'n_iterations': int(len(E)),
        'total_simulation_ns': float(len(E) * cfg['n_steps_per_move'] * 2 / 1000000),
        'total_time_s': float(round(results['total_time'], 1)),
        'periodic_boundary': results['periodic_boundary'],
        'cubic_box': results['cubic_box'],
        'box_dimensions_angstrom': [float(d) for d in results.get('box_dims', [82.407, 82.407, 109.362])],
        'box_size_max_angstrom': float(results['box_size']),
        'lambda_hot': float(results['lambda_hot']),
        'local_hot_effective_temperature_max_K': (float(cfg['T_max'] / results['lambda_hot'])
                                                  if results['local_heating'] else None),
        'residue_offset': int(results.get('residue_offset', 9)),
        'round_trip': {
            'per_walker': [int(v) for v in results['walker_round_trips']],
            'directional_0_top_0__top_0_top': results['walker_round_trip_directions'].astype(int).tolist(),
            'total': int(results['round_trip_total']),
            'walkers_with_at_least_one': int(results['walkers_with_round_trip']),
            'note': ('未观察到完整往返；当前采样长度不足以据此判断采样错误'
                     if results['round_trip_total'] == 0 else '已观察到完整端点往返'),
        },
        'rmsd_stats': rmsd_stats,
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

    logger.info(f"\n[v4.3] Loading MDAnalysis...")
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

                logger.info(f"  Replica {rep_idx}: OF gate Rg = {np.mean(rg_values_equilibrated):.2f} ± {np.std(rg_values_equilibrated):.2f} Å "
                           f"(T={temps[rep_idx]:.0f}K, {len(rg_values_equilibrated)} frames)")

        except Exception as e:
            logger.warning(f"  Replica {rep_idx}: Error calculating Rg - {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())

    return rg_results


def generate_rg_visualizations(rg_results, temps, output_dir, logger):
    """生成OF gate Rg可视化图表"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as e:
        logger.warning(f"[v4.3] matplotlib not installed, skipping OF gate Rg visualizations: {e}")
        return

    figures_dir = os.path.join(output_dir, 'figures')

    if not rg_results:
        logger.warning("No OF gate Rg data to visualize")
        return

    logger.info(f"\n[v4.3] Generating OF gate Rg visualizations...")

    rep_indices = sorted(rg_results.keys())
    rg_means = [rg_results[i]['rg_mean'] for i in rep_indices]
    rg_stds = [rg_results[i]['rg_std'] for i in rep_indices]
    temp_values = [temps[i] for i in rep_indices]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.errorbar(temp_values, rg_means, yerr=rg_stds, fmt='o-', capsize=3,
                color='#2c3e50', linewidth=2, markersize=6)
    ax.set_xlabel('Temperature (K)', fontsize=12)
    ax.set_ylabel('OF Gate Radius of Gyration (Å)', fontsize=12)
    ax.set_title('(a) OF Gate Rg vs Temperature', fontweight='bold')
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
        'temperatures_K': [float(t) for t in temp_values],
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
        logger.warning("[v4.3] Non-standard local-hot extension ENABLED by command line")
    elif args.no_local_heating:
        cfg['local_heating_enabled'] = False
        logger.info("[v4.3] Local-hot extension disabled; using standard REST2")

    # 仅当用户显式指定时覆盖默认位置约束设置
    if args.equilibration_restraints is not None:
        cfg['equilibration_restraints'] = args.equilibration_restraints

    cfg['start_iteration'] = args.start_iteration

    try:
        if args.restore:
            restore_data = restore_from_checkpoint(args.restore, cfg, logger)
            saved_cfg = restore_data['checkpoint_cfg']
            for key in ('local_heating_enabled', 'lambda_hot', 'hot_residue_ids', 'T_ref',
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
        logger.info(f"[v4.3] Periodic boundary conditions: {cfg['periodic_boundary']}")
        logger.info(f"[v4.3] Preserve input box shape: {not cfg['use_cubic_box']}")
        logger.info(f"[v4.3] Local heating: {cfg['local_heating_enabled']}")
        if cfg['local_heating_enabled']:
            logger.info(f"[v4.3] Hot residues: {cfg['hot_residue_ids']}")
            logger.info(f"[v4.3] Lambda_hot: {cfg['lambda_hot']} "
                        f"(T_eff_max = {cfg['T_max']/cfg['lambda_hot']:.0f} K)")
        logger.info(f"[v4.3] Equilibration restraints: {cfg.get('equilibration_restraints', False)}")
        logger.info(f"[v4.3] Equilibration steps (NPT/restrained NVT/unrestrained NVT): "
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
