#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
REST2 (Replica Exchange with Solute Tempering v2) 分子动力学模拟
4AU5 蛋白 + CHARMM36m + TIP3P + 0.15M NaCl
+ 局部加热(残基 F72, A167, I168, F344, I345)
+ DCD轨迹保存(每100次迭代)
+ OF gate Rg计算(使用MDAnalysis)
+ 周期性边界条件(正交盒子)

======================================================================
版本: v3.2 (2026-07-28)

输入文件来源: charmm-gui/4au5-aa-k300n/openmm/
  - PSF: step5_charmm2omm.psf
  - PDB: step5_charmm2omm.pdb
  - 盒子信息: step5_charmm2omm.str
    BOXLX = 82.407 Å
    BOXLY = 82.407 Å
    BOXLZ = 109.362 Å

v3.2更新内容 (2026-07-28):
  1. 修复PDB文件保存时的周期性边界问题
     - getState()添加enforcePeriodicBox=True参数
     - 确保保存的PDB中原子坐标被包裹回盒子内
     - 添加CRYST1记录到PDB文件，VMD可直接读取盒子信息

  2. 修复final_pdb中水分子跑出盒子的问题
     - 保存前强制施加周期性边界约束
     - 蛋白质和溶剂原子都被正确包裹

  3. 保持v3.1的所有功能
     - DCD文件周期性盒子向量
     - matplotlib容错处理
     - 日志版本标识

v3.1更新内容 (2026-07-26):
  1. 修复DCD文件周期性盒子向量缺失问题
     - writeModel()添加periodicBoxVectors参数
     - getState()添加enforcePeriodicBox=True参数
     - 确保VMD能正确读取盒子信息

  2. 修复matplotlib缺失导致脚本崩溃问题
     - generate_scientific_visualizations()添加try-except
     - generate_rg_visualizations()添加try-except
     - 缺少matplotlib时优雅跳过，不影响模拟结果

  3. 改进日志输出
     - 修复日志中误导性的默认值
     - 更清晰的周期性边界条件状态显示

v3.0功能 (2026-07-23):
  1. 实现OpenMM周期性边界条件
     - 使用setDefaultPeriodicBoxVectors显式设置周期性盒子
     - 从step5_charmm2omm.str读取盒子信息
     - 使用原始盒子尺寸(82.407 x 82.407 x 109.362 Å)

  2. 盒子设置
     - 从step5_charmm2omm.str读取原始盒子尺寸
     - 保持原始盒子形状(正交盒子)
     - 确保PME方法正确应用周期性边界

  3. DCD轨迹文件
     - 每100次迭代保存一次轨迹
     - 自动验证DCD文件完整性

核心参数:
  1. 温度范围: 300K-350K
     - 适中温度范围,温度间隔约1.5K
     - 适用于接近生理温度的增强采样

  2. 副本数量: 32
     - 温度间隔约1.50-1.74K(几何分布)
     - 平均温度间隔约1.61K,接近目标值1.5K

  3. 每次迭代步数: 2000步 (4ps)
     - 总模拟时间: 1000次迭代 × 4ps = 4ns

参考文献:
  Wang et al., PNAS 108(37): 15107-15112 (2011)  [REST2原始论文]
  Liu et al., JCP 139: 034115 (2013)             [局部REST/局部加热]
======================================================================
"""

__version__ = "3.2"
__date__ = "2026-07-28"

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

    "n_replicas":       32,
    "T_ref":            300.0,
    "T_max":            350.0,
    "n_steps_per_move": 2000,
    "n_iterations":     1000,
    "checkpoint_interval": 100,

    "timestep":         2.0,
    "friction":         1.0,

    "platform_preference": ["CUDA", "OpenCL", "CPU"],
    "use_barostat":     False,

    "version":          __version__,

    "cmap_scaling_mode": "scaled_maps",

    "local_heating_enabled": True,
    "hot_residue_ids": ["72", "167", "168", "344", "345"],
    "hot_residue_names": {
        "72": "PHE",
        "167": "ALA",
        "168": "ILE",
        "344": "PHE",
        "345": "ILE"
    },
    "lambda_hot": 0.5,

    "save_trajectory": True,
    "trajectory_interval": 100,
    "trajectory_format": "dcd",

    "rg_equilibration_fraction": 0.2,

    # v3.2新增: 周期性边界和盒子参数
    "periodic_boundary": True,
    "use_cubic_box": False,  # 使用原始盒子形状,不强制转换为立方体
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

AA_RESNAMES = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
    'HSD', 'HSE', 'HSP', 'HSC', 'ALAD', 'ACE', 'NME', 'CT3', 'NTER', 'CTER',
    'CYX', 'CYS2',
}

# OF gate残基定义(参考com_rg_dist.py第71-72行)
OF_GATE_RESIDUES = ['72', '167', '168', '344', '345']

# OF gate MDAnalysis选择字符串(与com_rg_dist.py完全一致)
OF_GATE_SELECTION = "(resid 344 or resid 345 or resid 167 or resid 168 or resid 72) and not backbone and not type H* and segid PROA"


def parse_command_line():
    parser = argparse.ArgumentParser(description='REST2 Molecular Dynamics Simulation v3.2')

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

    parser.add_argument('--restore', type=str, default=None,
                        help='Checkpoint file to restore from')
    parser.add_argument('--start_iteration', type=int, default=0,
                        help='Starting iteration number (for resume)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (required when restoring)')

    parser.add_argument('--checkpoint_interval', type=int, default=CONFIG['checkpoint_interval'],
                        help=f'Checkpoint saving interval (default: {CONFIG["checkpoint_interval"]})')

    return parser.parse_args()


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
            self.experiment_id = f"4au5_rest2_v3.2_{ts}"
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
        if not output_dir:
            self.save_config()
        return self.experiment_dir

    def setup_logging(self):
        self.log_file = os.path.join(self.experiment_dir, "experiment.log")
        self.logger = logging.getLogger(f"REST2_v3.2_4au5_{self.experiment_id}")
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
            integ = openmm.LangevinIntegrator(300*unit.kelvin, 1/unit.picosecond, 2*unit.femtoseconds)
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

    返回:
      box_dims: [a, b, c] 盒子尺寸(Å)
      box_angles: [alpha, beta, gamma] 盒子角度(度)
    """
    try:
        # 读取step5_charmm2omm.str文件
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

        # CHARMM-GUI生成的盒子通常是正交盒子(角度均为90度)
        box_angles = [90.0, 90.0, 90.0]

        logger.info(f"[v3.2] Box from step5_charmm2omm.str: {box_dims[0]:.6f} x {box_dims[1]:.6f} x {box_dims[2]:.6f} Å")
        return box_dims, box_angles

    except Exception as e:
        logger.warning(f"[v3.2] Could not read boxinfo file: {e}")

    # 默认盒子大小
    logger.warning("[v3.2] Using default box size")
    return [82.407, 82.407, 109.362], [90.0, 90.0, 90.0]


def setup_periodic_box_vectors(system, box_dims, box_angles, use_cubic=False, logger=None):
    """
    设置周期性盒子向量

    参数:
      system: OpenMM System对象
      box_dims: [a, b, c] 盒子尺寸(Å)
      box_angles: [alpha, beta, gamma] 盒子角度(度)
      use_cubic: 是否使用立方体盒子(默认False,使用原始盒子形状)
      logger: 日志记录器

    返回:
      box_size: 实际使用的盒子边长(Å)
    """
    a, b, c = box_dims
    alpha, beta, gamma = box_angles

    if use_cubic:
        # 使用立方体盒子: 取最大维度
        max_dim = max(a, b, c)
        box_size = max_dim
        L = box_size / 10.0  # 转换为nm

        # 设置立方体盒子向量
        vectors = [
            openmm.Vec3(L, 0, 0),
            openmm.Vec3(0, L, 0),
            openmm.Vec3(0, 0, L)
        ] * unit.nanometer

        if logger:
            logger.info(f"[v3.2] Setting cubic periodic box: {box_size:.2f} x {box_size:.2f} x {box_size:.2f} Å")
            logger.info(f"[v3.2] Original box: {a:.2f} x {b:.2f} x {c:.2f} Å -> Cubic: {box_size:.2f} Å")
    else:
        # 使用原始盒子形状(正交盒子)
        # 对于正交盒子(alpha=beta=gamma=90°)，向量设置更简单
        if alpha == 90.0 and beta == 90.0 and gamma == 90.0:
            # 正交盒子
            a_nm, b_nm, c_nm = a/10.0, b/10.0, c/10.0

            vectors = [
                openmm.Vec3(a_nm, 0, 0),
                openmm.Vec3(0, b_nm, 0),
                openmm.Vec3(0, 0, c_nm)
            ] * unit.nanometer

            box_size = max(a, b, c)
            if logger:
                logger.info(f"[v3.2] Setting orthogonal periodic box: {a:.2f} x {b:.2f} x {c:.2f} Å")
                logger.info(f"[v3.2] Box angles: α={alpha:.1f}°, β={beta:.1f}°, γ={gamma:.1f}° (orthogonal)")
        else:
            # 三斜盒子(非正交)
            alpha_rad = np.radians(alpha)
            beta_rad = np.radians(beta)
            gamma_rad = np.radians(gamma)

            a_nm, b_nm, c_nm = a/10.0, b/10.0, c/10.0

            lx = a_nm
            ly = b_nm * np.sin(gamma_rad)
            lz = c_nm
            xy = b_nm * np.cos(gamma_rad)
            xz = c_nm * np.cos(beta_rad)
            yz = (b_nm * c_nm * np.cos(alpha_rad) - xy * xz) / ly

            vectors = [
                openmm.Vec3(lx, 0, 0),
                openmm.Vec3(xy, ly, 0),
                openmm.Vec3(xz, yz, lz)
            ] * unit.nanometer

            box_size = max(a, b, c)
            if logger:
                logger.info(f"[v3.2] Setting triclinic periodic box: {a:.2f} x {b:.2f} x {c:.2f} Å")
                logger.info(f"[v3.2] Box angles: α={alpha:.1f}°, β={beta:.1f}°, γ={gamma:.1f}°")

    system.setDefaultPeriodicBoxVectors(*vectors)

    return box_size


def build_base_system(cfg, logger):
    """
    构建CHARMM36m基础系统(v3.2版本: 周期性边界和正交盒子)
    """
    logger.info("=== Building CHARMM36m Force Field System (v3.2) ===")
    logger.info("[v3.2] Periodic Boundary Conditions: ENABLED")
    logger.info(f"[v3.2] Cubic Box: {cfg.get('use_cubic_box', False)}")

    psf = app.CharmmPsfFile(cfg["psf_file"])
    pdb = app.PDBFile(cfg["pdb_file"])

    # 读取CHARMM-GUI生成的盒子信息
    box_dims, box_angles = read_charmm_gui_box_info(cfg["boxinfo_str"], logger)

    # 保存原始盒子尺寸
    original_box = box_dims.copy()

    # 设置PSF盒子尺寸(必须在createSystem之前设置,否则PME会报错)
    psf.setBox(box_dims[0]*unit.angstrom, box_dims[1]*unit.angstrom, box_dims[2]*unit.angstrom)
    logger.info(f"[v3.2] PSF box set: {box_dims[0]:.6f} x {box_dims[1]:.6f} x {box_dims[2]:.6f} Å")

    # 加载力场参数文件
    # toppar.str是CHARMM命令格式,需要解析提取实际路径
    par_files = []
    charmmgui_dir = os.path.dirname(cfg["toppar_str"])
    toppar_dir = os.path.join(charmmgui_dir, "toppar")
    
    with open(cfg["toppar_str"], 'r') as f:
        for line in f:
            # 去除注释
            line = line.split('!')[0].strip()
            if not line:
                continue
            
            # 解析CHARMM命令格式
            # 格式: open read card unit 10 name toppar/xxx.rtf
            # 或: stream toppar/xxx.str
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
    for pf in par_files[:20]:
        logger.debug(f"  - {os.path.basename(pf)}")
    if len(par_files) > 20:
        logger.debug(f"  ... ({len(par_files) - 20} more)")
    
    params = app.CharmmParameterSet(*par_files)

    # 创建系统(使用PME方法,需要周期性边界)
    # 注意: CHARMM-GUI原始设置使用Force-switch vdW (r_on=1.0nm, r_off=1.2nm)
    # OpenMM此处使用标准硬截断,大多数OpenMM CHARMM36m模拟采用此方式
    system = psf.createSystem(
        params,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.2*unit.nanometer,
        constraints=app.HBonds,
        rigidWater=True,
        ewaldErrorTolerance=0.0005
    )

    # 设置周期性盒子向量(v3.2: 使用原始盒子形状)
    if cfg.get("periodic_boundary", True):
        box_size = setup_periodic_box_vectors(
            system, box_dims, box_angles,
            use_cubic=cfg.get("use_cubic_box", False),
            logger=logger
        )
    else:
        # 旧版本行为(仅设置盒子大小,不显式设置周期性向量)
        box_size = max(box_dims)
        logger.warning("[v3.2] Periodic boundary vectors NOT explicitly set (legacy mode)")

    n_atoms = system.getNumParticles()
    n_forces = system.getNumForces()
    logger.info(f"System: {n_atoms} atoms, {n_forces} forces")
    logger.info(f"[v3.2] Periodic box: {original_box[0]:.6f} x {original_box[1]:.6f} x {original_box[2]:.6f} Å ({original_box[0]/10:.3f} x {original_box[1]/10:.3f} x {original_box[2]/10:.3f} nm)")

    # 记录力场信息
    cmap_count = 0
    cmap_idx = None
    for i in range(n_forces):
        force = system.getForce(i)
        ftype = type(force).__name__
        if isinstance(force, openmm.CMAPTorsionForce):
            cmap_count = force.getNumTorsions()
            cmap_idx = i
            logger.info(f"  Force {i}: {ftype} ({cmap_count} torsions)")
        elif isinstance(force, openmm.NonbondedForce):
            logger.info(f"  Force {i}: {ftype} ({force.getNumParticles()} particles)")
        else:
            logger.info(f"  Force {i}: {ftype}")

    return system, psf, pdb, box_size, params, cmap_idx, original_box


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
        logger.info(f"[v3.2] Using segment-based solute identification (attribute: {segment_attr})")
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
        logger.info("[v3.2] Using residue-name-based solute identification (segment attribute not found)")
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
    logger.info(f"[v3.2] Solute: {n_solute} atoms ({len(solute_residues)} residues), "
               f"Solvent: {n_solvent} atoms, Ratio: {n_solute/n_total*100:.1f}%")

    if n_solvent + n_solute != n_total:
        logger.warning(f"[v3.2] Atom count mismatch: {n_solute + n_solvent} vs {n_total}")

    return solute_atoms


def identify_hot_residue_atoms(topology, cfg, logger):
    hot_atoms = {}

    residue_atoms = {}
    for atom in topology.atoms():
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

    logger.info(f"[v3.2] Local heating enabled for {len(hot_atoms)} residues")
    logger.info(f"[v3.2] Total hot atoms: {len(all_hot_atoms)}")
    logger.info(f"[v3.2] Lambda_hot: {cfg['lambda_hot']}")

    return all_hot_atoms, hot_atoms


def create_scaled_cmap_force(original_force, solute_atoms, lam, logger):
    n_maps = original_force.getNumMaps()
    n_torsions = original_force.getNumTorsions()

    logger.info(f"  [CMAP] Creating scaled CMAP force with λ={lam:.4f}")
    logger.info(f"  [CMAP] Original: {n_maps} maps, {n_torsions} torsions")

    scaled_maps = {}
    new_force = openmm.CMAPTorsionForce()

    for map_idx in range(n_maps):
        size, energy_grid = original_force.getMapParameters(map_idx)
        scaled_energy = energy_grid * lam
        scaled_map_idx = new_force.addMap(size, scaled_energy)
        scaled_maps[map_idx] = scaled_map_idx

    n_scaled = 0
    n_preserved = 0

    for torsion_idx in range(n_torsions):
        params = original_force.getTorsionParameters(torsion_idx)
        orig_map_idx = params[0]
        atoms = params[1:9]

        all_solute = all(a in solute_atoms for a in atoms)

        new_map_idx = scaled_maps[orig_map_idx]
        new_force.addTorsion(new_map_idx, *atoms)
        if all_solute:
            n_scaled += 1
        else:
            n_preserved += 1

    logger.info(f"  [CMAP] Result: {n_scaled} torsions scaled, {n_preserved} preserved")

    return new_force, n_scaled


def create_rest2_system_with_local_heating(base_system, solute_atoms, hot_atoms, lam_global,
                                            lam_hot, params, cmap_idx, cfg, logger):
    system = copy.deepcopy(base_system)
    sqrt_lam_global = np.sqrt(lam_global)
    sqrt_lam_hot = np.sqrt(lam_global * lam_hot)
    lam_hot_effective = lam_global * lam_hot

    n_forces = system.getNumForces()

    logger.info(f"  Creating REST2 system with local heating:")
    logger.info(f"    λ_global={lam_global:.4f}, λ_hot={lam_hot:.4f}")
    logger.info(f"    λ_hot_effective={lam_hot_effective:.4f}")

    cmap_forces_to_replace = []

    for i in range(n_forces):
        force = system.getForce(i)

        if isinstance(force, openmm.NonbondedForce):
            n_normal = 0
            n_hot = 0
            for j in range(force.getNumParticles()):
                q, sig, eps = force.getParticleParameters(j)
                if j in hot_atoms:
                    force.setParticleParameters(j, q * sqrt_lam_hot, sig, eps * lam_hot_effective)
                    n_hot += 1
                elif j in solute_atoms:
                    force.setParticleParameters(j, q * sqrt_lam_global, sig, eps * lam_global)
                    n_normal += 1

            for j in range(force.getNumExceptions()):
                p1, p2, q12, sig, eps = force.getExceptionParameters(j)
                s1 = p1 in solute_atoms
                s2 = p2 in solute_atoms
                h1 = p1 in hot_atoms
                h2 = p2 in hot_atoms

                if s1 and s2:
                    if h1 or h2:
                        force.setExceptionParameters(j, p1, p2, q12 * lam_hot_effective, sig, eps * lam_hot_effective)
                    else:
                        force.setExceptionParameters(j, p1, p2, q12 * lam_global, sig, eps * lam_global)
                elif s1 or s2:
                    if h1 or h2:
                        force.setExceptionParameters(j, p1, p2, q12 * sqrt_lam_hot, sig, eps * sqrt_lam_hot)
                    else:
                        force.setExceptionParameters(j, p1, p2, q12 * sqrt_lam_global, sig, eps * sqrt_lam_global)

            logger.info(f"    NonbondedForce: {n_normal} normal + {n_hot} hot particles")

        elif isinstance(force, openmm.HarmonicBondForce):
            for j in range(force.getNumBonds()):
                p1, p2, r, k = force.getBondParameters(j)
                if p1 in solute_atoms and p2 in solute_atoms:
                    if p1 in hot_atoms or p2 in hot_atoms:
                        force.setBondParameters(j, p1, p2, r, k * lam_hot_effective)
                    else:
                        force.setBondParameters(j, p1, p2, r, k * lam_global)

        elif isinstance(force, openmm.HarmonicAngleForce):
            for j in range(force.getNumAngles()):
                p1, p2, p3, theta, k = force.getAngleParameters(j)
                if p1 in solute_atoms and p2 in solute_atoms and p3 in solute_atoms:
                    if p1 in hot_atoms or p2 in hot_atoms or p3 in hot_atoms:
                        force.setAngleParameters(j, p1, p2, p3, theta, k * lam_hot_effective)
                    else:
                        force.setAngleParameters(j, p1, p2, p3, theta, k * lam_global)

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
                scaled_cmap, n_cmap_scaled = create_scaled_cmap_force(force, solute_atoms, lam_global, logger)
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

    # 复制周期性盒子向量
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
        logger.info(f"REST2 Temperature Distribution ({n_replicas} replicas, {T_ref}-{T_max}K)")
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
        logger.info(f"[v3.2] Temperature spacing: min={min(dTs):.2f}K, max={max(dTs):.2f}K, avg={np.mean(dTs):.2f}K")

    return temps, lams


def save_full_checkpoint(cfg, iteration, contexts, energy_history, exchange_attempts,
                         exchange_accepts, temps, lams, delta_history, logger):
    ckpt_file = os.path.join(cfg["output_dir"], f"checkpoint_{iteration}_full.npz")

    positions_list = []
    velocities_list = []

    for ctx in contexts:
        state = ctx.getState(getPositions=True, getVelocities=True)
        pos = np.array([[p.x, p.y, p.z] for p in state.getPositions()])
        vel = np.array([[v.x, v.y, v.z] for v in state.getVelocities()])
        positions_list.append(pos)
        velocities_list.append(vel)

    save_dict = {
        'iteration': iteration,
        'energies': np.array(energy_history),
        'attempts': exchange_attempts,
        'accepts': exchange_accepts,
        'temps': temps,
        'lams': lams,
        'lambda_hot': cfg["lambda_hot"],
        'hot_residue_ids': cfg["hot_residue_ids"],
    }

    for i, (pos, vel) in enumerate(zip(positions_list, velocities_list)):
        save_dict[f'positions_{i}'] = pos
        save_dict[f'velocities_{i}'] = vel

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

    start_iteration = int(data['iteration'])
    energy_history = list(data['energies'])
    exchange_attempts = data['attempts']
    exchange_accepts = data['accepts']
    temps = data['temps']
    lams = data['lams']

    delta_history = {}
    for k in range(len(temps) - 1):
        key = f'delta_{k}'
        if key in data:
            delta_history[k] = list(data[key])

    n_replicas = len(temps)
    positions_list = []
    velocities_list = []
    for i in range(n_replicas):
        positions_list.append(data[f'positions_{i}'])
        velocities_list.append(data[f'velocities_{i}'])

    logger.info(f"  Restored iteration: {start_iteration}")
    logger.info(f"  Energy history length: {len(energy_history)}")
    logger.info(f"  Total replicas: {n_replicas}")

    return {
        'start_iteration': start_iteration,
        'energy_history': energy_history,
        'exchange_attempts': exchange_attempts,
        'exchange_accepts': exchange_accepts,
        'temps': temps,
        'lams': lams,
        'delta_history': delta_history,
        'positions_list': positions_list,
        'velocities_list': velocities_list,
    }


def verify_dcd_file(dcd_file, logger):
    """
    验证DCD文件是否正确生成并包含周期性盒子信息

    参数:
      dcd_file: DCD文件路径
      logger: 日志记录器
    """
    try:
        import MDAnalysis as mda

        # 尝试加载DCD文件
        u = mda.Universe(dcd_file)

        # 检查轨迹信息
        n_frames = len(u.trajectory)
        logger.info(f"[v3.2] DCD verification: {dcd_file}")
        logger.info(f"[v3.2]   Frames: {n_frames}")

        # 检查盒子信息
        if hasattr(u.trajectory, 'dimensions'):
            dims = u.trajectory.dimensions
            if dims is not None:
                logger.info(f"[v3.2]   Box dimensions: {dims[:3]} Å")
                logger.info(f"[v3.2]   Box angles: {dims[3:]} degrees")
                logger.info(f"[v3.2]   Periodic boundary: VERIFIED")
            else:
                logger.warning(f"[v3.2]   Box dimensions: NOT FOUND in DCD file")

        return True

    except ImportError:
        logger.warning("[v3.2] MDAnalysis not available for DCD verification")
        return False
    except Exception as e:
        logger.error(f"[v3.2] DCD verification failed: {e}")
        return False


def run_rest2(cfg, logger, restore_data=None):
    kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
    kB_val = kB.value_in_unit(unit.kilojoule_per_mole / unit.kelvin)

    logger.info(f"\n{'='*70}")
    logger.info(f"REST2 Local Heating Simulation [v{cfg['version']}]")
    logger.info(f"Temperature Range: {cfg['T_ref']}K - {cfg['T_max']}K")
    logger.info(f"Hot Residues: {cfg['hot_residue_ids']}")
    logger.info(f"Lambda_hot: {cfg['lambda_hot']}")
    logger.info(f"[v3.2] Periodic Boundary: {cfg.get('periodic_boundary', True)}")
    logger.info(f"[v3.2] Original Box Shape: {cfg.get('use_cubic_box', False) == False} (using input file dimensions)")
    if cfg.get("save_trajectory", False):
        logger.info(f"Trajectory Saving: Every {cfg['trajectory_interval']} iterations")
    logger.info(f"{'='*70}")

    base_system, psf, pdb, box_size, params, cmap_idx, box_dims = build_base_system(cfg, logger)
    solute_atoms = identify_solute_atoms(pdb.topology, logger)

    if cfg["local_heating_enabled"]:
        hot_atoms, hot_atoms_by_residue = identify_hot_residue_atoms(pdb.topology, cfg, logger)
    else:
        hot_atoms = set()
        hot_atoms_by_residue = {}

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
    else:
        temps, lams = optimize_temperatures_geometric(n_rep, T_ref, T_max, logger)
        start_iteration = 0
        energy_history = []
        exchange_attempts = np.zeros(n_rep-1, dtype=int)
        exchange_accepts = np.zeros(n_rep-1, dtype=int)
        delta_history = {k: [] for k in range(n_rep-1)}

    logger.info(f"\n=== Creating REST2 Replica Systems with Local Heating ===")
    replica_systems = []

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

        replica_systems.append(rep_sys)

    platform, plat_name = select_platform(cfg["platform_preference"], logger)

    logger.info(f"\n=== Initializing Replica Contexts ===")
    contexts = []

    for i, (rep_sys, T) in enumerate(zip(replica_systems, temps)):
        integ = openmm.LangevinIntegrator(
            T*unit.kelvin, cfg["friction"]/unit.picosecond, cfg["timestep"]*unit.femtoseconds
        )
        ctx = openmm.Context(rep_sys, integ, platform)

        if restore_data:
            pos = restore_data['positions_list'][i]
            vel = restore_data['velocities_list'][i]
            pos_openmm = [openmm.Vec3(p[0], p[1], p[2]) for p in pos] * unit.angstrom
            vel_openmm = [openmm.Vec3(v[0], v[1], v[2]) for v in vel] * unit.angstrom/unit.picosecond
            ctx.setPositions(pos_openmm)
            ctx.setVelocities(vel_openmm)
            # 恢复周期性盒子向量
            ctx.setPeriodicBoxVectors(*base_system.getDefaultPeriodicBoxVectors())
        else:
            ctx.setPositions(pdb.positions)
            # 设置周期性盒子向量
            ctx.setPeriodicBoxVectors(*base_system.getDefaultPeriodicBoxVectors())
            openmm.LocalEnergyMinimizer.minimize(ctx, 10*unit.kilojoule_per_mole/unit.nanometer)
            ctx.setVelocitiesToTemperature(T*unit.kelvin)

        contexts.append(ctx)

    if not restore_data:
        logger.info(f"\n=== Staged Equilibration ===")
        equil_protocol = [
            (50,   50,  0.5, "Low-T relaxation"),
            (100, 100,  1.0, "Medium-T thermalization"),
            (200, 200,  1.0, "High-T sampling preparation"),
            (300, 500,  2.0, "Target-T equilibration"),
        ]

        for i, (ctx, T) in enumerate(zip(contexts, temps)):
            for stage_T, stage_steps, stage_dt, stage_name in equil_protocol:
                integ = ctx.getIntegrator()
                integ.setTemperature(stage_T*unit.kelvin)
                integ.setStepSize(stage_dt*unit.femtoseconds)
                integ.step(stage_steps)

            integ = ctx.getIntegrator()
            integ.setTemperature(T*unit.kelvin)
            integ.setStepSize(cfg["timestep"]*unit.femtoseconds)
            integ.step(500)

            if i < 5:
                state = ctx.getState(getEnergy=True)
                E = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                logger.info(f"  Replica {i} (T={T:.1f}K): Equilibrated E={E:.1f} kJ/mol")
    else:
        logger.info(f"\n=== Skipping Equilibration (Restored from checkpoint) ===")

    # 初始化DCD轨迹写入器
    dcd_writers = []
    dcd_files = []
    if cfg.get("save_trajectory", False):
        logger.info(f"\n=== Initializing Trajectory Writers ===")
        trajectories_dir = cfg.get("trajectories_dir", cfg["output_dir"])
        for i in range(n_rep):
            dcd_file_path = os.path.join(trajectories_dir, f"replica_{i}.dcd")
            dt = cfg["timestep"] * cfg["trajectory_interval"] * cfg["n_steps_per_move"] / 1000.0
            dcd_file_obj = open(dcd_file_path, 'wb')
            dcd_writer = app.DCDFile(dcd_file_obj, pdb.topology, dt)
            dcd_writers.append(dcd_writer)
            dcd_files.append(dcd_file_obj)
            if i < 5:
                logger.info(f"  Replica {i}: {dcd_file_path}")
        logger.info(f"  Trajectory interval: Every {cfg['trajectory_interval']} iterations")
        logger.info(f"  Trajectory mode: Write (new file)")

    n_steps = cfg["n_steps_per_move"]
    n_iter = cfg["n_iterations"]
    dt = cfg["timestep"]
    total_ps = n_iter * n_steps * dt / 1000
    total_ns = total_ps / 1000

    logger.info(f"\n{'='*70}")
    logger.info(f"REST2 Production Simulation with Local Heating")
    logger.info(f"  Replicas: {n_rep}, T: {T_ref:.0f}-{T_max:.0f}K")
    logger.info(f"  Steps/iter: {n_steps}")
    logger.info(f"  Iteration range: {start_iteration} - {n_iter}")
    logger.info(f"  Remaining iterations: {n_iter - start_iteration}")
    logger.info(f"  Platform: {plat_name}")
    logger.info(f"[v3.2] Periodic boundary conditions: ACTIVE")
    logger.info(f"[v3.2] Using original box shape from input file")
    logger.info(f"{'='*70}\n")

    t0 = time.time()

    try:
        for iteration in range(start_iteration, n_iter):
            for ctx in contexts:
                ctx.getIntegrator().step(n_steps)

            pairs = range(0, n_rep-1, 2) if iteration % 2 == 0 else range(1, n_rep-1, 2)

            positions = []
            energies = []
            for ctx in contexts:
                state = ctx.getState(getEnergy=True, getPositions=True)
                positions.append(state.getPositions())
                energies.append(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))

            for m in pairs:
                n = m + 1
                exchange_attempts[m] += 1

                pos_m, pos_n = positions[m], positions[n]
                E_m_xm, E_n_xn = energies[m], energies[n]

                contexts[m].setPositions(pos_n)
                E_m_xn = contexts[m].getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

                contexts[n].setPositions(pos_m)
                E_n_xm = contexts[n].getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

                beta_m, beta_n = 1/(kB_val*temps[m]), 1/(kB_val*temps[n])
                delta = beta_m*(E_m_xn-E_m_xm) + beta_n*(E_n_xm-E_n_xn)
                delta_history[m].append(delta)

                accept = delta < 0 or np.random.random() < np.exp(-delta)

                if accept:
                    exchange_accepts[m] += 1
                    contexts[m].setPositions(pos_n)
                    contexts[n].setPositions(pos_m)
                    contexts[m].setVelocitiesToTemperature(temps[m]*unit.kelvin)
                    contexts[n].setVelocitiesToTemperature(temps[n]*unit.kelvin)
                    positions[m], positions[n] = pos_n, pos_m
                    energies[m], energies[n] = E_m_xn, E_n_xm
                else:
                    contexts[m].setPositions(pos_m)
                    contexts[n].setPositions(pos_n)

            energy_history.append(energies)

            if cfg.get("save_trajectory", False) and dcd_writers:
                if (iteration + 1) % cfg["trajectory_interval"] == 0:
                    for i, ctx in enumerate(contexts):
                        state = ctx.getState(getPositions=True, enforcePeriodicBox=True)
                        pos = state.getPositions()
                        box_vecs = state.getPeriodicBoxVectors()
                        dcd_writers[i].writeModel(pos, periodicBoxVectors=box_vecs)

            if (iteration+1) % 20 == 0:
                rates = [f"{exchange_accepts[k]/exchange_attempts[k]*100:.0f}%"
                        if exchange_attempts[k] > 0 else "-" for k in range(n_rep-1)]
                logger.info(f"  Iter {iteration+1:4d}/{n_iter}: "
                           f"E={[int(e) for e in energies[:4]]}..., "
                           f"Rates=[{','.join(rates[:4])}...]")

            if (iteration+1) % cfg["checkpoint_interval"] == 0:
                save_full_checkpoint(cfg, iteration+1, contexts, energy_history, exchange_attempts,
                                    exchange_accepts, temps, lams, delta_history, logger)
                save_checkpoint(cfg, iteration+1, energy_history, exchange_attempts,
                               exchange_accepts, temps, lams, delta_history)
    finally:
        if dcd_writers:
            logger.info(f"\n=== Closing Trajectory Files ===")
            for i, dcd_f in enumerate(dcd_files):
                dcd_f.close()
                if i < 5:
                    logger.info(f"  Replica {i}: DCD file closed")

    total_time = time.time() - t0

    avg_rate = np.mean([exchange_accepts[k]/exchange_attempts[k]*100
                       for k in range(n_rep-1) if exchange_attempts[k] > 0])

    logger.info(f"\n{'='*70}")
    logger.info(f"REST2 Local Heating Simulation Complete! [v{cfg['version']}]")
    logger.info(f"  Total time: {total_time:.1f}s")
    logger.info(f"  Avg exchange rate: {avg_rate:.1f}%")
    logger.info(f"{'='*70}")

    results = {
        'energies': np.array(energy_history),
        'exchange_attempts': exchange_attempts,
        'exchange_accepts': exchange_accepts,
        'temperatures': np.array(temps),
        'lam_values': np.array(lams),
        'total_time': total_time,
        'delta_history': delta_history,
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
    }

    save_final_results(cfg, results, contexts, pdb)
    try:
        generate_scientific_visualizations(results, cfg["output_dir"], logger)
    except ImportError as e:
        logger.warning(f"[v3.2] matplotlib not installed, skipping visualizations: {e}")
    save_scientific_report(cfg, results, logger)

    # 使用MDAnalysis计算OF gate Rg(与com_rg_dist.py一致)
    logger.info(f"\n{'='*70}")
    logger.info(f"[v3.2] Calculating OF Gate Rg using MDAnalysis")
    logger.info(f"{'='*70}")
    logger.info(f"  OF gate selection: {OF_GATE_SELECTION}")
    logger.info(f"  Equilibration fraction: {cfg['rg_equilibration_fraction']*100:.0f}%")

    rg_results = calculate_rg_with_mdanalysis(cfg, logger, temps)

    if rg_results:
        generate_rg_visualizations(rg_results, temps, cfg["output_dir"], logger)

    # 验证DCD文件
    if cfg.get("save_trajectory", False):
        logger.info(f"\n[v3.2] Verifying DCD trajectory files...")
        trajectories_dir = cfg.get("trajectories_dir", cfg["output_dir"])
        for i in range(min(3, n_rep)):  # 只验证前3个副本
            dcd_file = os.path.join(trajectories_dir, f"replica_{i}.dcd")
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
        **{f"delta_{k}": np.array(v) for k, v in results['delta_history'].items() if v})

    # 获取盒子尺寸用于CRYST1记录
    box_dims = results.get('box_dims', [82.407, 82.407, 109.362])
    box_angles = [90.0, 90.0, 90.0]

    for i, ctx in enumerate(contexts):
        # v3.2修复: 添加enforcePeriodicBox=True，确保原子坐标被包裹回盒子内
        state = ctx.getState(getPositions=True, enforcePeriodicBox=True)
        pos = state.getPositions()
        
        pdb_file = os.path.join(cfg["output_dir"], f"replica_{i}_final.pdb")
        with open(pdb_file, 'w') as f:
            # v3.2修复: 手动写入CRYST1记录，VMD可直接读取盒子信息
            cryst_line = f"CRYST1{box_dims[0]:9.3f}{box_dims[1]:9.3f}{box_dims[2]:9.3f}{box_angles[0]:7.2f}{box_angles[1]:7.2f}{box_angles[2]:7.2f} P 1           1\n"
            f.write(cryst_line)
            f.write("REMARK   REST2 v3.2 - Final Structure\n")
            app.PDBFile.writeFile(pdb.topology, pos, f)

    mbar_file = os.path.join(cfg["mbar_dir"], "reduced_energies.npz")
    np.savez_compressed(mbar_file,
        u_kn=results['energies'],
        N_k=np.ones(len(results['temperatures'])),
        temps=results['temperatures'],
        lambda_hot=results['lambda_hot'])


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
        if deltas[k]:
            cum = np.cumsum([d<0 for d in deltas[k]])/np.arange(1,len(deltas[k])+1)*100
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


def save_scientific_report(cfg, results, logger):
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
        'total_simulation_ns': float(len(E) * cfg['n_steps_per_move'] * 2 / 1000000),
        'total_time_s': float(round(results['total_time'], 1)),
        'periodic_boundary': results['periodic_boundary'],
        'cubic_box': results['cubic_box'],
        'box_dimensions_angstrom': [float(d) for d in results.get('box_dims', [82.407, 82.407, 109.362])],
        'box_size_max_angstrom': float(results['box_size']),
    }

    report_file = os.path.join(cfg["output_dir"], "scientific_report.json")
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info(f"  Scientific report saved to {report_file}")


# ======================================================================
# MDAnalysis Rg计算(与com_rg_dist.py完全一致)
# ======================================================================

def radgyr(atomgroup, masses, total_mass=None):
    """
    回转半径计算函数(参考com_rg_dist.py第15-31行)

    参数:
      atomgroup: MDAnalysis AtomGroup对象
      masses: 原子质量数组
      total_mass: 总质量(可选)

    返回:
      Rg值数组
    """
    # 坐标随帧变化
    coordinates = atomgroup.positions
    center_of_mass = atomgroup.center_of_mass()

    # 计算到质心的平方距离
    ri_sq = (coordinates - center_of_mass) ** 2
    # 求和未加权的位置
    sq = np.sum(ri_sq, axis=1)

    # 转换为数组
    sq_rs = np.array([sq])

    # 加权位置
    if total_mass is None:
        total_mass = np.sum(masses)
    rog_sq = np.sum(masses * sq_rs, axis=1) / total_mass

    # 开平方并返回
    return np.sqrt(rog_sq)


def calculate_rg_with_mdanalysis(cfg, logger, temps):
    """
    使用MDAnalysis计算OF gate回转半径(与com_rg_dist.py一致)

    步骤:
      1. 使用MDAnalysis创建Universe(PSF + DCD)
      2. 使用select_atoms选择OF gate残基
      3. 使用analysis_class框架计算Rg
      4. 跳过平衡阶段(前20%帧)
      5. 返回每个温度下的Rg统计

    参数:
      cfg: 配置字典
      logger: 日志记录器
      temps: 温度数组

    返回:
      rg_results: 包含每个副本Rg统计的字典
    """
    try:
        import MDAnalysis as mda
        from MDAnalysis.analysis.base import analysis_class
    except ImportError:
        logger.error("MDAnalysis not installed! Cannot calculate OF gate Rg.")
        logger.error("Install: pip install MDAnalysis")
        return None

    logger.info(f"\n[v3.2] Loading MDAnalysis...")
    logger.info(f"  MDAnalysis version: {mda.__version__}")

    trajectories_dir = cfg.get("trajectories_dir", cfg["output_dir"])
    n_replicas = cfg["n_replicas"]
    psf_file = cfg["psf_file"]

    equil_fraction = cfg.get("rg_equilibration_fraction", 0.2)

    rg_results = {}

    for rep_idx in range(n_replicas):
        dcd_file = os.path.join(trajectories_dir, f"replica_{rep_idx}.dcd")

        if not os.path.exists(dcd_file):
            logger.warning(f"  Replica {rep_idx}: DCD file not found - {dcd_file}")
            continue

        try:
            # 创建MDAnalysis Universe(参考com_rg_dist.py第58行)
            logger.info(f"  Replica {rep_idx}: Loading trajectory...")
            u = mda.Universe(psf_file, dcd_file)

            # 选择OF gate原子(与com_rg_dist.py第71-72行完全一致)
            atom_of = u.select_atoms(OF_GATE_SELECTION)

            if atom_of.n_atoms == 0:
                logger.warning(f"  Replica {rep_idx}: No OF gate atoms selected!")
                continue

            logger.info(f"  Replica {rep_idx}: OF gate atoms selected: {atom_of.n_atoms}")

            # 创建RadiusOfGyration分析类(参考com_rg_dist.py第35行)
            RadiusOfGyration = analysis_class(radgyr)

            # 运行分析(参考com_rg_dist.py第73-76行)
            total_mass = np.sum(atom_of.masses)
            rog_u_of = RadiusOfGyration(u.trajectory, atom_of, atom_of.masses, total_mass=total_mass)
            rog_u_of.run()

            # 获取结果(参考com_rg_dist.py第78-79行)
            rg_values = rog_u_of.results['timeseries'][:, 0]

            # 跳过平衡阶段(前20%帧)
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
                           f"(T={temps[rep_idx]:.0f}K, {len(rg_values_equilibrated)} frames, skipped {equil_cutoff} equilibration frames)")

        except Exception as e:
            logger.warning(f"  Replica {rep_idx}: Error calculating Rg - {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())

    return rg_results


def generate_rg_visualizations(rg_results, temps, output_dir, logger):
    """
    生成OF gate Rg可视化图表
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as e:
        logger.warning(f"[v3.2] matplotlib not installed, skipping OF gate Rg visualizations: {e}")
        return

    figures_dir = os.path.join(output_dir, 'figures')

    if not rg_results:
        logger.warning("No OF gate Rg data to visualize")
        return

    logger.info(f"\n[v3.2] Generating OF gate Rg visualizations...")

    rep_indices = sorted(rg_results.keys())
    rg_means = [rg_results[i]['rg_mean'] for i in rep_indices]
    rg_stds = [rg_results[i]['rg_std'] for i in rep_indices]
    temp_values = [temps[i] for i in rep_indices]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图: OF gate Rg vs Temperature
    ax = axes[0]
    ax.errorbar(temp_values, rg_means, yerr=rg_stds, fmt='o-', capsize=3,
                color='#2c3e50', linewidth=2, markersize=6)
    ax.set_xlabel('Temperature (K)', fontsize=12)
    ax.set_ylabel('OF Gate Radius of Gyration (Å)', fontsize=12)
    ax.set_title('(a) OF Gate Rg vs Temperature', fontweight='bold')
    ax.grid(alpha=0.3)

    # 添加趋势线
    try:
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(temp_values, rg_means)
        ax.plot(temp_values, [slope*t + intercept for t in temp_values],
                '--', color='#e74c3c', label=f'Trend: Rg = {slope:.4f}T + {intercept:.2f}')
        ax.legend(fontsize=10)
    except ImportError:
        logger.warning("scipy not available, skipping trend line")

    # 右图: Rg分布
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

    # 保存Rg数据
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
        restore_data = restore_from_checkpoint(args.restore, CONFIG,
                                                logging.getLogger("restore"))
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

    if args.start_iteration > 0:
        cfg['start_iteration'] = args.start_iteration

    try:
        logger.info(f"\n{'='*70}")
        logger.info(f"REST2 Local Heating Simulation - 4AU5 Protein + CHARMM36m")
        logger.info(f"Version: {cfg['version']}")
        logger.info(f"Temperature: {cfg['T_ref']}K - {cfg['T_max']}K")
        logger.info(f"Steps per iteration: {cfg['n_steps_per_move']}")
        logger.info(f"Total iterations: {cfg['n_iterations']}")
        logger.info(f"Trajectory interval: {cfg['trajectory_interval']} iterations")
        logger.info(f"[v3.2] Periodic boundary conditions: {cfg['periodic_boundary']}")
        logger.info(f"[v3.2] Cubic box: {cfg['use_cubic_box']}")
        if args.restore:
            logger.info(f"Restored from: {args.restore}")
            logger.info(f"Starting from iteration: {args.start_iteration}")
        logger.info(f"Output: {exp_dir}")
        logger.info(f"{'='*70}")

        results = run_rest2(cfg, logger, restore_data)

        logger.info(f"\n[SUCCESS] Experiment completed!")
        logger.info(f"Results saved to: {exp_dir}")

    except Exception as e:
        logger.error(f"\n[FATAL ERROR] {str(e)}")
        logger.error(traceback.format_exc())
        raise