#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
REST2 (Replica Exchange with Solute Tempering v2) 分子动力学模拟
4AU5 蛋白 + CHARMM36m + TIP3P + 0.15M NaCl
+ 局部加热（残基 F72, A167, I168, F344, I345）
+ DCD轨迹保存
+ 分段运行与恢复功能

======================================================================
版本: v2.1 (2026-07-22)

新增功能:
  1. 命令行参数支持
     - --n_iterations: 迭代次数
     - --n_steps_per_move: 每次迭代步数
     - --restore: 从checkpoint恢复
     - --start_iteration: 起始迭代次数
     - --output_dir: 指定输出目录（恢复时需要）

  2. 完整checkpoint保存
     - 保存所有副本的位置和速度
     - 保存能量历史和交换统计
     - 支持断点续传

  3. 分段运行支持
     - 可以分段完成长模拟
     - 每段完成后自动保存checkpoint
     - 下次运行可从checkpoint恢复

核心参数:
  1. 温度范围: 300K-350K
     - 适中温度范围，温度间隔约1.5K
     - 适用于接近生理温度的增强采样

  2. 副本数量: 32
     - 温度间隔约1.50-1.74K（几何分布）
     - 平均温度间隔约1.61K，接近目标值1.5K

  3. 每次迭代步数: 500000步 (1ns)
     - 大幅增加采样时间
     - 总模拟时间: 1000次迭代 × 1ns = 1μs

使用示例:
  # 首次运行（迭代0-99）
  python rest2_4au5_v2.1.py --n_iterations 100

  # 从checkpoint恢复继续运行（迭代100-199）
  python rest2_4au5_v2.1.py --restore outputs/experiment_XXX/checkpoint_100_full.npz --start_iteration 100 --n_iterations 200 --output_dir outputs/experiment_XXX

参考文献:
  Wang et al., PNAS 108(37): 15107-15112 (2011)  [REST2原始论文]
  Liu et al., JCP 139: 034115 (2013)             [局部REST/局部加热]
======================================================================
"""

__version__ = "2.1"
__date__ = "2026-07-22"

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
CHARMMGUI_DIR = os.path.join(SCRIPT_DIR, "charmm-gui", "charmm-gui-8425204688")
OPENMM_DIR = os.path.join(CHARMMGUI_DIR, "openmm")

CONFIG = {
    "psf_file":     os.path.join(OPENMM_DIR, "step3_input.psf"),
    "pdb_file":     os.path.join(OPENMM_DIR, "step3_input.pdb"),
    "toppar_str":   os.path.join(OPENMM_DIR, "toppar.str"),
    "sysinfo_dat":  os.path.join(OPENMM_DIR, "sysinfo.dat"),
    "base_output_dir": os.path.join(SCRIPT_DIR, "outputs"),

    # 温度范围和副本数
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

    # CMAP缩放模式
    "cmap_scaling_mode": "scaled_maps",

    # 局部加热参数 - 5个残基位点
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

    # 轨迹保存参数
    "save_trajectory": True,
    "trajectory_interval": 1,
    "trajectory_format": "dcd",
}

SOLVENT_RESNAMES = {
    'HOH', 'WAT', 'TIP', 'SOL', 'TIP3', 'TIP4', 'TIP5',
    'CL', 'NA', 'K', 'CA', 'MG', 'ZN', 'SOD', 'CLA', 'CL-', 'NA+',
    'K+', 'CA2', 'MG2', 'ZN2', 'BR', 'BR-', 'I', 'I-', 'F', 'F-',
}

AA_RESNAMES = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
    'HSD', 'HSE', 'HSP', 'HSC', 'ALAD', 'ACE', 'NME', 'CT3', 'NTER', 'CTER',
    'CYX', 'CYS2',
}


def parse_command_line():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='REST2 Molecular Dynamics Simulation v2.1')

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
    """实验管理器 - 符合科研数据管理标准"""

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
            # 使用现有目录（恢复模式）
            self.experiment_dir = output_dir
            self.experiment_id = os.path.basename(output_dir)
        else:
            # 创建新目录
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_id = f"4au5_rest2_v2.1_{ts}"
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
        self.logger = logging.getLogger(f"REST2_v2.1_4au5_{self.experiment_id}")
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


def build_base_system(cfg, logger):
    """构建CHARMM36m基础系统"""
    logger.info("=== Building CHARMM36m Force Field System ===")

    psf = app.CharmmPsfFile(cfg["psf_file"])
    pdb = app.PDBFile(cfg["pdb_file"])

    try:
        with open(cfg["sysinfo_dat"], 'rb') as f:
            data = f.read()
            try:
                sysinfo = json.loads(data.decode('utf-8'))
                box = sysinfo["dimensions"][:3]
            except:
                logger.warning("Could not parse sysinfo.dat, using default box size")
                box = [78.812, 78.812, 106.959]
    except Exception as e:
        logger.warning(f"Error reading sysinfo.dat: {e}, using PDB box size")
        box = [78.812, 78.812, 106.959]

    psf.setBox(box[0]*unit.angstrom, box[1]*unit.angstrom, box[2]*unit.angstrom)

    par_files = []
    openmm_dir = os.path.dirname(cfg["toppar_str"])
    with open(cfg["toppar_str"], 'r') as f:
        for line in f:
            par = line.split('!')[0].strip()
            if par:
                full = os.path.normpath(os.path.join(openmm_dir, par))
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

    n_atoms = system.getNumParticles()
    n_forces = system.getNumForces()
    logger.info(f"System: {n_atoms} atoms, {n_forces} forces, Box: {box[0]:.1f}x{box[1]:.1f}x{box[2]:.1f} A")

    cmap_count = 0
    cmap_idx = None
    for i in range(n_forces):
        force = system.getForce(i)
        ftype = type(force).__name__
        if isinstance(force, openmm.CMAPTorsionForce):
            cmap_count = force.getNumTorsions()
            cmap_idx = i
            logger.info(f"  Force {i}: {ftype} ({cmap_count} torsions) - CMAP scaling: {cfg['cmap_scaling_mode']}")
        elif isinstance(force, openmm.NonbondedForce):
            logger.info(f"  Force {i}: {ftype} ({force.getNumParticles()} particles)")
        else:
            logger.info(f"  Force {i}: {ftype}")

    if cmap_count > 0:
        logger.info(f"CMAP detected: {cmap_count} torsions")
        logger.info(f"CMAP scaling mode: {cfg['cmap_scaling_mode']}")

    return system, psf, pdb, box, params, cmap_idx


def identify_solute_atoms(topology, logger):
    """识别溶质原子 (蛋白质+配体)"""
    solute_atoms = set()
    solute_residues = []

    for atom in topology.atoms():
        resname = atom.residue.name.upper()
        if resname not in SOLVENT_RESNAMES:
            solute_atoms.add(atom.index)
            if atom.residue.id not in solute_residues:
                solute_residues.append(atom.residue.id)

    n_total = topology.getNumAtoms()
    n_solute = len(solute_atoms)
    logger.info(f"Solute: {n_solute} atoms ({len(solute_residues)} residues), "
               f"Solvent: {n_total - n_solute} atoms, Ratio: {n_solute/n_total*100:.1f}%")

    return solute_atoms


def identify_hot_residue_atoms(topology, cfg, logger):
    """识别局部加热残基的原子"""
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

    logger.info(f"[v2.1] Local heating enabled for {len(hot_atoms)} residues")
    logger.info(f"[v2.1] Total hot atoms: {len(all_hot_atoms)}")
    logger.info(f"[v2.1] Lambda_hot: {cfg['lambda_hot']} (additional scaling factor)")

    return all_hot_atoms, hot_atoms


def create_scaled_cmap_force(original_force, solute_atoms, lam, logger):
    """创建scaled CMAP force - GROMACS风格完整实现"""
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
    """创建带局部加热的REST2系统"""
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
    """使用scaled CMAP重建system"""
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
    """几何分布温度"""
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
        logger.info(f"[v2.1] Temperature spacing: min={min(dTs):.2f}K, max={max(dTs):.2f}K, avg={np.mean(dTs):.2f}K")

    return temps, lams


def save_full_checkpoint(cfg, iteration, contexts, energy_history, exchange_attempts,
                         exchange_accepts, temps, lams, delta_history, logger):
    """保存完整的checkpoint（包含位置和速度）"""
    ckpt_file = os.path.join(cfg["output_dir"], f"checkpoint_{iteration}_full.npz")

    # 收集所有副本的位置和速度
    positions_list = []
    velocities_list = []

    for ctx in contexts:
        state = ctx.getState(getPositions=True, getVelocities=True)
        # 转换为numpy数组
        pos = np.array([[p.x, p.y, p.z] for p in state.getPositions()])
        vel = np.array([[v.x, v.y, v.z] for v in state.getVelocities()])
        positions_list.append(pos)
        velocities_list.append(vel)

    # 保存所有数据
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

    # 添加每个副本的位置和速度
    for i, (pos, vel) in enumerate(zip(positions_list, velocities_list)):
        save_dict[f'positions_{i}'] = pos
        save_dict[f'velocities_{i}'] = vel

    # 添加delta历史
    for k, v in delta_history.items():
        if v:
            save_dict[f"delta_{k}"] = np.array(v)

    np.savez_compressed(ckpt_file, **save_dict)
    logger.info(f"  Full checkpoint saved: {ckpt_file}")

    return ckpt_file


def restore_from_checkpoint(checkpoint_file, cfg, logger):
    """从checkpoint恢复模拟状态"""
    logger.info(f"\n=== Restoring from Checkpoint ===")
    logger.info(f"  Loading: {checkpoint_file}")

    data = np.load(checkpoint_file, allow_pickle=True)

    # 恢复基本数据
    start_iteration = int(data['iteration'])
    energy_history = list(data['energies'])
    exchange_attempts = data['attempts']
    exchange_accepts = data['accepts']
    temps = data['temps']
    lams = data['lams']

    # 恢复delta历史
    delta_history = {}
    for k in range(len(temps) - 1):
        key = f'delta_{k}'
        if key in data:
            delta_history[k] = list(data[key])

    # 提取位置和速度
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


def run_rest2(cfg, logger, restore_data=None):
    """执行带局部加热的REST2模拟"""

    kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
    kB_val = kB.value_in_unit(unit.kilojoule_per_mole / unit.kelvin)

    logger.info(f"\n{'='*70}")
    logger.info(f"REST2 Local Heating Simulation [v{cfg['version']}]")
    logger.info(f"Temperature Range: {cfg['T_ref']}K - {cfg['T_max']}K")
    logger.info(f"Hot Residues: {cfg['hot_residue_ids']}")
    logger.info(f"Lambda_hot: {cfg['lambda_hot']}")
    if cfg.get("save_trajectory", False):
        logger.info(f"Trajectory Saving: Enabled")
    logger.info(f"{'='*70}")

    base_system, psf, pdb, box, params, cmap_idx = build_base_system(cfg, logger)
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
        if not restore_data or i < 5:  # 恢复时只显示前几个
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
            # 从checkpoint恢复位置和速度
            pos = restore_data['positions_list'][i]
            vel = restore_data['velocities_list'][i]
            # 转换为OpenMM单位
            pos_openmm = [openmm.Vec3(p[0], p[1], p[2]) for p in pos] * unit.angstrom
            vel_openmm = [openmm.Vec3(v[0], v[1], v[2]) for v in vel] * unit.angstrom/unit.picosecond
            ctx.setPositions(pos_openmm)
            ctx.setVelocities(vel_openmm)
            ctx.setPeriodicBoxVectors(*base_system.getDefaultPeriodicBoxVectors())
        else:
            ctx.setPositions(pdb.positions)
            ctx.setPeriodicBoxVectors(*base_system.getDefaultPeriodicBoxVectors())
            openmm.LocalEnergyMinimizer.minimize(ctx, 10*unit.kilojoule_per_mole/unit.nanometer)
            ctx.setVelocitiesToTemperature(T*unit.kelvin)

        contexts.append(ctx)

    if not restore_data:
        # 平衡阶段（仅首次运行）
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
            dcd_file_obj = open(dcd_file_path, 'ab')  # 使用追加模式
            dcd_writer = app.DCDFile(dcd_file_obj, pdb.topology, dt)
            dcd_writers.append(dcd_writer)
            dcd_files.append(dcd_file_obj)
            if i < 5:
                logger.info(f"  Replica {i}: {dcd_file_path}")
        logger.info(f"  Trajectory mode: Append (for resuming)")

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
    logger.info(f"{'='*70}\n")

    replica_state_history = []

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
            replica_state_history.append({
                'iteration': iteration,
                'energies': energies,
                'temperatures': temps,
            })

            # 保存轨迹帧
            if cfg.get("save_trajectory", False) and dcd_writers:
                if (iteration + 1) % cfg["trajectory_interval"] == 0:
                    for i, ctx in enumerate(contexts):
                        state = ctx.getState(getPositions=True)
                        pos = state.getPositions()
                        dcd_writers[i].writeModel(pos)

            if (iteration+1) % 20 == 0:
                rates = [f"{exchange_accepts[k]/exchange_attempts[k]*100:.0f}%"
                        if exchange_attempts[k] > 0 else "-" for k in range(n_rep-1)]
                logger.info(f"  Iter {iteration+1:4d}/{n_iter}: "
                           f"E={[int(e) for e in energies[:4]]}..., "
                           f"Rates=[{','.join(rates[:4])}...]")

            if (iteration+1) % cfg["checkpoint_interval"] == 0:
                # 保存完整checkpoint
                save_full_checkpoint(cfg, iteration+1, contexts, energy_history, exchange_attempts,
                                    exchange_accepts, temps, lams, delta_history, logger)
                # 同时保存简化checkpoint
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
        'replica_state_history': replica_state_history,
        'version': cfg['version'],
        'cmap_scaling_mode': cfg['cmap_scaling_mode'],
        'local_heating': cfg['local_heating_enabled'],
        'hot_residue_ids': cfg['hot_residue_ids'],
        'hot_residue_names': cfg['hot_residue_names'],
        'lambda_hot': cfg['lambda_hot'],
    }

    save_final_results(cfg, results, contexts, pdb)
    generate_scientific_visualizations(results, cfg["output_dir"], logger)
    save_scientific_report(cfg, results, logger)

    return results


def save_checkpoint(cfg, iteration, energy_history, attempts, accepts, temps, lams, deltas):
    """保存简化checkpoint（仅统计数据）"""
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
        **{f"delta_{k}": np.array(v) for k, v in results['delta_history'].items() if v})

    for i, ctx in enumerate(contexts):
        state = ctx.getState(getPositions=True)
        pdb_file = os.path.join(cfg["output_dir"], f"replica_{i}_final.pdb")
        with open(pdb_file, 'w') as f:
            app.PDBFile.writeFile(pdb.topology, state.getPositions(), f)

    mbar_file = os.path.join(cfg["mbar_dir"], "reduced_energies.npz")
    np.savez_compressed(mbar_file,
        u_kn=results['energies'],
        N_k=np.ones(len(results['temperatures'])),
        temps=results['temperatures'],
        lambda_hot=results['lambda_hot'])


def generate_scientific_visualizations(results, output_dir, logger):
    """生成科研标准可视化"""
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

    # Fig 1: 综合分析
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
    """保存科研报告"""
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
    }

    report_file = os.path.join(cfg["output_dir"], "scientific_report.json")
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info(f"  Scientific report saved to {report_file}")


if __name__ == "__main__":
    args = parse_command_line()

    manager = ExperimentManager()

    # 处理恢复模式
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

    # 应用命令行参数
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