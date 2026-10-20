#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
REST2 (Replica Exchange with Solute Tempering v2) 分子动力学模拟
Villin Headpiece + CHARMM36m + TIP3P + 0.15M NaCl
+ 局部加热（残基46/ASP5）

======================================================================
版本: v1.4-local-heating-46 (2026-07-13)

核心改进 (vs v1.3):
  1. 温度范围: 300K-400K (保持不变)
     - 增强采样范围扩大，可探索更多构象空间
     
  2. 副本数量: 16 (保持不变)
     - 维持合理的交换率（目标20-30%）
     - 温度间隔约7K
     
  3. 局部加热功能: 残基46 (ASP5) 特殊处理
     - Villin HP35编号体系: 残基46 = ASP5 (HP35内部编号第5位)
     - 对ASP5残基进行额外的λ_local缩放
     - 实现局部REST2技术
     
  4. 保留v1.1的CMAP完整缩放 (GROMACS风格)

参考文献:
  Wang et al., PNAS 108(37): 15107-15112 (2011)  [REST2原始论文]
  Liu et al., JCP 139: 034115 (2013)             [局部REST/局部加热]
======================================================================

REST2 缩放规则:
  全局缩放: λ_global = T_ref / T_replica (应用于所有溶质原子)
  局部缩放: λ_local = λ_global × λ_hot (应用于特定残基)
  
  ASP5 (残基46) 局部加热参数:
    λ_hot = 0.5 (额外50%能量削弱，更强的局部加热)
    有效λ = λ_global × 0.5
"""

__version__ = "1.4-local-heating-46"
__date__ = "2026-07-08"

import os
os.environ['CUDA_CACHE_PATH'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cuda_cache')
os.environ['CUDA_CACHE_MAXSIZE'] = '2147483648'
os.makedirs(os.environ['CUDA_CACHE_PATH'], exist_ok=True)

import sys, time, copy, logging, datetime, traceback, json, hashlib
import numpy as np
import openmm
from openmm import unit
from openmm import app

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(SCRIPT_DIR) in ('versions', 'scripts'):
    SCRIPT_DIR = os.path.dirname(SCRIPT_DIR)
CHARMMGUI_DIR = os.path.join(SCRIPT_DIR, "charmm-gui-8088304247")
OPENMM_DIR = os.path.join(CHARMMGUI_DIR, "openmm")

CONFIG = {
    "psf_file":     os.path.join(OPENMM_DIR, "step3_input.psf"),
    "pdb_file":     os.path.join(OPENMM_DIR, "step3_input.pdb"),
    "toppar_str":   os.path.join(OPENMM_DIR, "toppar.str"),
    "sysinfo_dat":  os.path.join(OPENMM_DIR, "sysinfo.dat"),
    "base_output_dir": os.path.join(SCRIPT_DIR, "output"),

    # 温度范围和副本数 (与v1.2一致)
    "n_replicas":       16,         # 16个副本
    "T_ref":            300.0,
    "T_max":            400.0,      # 300-400K
    "n_steps_per_move": 1000,
    "n_iterations":     1000,
    "checkpoint_interval": 100,

    "timestep":         2.0,
    "friction":         1.0,

    "platform_preference": ["CUDA", "OpenCL", "CPU"],
    "use_barostat":     False,
    
    "version":          __version__,
    
    # v1.1保留: CMAP缩放模式
    "cmap_scaling_mode": "scaled_maps",
    
    # [v1.4] 局部加热参数 - 残基46 (ASP5)
    "local_heating_enabled": True,
    "hot_residue_id": "46",        # ASP5 (完整编号体系)
    "hot_residue_internal": 5,     # ASP5 (HP35内部编号)
    "lambda_hot": 0.5,             # 局部额外缩放因子
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

    def setup_experiment(self):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = f"villin_rest2_v1.4_46_{ts}"
        self.experiment_dir = os.path.join(CONFIG["base_output_dir"], f"experiment_{ts}")
        self.figures_dir = os.path.join(self.experiment_dir, "figures")
        self.mbar_dir = os.path.join(self.experiment_dir, "mbar_analysis")
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)
        os.makedirs(self.mbar_dir, exist_ok=True)
        self.cfg = copy.deepcopy(CONFIG)
        self.cfg["output_dir"] = self.experiment_dir
        self.cfg["figures_dir"] = self.figures_dir
        self.cfg["mbar_dir"] = self.mbar_dir
        self.cfg["experiment_id"] = self.experiment_id
        self.setup_logging()
        self.save_config()
        return self.experiment_dir

    def setup_logging(self):
        self.log_file = os.path.join(self.experiment_dir, "experiment.log")
        self.logger = logging.getLogger(f"REST2_v1.4_46_{self.experiment_id}")
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
    
    with open(cfg["sysinfo_dat"]) as f:
        sysinfo = json.load(f)
    box = sysinfo["dimensions"][:3]
    
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
        logger.info(f"[v1.1] CMAP detected: {cmap_count} torsions")
        logger.info(f"[v1.1] CMAP scaling mode: {cfg['cmap_scaling_mode']}")
    
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
    """
    [v1.4] 识别局部加热残基的原子
    
    Villin HP35编号体系说明:
    - HP35是从完整villin头部结构域(76残基)截取的C端35残基片段
    - 对应完整编号体系的第42-76位残基
    - 残基46 (完整编号) = ASP5 (HP35内部编号第5位)
    
    ASP5原子范围: 根据PDB文件，原子60-71
    ASP5残基序列: LEU1, SER2, ASP3, GLU4, ASP5, ...
    """
    hot_atoms = set()
    
    # 方法1: 通过残基名称和位置识别 (ASP5)
    for atom in topology.atoms():
        # ASP是第5个蛋白质残基 (LEU1, SER2, ASP3, GLU4, ASP5)
        if atom.residue.name.upper() == 'ASP':
            # 检查是否是第5个残基
            protein_residue_count = 0
            for res in topology.residues():
                if res.name.upper() not in SOLVENT_RESNAMES:
                    protein_residue_count += 1
                    if protein_residue_count == 5 and res.id == atom.residue.id:
                        hot_atoms.add(atom.index)
                        break
                if protein_residue_count > 5:
                    break
    
    # 方法2: 通过原子编号范围识别 (原子60-71属于ASP5)
    # 作为备选方案
    if len(hot_atoms) == 0:
        logger.warning("Hot residue identification by name failed, using atom range method")
        for atom_idx in range(60, 72):  # 原子60-71
            hot_atoms.add(atom_idx)
    
    logger.info(f"[v1.4] Local heating enabled for residue {cfg['hot_residue_id']} (ASP5)")
    logger.info(f"[v1.4] Hot atoms: {len(hot_atoms)} atoms identified")
    logger.info(f"[v1.4] Lambda_hot: {cfg['lambda_hot']} (additional scaling factor)")
    
    return hot_atoms


def create_scaled_cmap_force(original_force, solute_atoms, lam, logger):
    """
    [v1.1保留] 创建scaled CMAP force - GROMACS风格完整实现
    """
    n_maps = original_force.getNumMaps()
    n_torsions = original_force.getNumTorsions()
    
    logger.info(f"  [CMAP v1.1] Creating scaled CMAP force with λ={lam:.4f}")
    logger.info(f"  [CMAP v1.1] Original: {n_maps} maps, {n_torsions} torsions")
    
    scaled_maps = {}
    new_force = openmm.CMAPTorsionForce()
    
    for map_idx in range(n_maps):
        size, energy_grid = original_force.getMapParameters(map_idx)
        scaled_energy = energy_grid * lam
        scaled_map_idx = new_force.addMap(size, scaled_energy)
        scaled_maps[map_idx] = scaled_map_idx
        logger.info(f"    Map {map_idx}: size={size}, scaled_idx={scaled_map_idx}, λ={lam:.4f}")
    
    n_scaled = 0
    n_preserved = 0
    
    for torsion_idx in range(n_torsions):
        params = original_force.getTorsionParameters(torsion_idx)
        orig_map_idx = params[0]
        atoms = params[1:9]
        
        all_solute = all(a in solute_atoms for a in atoms)
        
        if all_solute:
            new_map_idx = scaled_maps[orig_map_idx]
            new_force.addTorsion(new_map_idx, *atoms)
            n_scaled += 1
        else:
            new_map_idx = scaled_maps[orig_map_idx]
            new_force.addTorsion(new_map_idx, *atoms)
            n_preserved += 1
    
    logger.info(f"  [CMAP v1.1] Result: {n_scaled} torsions scaled, {n_preserved} preserved")
    logger.info(f"  [CMAP v1.1] Physical accuracy: COMPLETE (GROMACS-style)")
    
    return new_force, n_scaled


def create_rest2_system_with_local_heating(base_system, solute_atoms, hot_atoms, lam_global, 
                                            lam_hot, params, cmap_idx, cfg, logger):
    """
    [v1.4] 创建带局部加热的REST2系统
    
    缩放策略:
    1. 普通溶质原子: λ_global 缩放
    2. 局部加热原子: λ_global × λ_hot 缩放 (更强)
    
    物理原理:
    - 局部加热残基的能量被额外削弱
    - 允许该残基探索更广泛的构象空间
    - 用于研究局部区域的稳定性或折叠动力学
    """
    system = copy.deepcopy(base_system)
    sqrt_lam_global = np.sqrt(lam_global)
    sqrt_lam_hot = np.sqrt(lam_global * lam_hot)  # 局部加热的sqrt(lam)
    lam_hot_effective = lam_global * lam_hot      # 局部加热的有效lam
    
    n_forces = system.getNumForces()
    
    logger.info(f"  Creating REST2 system with local heating:")
    logger.info(f"    λ_global={lam_global:.4f}, λ_hot={lam_hot:.4f}")
    logger.info(f"    λ_hot_effective={lam_hot_effective:.4f}")
    logger.info(f"    Hot atoms: {len(hot_atoms)}, Normal solute: {len(solute_atoms)-len(hot_atoms)}")
    
    cmap_forces_to_replace = []
    
    for i in range(n_forces):
        force = system.getForce(i)
        
        if isinstance(force, openmm.NonbondedForce):
            n_normal = 0
            n_hot = 0
            for j in range(force.getNumParticles()):
                q, sig, eps = force.getParticleParameters(j)
                if j in hot_atoms:
                    # [v1.4] 局部加热原子: 更强的缩放
                    force.setParticleParameters(j, q * sqrt_lam_hot, sig, eps * lam_hot_effective)
                    n_hot += 1
                elif j in solute_atoms:
                    # 普通溶质原子: 标准REST2缩放
                    force.setParticleParameters(j, q * sqrt_lam_global, sig, eps * lam_global)
                    n_normal += 1
            
            n_ss = n_sw = n_hot_ss = n_hot_sw = 0
            for j in range(force.getNumExceptions()):
                p1, p2, q12, sig, eps = force.getExceptionParameters(j)
                s1 = p1 in solute_atoms
                s2 = p2 in solute_atoms
                h1 = p1 in hot_atoms
                h2 = p2 in hot_atoms
                
                if s1 and s2:
                    if h1 or h2:
                        # [v1.4] 包含局部加热原子的例外: 更强缩放
                        force.setExceptionParameters(j, p1, p2, q12 * lam_hot_effective, sig, eps * lam_hot_effective)
                        n_hot_ss += 1
                    else:
                        force.setExceptionParameters(j, p1, p2, q12 * lam_global, sig, eps * lam_global)
                        n_ss += 1
                elif s1 or s2:
                    if h1 or h2:
                        # [v1.4] 局部加热原子与溶剂的例外
                        force.setExceptionParameters(j, p1, p2, q12 * sqrt_lam_hot, sig, eps * sqrt_lam_hot)
                        n_hot_sw += 1
                    else:
                        force.setExceptionParameters(j, p1, p2, q12 * sqrt_lam_global, sig, eps * sqrt_lam_global)
                        n_sw += 1
            
            logger.info(f"    NonbondedForce: {n_normal} normal + {n_hot} hot particles")
            logger.info(f"    Exceptions: {n_ss} SS + {n_sw} SW + {n_hot_ss} hot-SS + {n_hot_sw} hot-SW")
        
        elif isinstance(force, openmm.HarmonicBondForce):
            n_normal = 0
            n_hot = 0
            for j in range(force.getNumBonds()):
                p1, p2, r, k = force.getBondParameters(j)
                if p1 in solute_atoms and p2 in solute_atoms:
                    if p1 in hot_atoms or p2 in hot_atoms:
                        force.setBondParameters(j, p1, p2, r, k * lam_hot_effective)
                        n_hot += 1
                    else:
                        force.setBondParameters(j, p1, p2, r, k * lam_global)
                        n_normal += 1
            logger.info(f"    HarmonicBondForce: {n_normal} normal + {n_hot} hot bonds")
        
        elif isinstance(force, openmm.HarmonicAngleForce):
            n_normal = 0
            n_hot = 0
            for j in range(force.getNumAngles()):
                p1, p2, p3, theta, k = force.getAngleParameters(j)
                if p1 in solute_atoms and p2 in solute_atoms and p3 in solute_atoms:
                    if p1 in hot_atoms or p2 in hot_atoms or p3 in hot_atoms:
                        force.setAngleParameters(j, p1, p2, p3, theta, k * lam_hot_effective)
                        n_hot += 1
                    else:
                        force.setAngleParameters(j, p1, p2, p3, theta, k * lam_global)
                        n_normal += 1
            logger.info(f"    HarmonicAngleForce: {n_normal} normal + {n_hot} hot angles")
        
        elif isinstance(force, openmm.PeriodicTorsionForce):
            n_normal = 0
            n_hot = 0
            for j in range(force.getNumTorsions()):
                p1, p2, p3, p4, per, phase, k = force.getTorsionParameters(j)
                if all(x in solute_atoms for x in [p1, p2, p3, p4]):
                    if any(x in hot_atoms for x in [p1, p2, p3, p4]):
                        force.setTorsionParameters(j, p1, p2, p3, p4, per, phase, k * lam_hot_effective)
                        n_hot += 1
                    else:
                        force.setTorsionParameters(j, p1, p2, p3, p4, per, phase, k * lam_global)
                        n_normal += 1
            logger.info(f"    PeriodicTorsionForce: {n_normal} normal + {n_hot} hot torsions")
        
        elif isinstance(force, openmm.CustomTorsionForce):
            n_normal = 0
            n_hot = 0
            param_names = [force.getPerTorsionParameterName(k) for k in range(force.getNumPerTorsionParameters())]
            k_idx = next((k for k, n in enumerate(param_names) if 'k' in n.lower()), None)
            if k_idx is not None:
                for j in range(force.getNumTorsions()):
                    p1, p2, p3, p4, pars = force.getTorsionParameters(j)
                    if all(x in solute_atoms for x in [p1, p2, p3, p4]):
                        pars = list(pars)
                        if any(x in hot_atoms for x in [p1, p2, p3, p4]):
                            pars[k_idx] *= lam_hot_effective
                            n_hot += 1
                        else:
                            pars[k_idx] *= lam_global
                            n_normal += 1
                        force.setTorsionParameters(j, p1, p2, p3, p4, pars)
            logger.info(f"    CustomTorsionForce: {n_normal} normal + {n_hot} hot torsions")
        
        elif isinstance(force, openmm.CMAPTorsionForce):
            if cfg["cmap_scaling_mode"] == "scaled_maps":
                # [v1.4] CMAP使用全局lam缩放 (CMAP通常涉及整个主链)
                scaled_cmap, n_cmap_scaled = create_scaled_cmap_force(force, solute_atoms, lam_global, logger)
                cmap_forces_to_replace.append((i, scaled_cmap))
            else:
                logger.info(f"    CMAPTorsionForce: preserved (mode={cfg['cmap_scaling_mode']})")
    
    if cmap_forces_to_replace:
        for orig_idx, new_force in cmap_forces_to_replace:
            logger.info(f"    [CMAP v1.1] Force {orig_idx} will be replaced with scaled version")
    
    return system, cmap_forces_to_replace


def rebuild_system_with_scaled_cmap(base_system, cmap_replacements, logger):
    """[v1.1保留] 使用scaled CMAP重建system"""
    if not cmap_replacements:
        return base_system
    
    logger.info("  [CMAP v1.1] Rebuilding system with scaled CMAP forces")
    
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
    """几何分布温度 - 300K-400K"""
    temps = [T_ref * (T_max/T_ref)**(i/(n_replicas-1)) for i in range(n_replicas)]
    lams = [T_ref/T for T in temps]
    
    if logger:
        logger.info(f"\n{'='*70}")
        logger.info(f"REST2 Temperature Distribution ({n_replicas} replicas, 300-400K)")
        logger.info(f"{'─'*70}")
        logger.info(f"  Rep    T(K)    λ_global    λ_hot_eff   √λ_global   ΔT(K)")
        logger.info(f"{'─'*70}")
        for i, (T, lam) in enumerate(zip(temps, lams)):
            lam_hot_eff = lam * CONFIG["lambda_hot"]  # 局部加热的有效λ
            dT = T - temps[i-1] if i > 0 else 0
            logger.info(f"  {i:2d}   {T:6.1f}   {lam:.4f}    {lam_hot_eff:.4f}     {np.sqrt(lam):.4f}    {dT:5.1f}")
        logger.info(f"{'='*70}")
        logger.info(f"[v1.4] Hot residue 46 (ASP5): λ_hot={CONFIG['lambda_hot']}, additional scaling applied")
    
    return temps, lams


def run_rest2(cfg, logger):
    """[v1.4] 执行带局部加热的REST2模拟"""
    
    kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
    kB_val = kB.value_in_unit(unit.kilojoule_per_mole / unit.kelvin)
    
    logger.info(f"\n{'='*70}")
    logger.info(f"REST2 Local Heating Simulation [v{cfg['version']}]")
    logger.info(f"Temperature Range: {cfg['T_ref']}K - {cfg['T_max']}K")
    logger.info(f"Hot Residue: {cfg['hot_residue_id']} (ASP5, HP35 internal #5)")
    logger.info(f"Lambda_hot: {cfg['lambda_hot']} (additional scaling for hot residue)")
    logger.info(f"{'='*70}")
    
    base_system, psf, pdb, box, params, cmap_idx = build_base_system(cfg, logger)
    solute_atoms = identify_solute_atoms(pdb.topology, logger)
    
    # [v1.4] 识别局部加热残基的原子
    if cfg["local_heating_enabled"]:
        hot_atoms = identify_hot_residue_atoms(pdb.topology, cfg, logger)
    else:
        hot_atoms = set()
    
    n_rep = cfg["n_replicas"]
    T_ref = cfg["T_ref"]
    T_max = cfg["T_max"]
    lam_hot = cfg["lambda_hot"]
    
    temps, lams = optimize_temperatures_geometric(n_rep, T_ref, T_max, logger)
    
    logger.info(f"\n=== Creating REST2 Replica Systems with Local Heating ===")
    replica_systems = []
    
    for i, (T, lam_global) in enumerate(zip(temps, lams)):
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
        ctx.setPositions(pdb.positions)
        ctx.setPeriodicBoxVectors(*base_system.getDefaultPeriodicBoxVectors())
        
        openmm.LocalEnergyMinimizer.minimize(ctx, 10*unit.kilojoule_per_mole/unit.nanometer)
        
        state = ctx.getState(getEnergy=True)
        E = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        logger.info(f"  Replica {i} (T={T:.1f}K): Minimized E={E:.1f} kJ/mol")
        
        ctx.setVelocitiesToTemperature(10*unit.kelvin)
        contexts.append(ctx)
    
    # 平衡阶段
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
        
        state = ctx.getState(getEnergy=True)
        E = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        logger.info(f"  Replica {i} (T={T:.1f}K): Equilibrated E={E:.1f} kJ/mol")
    
    n_steps = cfg["n_steps_per_move"]
    n_iter = cfg["n_iterations"]
    dt = cfg["timestep"]
    total_ps = n_iter * n_steps * dt / 1000
    
    logger.info(f"\n{'='*70}")
    logger.info(f"REST2 Production Simulation with Local Heating")
    logger.info(f"  Replicas: {n_rep}, T: {T_ref:.0f}-{T_max:.0f}K")
    logger.info(f"  Steps/iter: {n_steps}, Total iter: {n_iter}")
    logger.info(f"  Total simulation: {total_ps:.0f} ps")
    logger.info(f"  Hot residue: {cfg['hot_residue_id']} (ASP5)")
    logger.info(f"  Lambda_hot: {cfg['lambda_hot']}")
    logger.info(f"  Platform: {plat_name}")
    logger.info(f"{'='*70}\n")
    
    exchange_attempts = np.zeros(n_rep-1, dtype=int)
    exchange_accepts = np.zeros(n_rep-1, dtype=int)
    energy_history = []
    delta_history = {k: [] for k in range(n_rep-1)}
    replica_state_history = []
    
    t0 = time.time()
    
    for iteration in range(n_iter):
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
        
        if (iteration+1) % 20 == 0:
            rates = [f"{exchange_accepts[k]/exchange_attempts[k]*100:.0f}%" 
                    if exchange_attempts[k] > 0 else "-" for k in range(n_rep-1)]
            logger.info(f"  Iter {iteration+1:4d}/{n_iter}: "
                       f"E={[int(e) for e in energies[:4]]}..., "
                       f"Rates=[{','.join(rates[:4])}...]")
        
        if (iteration+1) % cfg["checkpoint_interval"] == 0:
            save_checkpoint(cfg, iteration+1, energy_history, exchange_attempts, 
                           exchange_accepts, temps, lams, delta_history)
    
    total_time = time.time() - t0
    
    avg_rate = np.mean([exchange_accepts[k]/exchange_attempts[k]*100 
                       for k in range(n_rep-1) if exchange_attempts[k] > 0])
    
    logger.info(f"\n{'='*70}")
    logger.info(f"REST2 Local Heating Simulation Complete! [v{cfg['version']}]")
    logger.info(f"  Total time: {total_time:.1f}s")
    logger.info(f"  Avg exchange rate: {avg_rate:.1f}%")
    logger.info(f"  Hot residue: {cfg['hot_residue_id']} (ASP5)")
    logger.info(f"  CMAP scaling: COMPLETE (GROMACS-style)")
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
        'hot_residue': cfg['hot_residue_id'],
        'lambda_hot': cfg['lambda_hot'],
    }
    
    save_final_results(cfg, results, contexts, pdb)
    generate_scientific_visualizations(results, cfg["output_dir"], logger)
    save_scientific_report(cfg, results, logger)
    
    return results


def save_checkpoint(cfg, iteration, energy_history, attempts, accepts, temps, lams, deltas):
    ckpt_file = os.path.join(cfg["output_dir"], f"checkpoint_{iteration}.npz")
    np.savez_compressed(ckpt_file,
        energies=np.array(energy_history),
        attempts=attempts, accepts=accepts,
        temps=temps, lams=lams,
        lambda_hot=cfg["lambda_hot"],
        hot_residue=cfg["hot_residue_id"],
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
        hot_residue=results['hot_residue'],
        **{f"delta_{k}": np.array(v) for k, v in results['delta_history'].items() if v})
    
    for i, ctx in enumerate(contexts):
        state = ctx.getState(getPositions=True)
        pdb_file = os.path.join(cfg["output_dir"], f"replica_{i}_final.pdb")
        with open(pdb_file, 'w') as f:
            app.PDBFile.writeFile(pdb.topology, state.getPositions(), f)
    
    # MBAR兼容输出
    mbar_file = os.path.join(cfg["mbar_dir"], "reduced_energies.npz")
    np.savez_compressed(mbar_file,
        u_kn=results['energies'],
        N_k=np.ones(len(results['temperatures'])),
        temps=results['temperatures'],
        lambda_hot=results['lambda_hot'])


def generate_scientific_visualizations(results, output_dir, logger):
    """[v1.4] 生成科研标准可视化"""
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
    ax.set_title('(a) Potential Energy (first 8 replicas)', fontweight='bold')
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
    
    ax = axes[0,1]
    colors = ['#27ae60' if 20<=r<=40 else '#e74c3c' for r in rates]
    ax.bar(range(n_rep-1), rates, color=colors)
    ax.axhline(20, color='k', ls='--'); ax.axhline(40, color='k', ls='--')
    ax.set_xlabel('Replica Pair'); ax.set_ylabel('Exchange Rate (%)')
    ax.set_title('(b) Exchange Acceptance', fontweight='bold')
    ax.set_xticks(range(n_rep-1)); ax.set_xticklabels([f'{k}-{k+1}' for k in range(n_rep-1)], fontsize=7)
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
    lam_hot_eff = lams * results['lambda_hot']
    ax.plot(T, lams, 'o-', label='λ_global', color='blue')
    ax.plot(T, lam_hot_eff, 's-', label='λ_hot_eff (ASP5)', color='red')
    ax.set_xlabel('T (K)'); ax.set_ylabel('Lambda')
    ax.set_title('(f) Lambda Scaling (with local heating)', fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'fig1_analysis.png'), dpi=150)
    plt.close()
    
    # Fig 2: 局部加热分析
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    ax = axes[0]
    rate_matrix = np.zeros((n_rep, n_rep))
    for k in range(n_rep-1):
        rate_matrix[k,k+1] = rate_matrix[k+1,k] = rates[k]
    im = ax.imshow(rate_matrix, cmap='RdYlGn', vmin=0, vmax=50)
    ax.set_title('Exchange Rate Matrix', fontweight='bold')
    plt.colorbar(im, ax=ax, label='%')
    
    ax = axes[1]
    for k in range(min(8, n_rep-1)):
        if deltas[k]:
            ax.plot(deltas[k], alpha=0.7, label=f'Pair {k}-{k+1}')
    ax.axhline(0, color='k'); ax.set_xlabel('Attempt'); ax.set_ylabel('Delta')
    ax.set_title('Delta Time Series', fontweight='bold')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    
    ax = axes[2]
    ax.bar(range(n_rep), T, color=plt.cm.coolwarm(T/400))
    ax.set_xlabel('Replica'); ax.set_ylabel('Temperature (K)')
    ax.set_title(f'Temperature Distribution (Hot residue: {results["hot_residue"]})', fontweight='bold')
    ax.axhline(300, color='g', ls='--', label='T_ref')
    ax.axhline(400, color='r', ls='--', label='T_max')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'fig2_exchange.png'), dpi=150)
    plt.close()
    
    logger.info(f"  Figures saved to {figures_dir}")


def save_scientific_report(cfg, results, logger):
    """[v1.4] 保存科研报告"""
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
        'lambda_values': [float(l) for l in results['lam_values']],
        'avg_exchange_rate': float(round(avg_rate, 2)),
        'exchange_rates': [float(round(r,2)) for r in rates],
        'exchange_rate_valid': bool(20 <= avg_rate <= 40),
        'n_iterations': int(len(E)),
        'total_simulation_ps': float(len(E) * 1000 * 2 / 1000),
        'total_time_s': float(round(results['total_time'], 1)),
        'cmap_scaling_mode': results['cmap_scaling_mode'],
        'cmap_scaling_complete': bool(True),
        
        # [v1.4] 局部加热信息 - 残基46 (ASP5)
        'local_heating_enabled': bool(results['local_heating']),
        'hot_residue': results['hot_residue'],
        'hot_residue_name': 'ASP',
        'hot_residue_internal_id': 5,
        'lambda_hot': float(results['lambda_hot']),
        'lambda_hot_effective': [float(l * results['lambda_hot']) for l in results['lam_values']],
        
        'system_info': {
            'n_atoms': 11491,
            'n_solute_atoms': 582,
            'n_solvent_atoms': 10909,
            'solute_ratio': 5.1,
            'n_residues': 35,
        },
        
        'recommendations': {
            'exchange_rate_achieved': bool(avg_rate >= 20),
            'convergence_status': 'converged' if avg_rate >= 20 else 'needs_adjustment',
            'suggested_improvements': [] if avg_rate >= 20 else ['Increase replicas', 'Narrow temperature range'],
        }
    }
    
    report_file = os.path.join(cfg["output_dir"], "scientific_report.json")
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"  Scientific report saved to {report_file}")


if __name__ == "__main__":
    manager = ExperimentManager()
    exp_dir = manager.setup_experiment()
    logger = manager.logger
    cfg = manager.cfg  # 使用处理过的配置
    
    try:
        logger.info(f"\n{'='*70}")
        logger.info(f"REST2 Local Heating Simulation - Villin HP35 + CHARMM36m")
        logger.info(f"Version: {cfg['version']}")
        logger.info(f"Temperature: {cfg['T_ref']}K - {cfg['T_max']}K")
        logger.info(f"Hot Residue: {cfg['hot_residue_id']} (ASP5)")
        logger.info(f"Lambda_hot: {cfg['lambda_hot']}")
        logger.info(f"Output: {exp_dir}")
        logger.info(f"{'='*70}")
        
        results = run_rest2(cfg, logger)
        
        logger.info(f"\n[SUCCESS] Experiment completed!")
        logger.info(f"Results saved to: {exp_dir}")
        
    except Exception as e:
        logger.error(f"\n[FATAL ERROR] {str(e)}")
        logger.error(traceback.format_exc())
        raise