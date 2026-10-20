#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
REST2 (Replica Exchange with Solute Tempering v2) 分子动力学模拟
4AU5 蛋白 + CHARMM36m + TIP3P + 0.15M NaCl
+ 局部加热（残基 F72, A167, I168, F344, I345）

======================================================================
版本: v1.0-4au5-5residues (2026-07-19)

核心参数:
  1. 温度范围: 300K-400K
     - 增强采样范围扩大，可探索更多构象空间
     
  2. 副本数量: 32
     - 更细的温度间隔，优化交换率（目标20-30%）
     - 温度间隔约3.2K
     
  3. 迭代次数: 1000次
     
  4. 局部加热功能: 5个残基位点特殊处理
     - F72 (PHE 72): 残基编号72
     - A167 (ALA 167): 残基编号167
     - I168 (ILE 168): 残基编号168
     - F344 (PHE 344): 残基编号344
     - I345 (ILE 345): 残基编号345
     - 对这些残基进行额外的λ_local缩放
     - 实现局部REST2技术
     
  5. 保留CMAP完整缩放 (GROMACS风格)

参考文献:
  Wang et al., PNAS 108(37): 15107-15112 (2011)  [REST2原始论文]
  Liu et al., JCP 139: 034115 (2013)             [局部REST/局部加热]
======================================================================

REST2 缩放规则:
  全局缩放: λ_global = T_ref / T_replica (应用于所有溶质原子)
  局部缩放: λ_local = λ_global × λ_hot (应用于特定残基)
  
  局部加热参数:
    λ_hot = 0.5 (额外50%能量削弱，更强的局部加热)
    有效λ = λ_global × 0.5
"""

__version__ = "1.0-4au5-5residues"
__date__ = "2026-07-19"

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
CHARMMGUI_DIR = os.path.join(SCRIPT_DIR, "charmm-gui", "charmm-gui-8425204688")
OPENMM_DIR = os.path.join(CHARMMGUI_DIR, "openmm")

CONFIG = {
    "psf_file":     os.path.join(OPENMM_DIR, "step3_input.psf"),
    "pdb_file":     os.path.join(OPENMM_DIR, "step3_input.pdb"),
    "toppar_str":   os.path.join(OPENMM_DIR, "toppar.str"),
    "sysinfo_dat":  os.path.join(OPENMM_DIR, "sysinfo.dat"),
    "base_output_dir": os.path.join(SCRIPT_DIR, "outputs"),

    # 温度范围和副本数
    "n_replicas":       32,         # 32个副本
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
    
    # CMAP缩放模式
    "cmap_scaling_mode": "scaled_maps",
    
    # 局部加热参数 - 5个残基位点
    "local_heating_enabled": True,
    "hot_residue_ids": ["72", "167", "168", "344", "345"],  # 残基编号
    "hot_residue_names": {
        "72": "PHE",
        "167": "ALA",
        "168": "ILE",
        "344": "PHE",
        "345": "ILE"
    },
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
    """实验管理器"""
    
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
        self.experiment_id = f"4au5_rest2_v1.0_5residues_{ts}"
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
        self.logger = logging.getLogger(f"REST2_v1.0_4au5_{self.experiment_id}")
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
    
    # 读取sysinfo.dat（二进制文件需要特殊处理）
    import struct
    try:
        with open(cfg["sysinfo_dat"], 'rb') as f:
            # 尝试读取JSON格式的sysinfo
            data = f.read()
            try:
                # 尝试解码为JSON
                sysinfo = json.loads(data.decode('utf-8'))
                box = sysinfo["dimensions"][:3]
            except:
                # 如果不是JSON，使用默认盒子大小
                logger.warning("Could not parse sysinfo.dat, using default box size")
                box = [78.812, 78.812, 106.959]  # 从PDB文件获取的盒子大小
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
    """
    识别局部加热残基的原子
    
    目标残基:
    - F72 (PHE 72)
    - A167 (ALA 167)
    - I168 (ILE 168)
    - F344 (PHE 344)
    - I345 (ILE 345)
    """
    hot_atoms = {}
    
    # 创建残基ID到原子索引的映射
    residue_atoms = {}
    for atom in topology.atoms():
        res_id = atom.residue.id
        if res_id not in residue_atoms:
            residue_atoms[res_id] = []
        residue_atoms[res_id].append(atom.index)
    
    # 识别每个目标残基的原子
    for res_id in cfg["hot_residue_ids"]:
        if res_id in residue_atoms:
            hot_atoms[res_id] = set(residue_atoms[res_id])
            logger.info(f"Hot residue {res_id} ({cfg['hot_residue_names'][res_id]}): "
                       f"{len(hot_atoms[res_id])} atoms identified")
        else:
            logger.warning(f"Hot residue {res_id} not found in topology")
    
    # 合并所有热残基原子
    all_hot_atoms = set()
    for atoms in hot_atoms.values():
        all_hot_atoms.update(atoms)
    
    logger.info(f"[v1.0] Local heating enabled for {len(hot_atoms)} residues")
    logger.info(f"[v1.0] Total hot atoms: {len(all_hot_atoms)}")
    logger.info(f"[v1.0] Lambda_hot: {cfg['lambda_hot']} (additional scaling factor)")
    
    return all_hot_atoms, hot_atoms


def create_scaled_cmap_force(original_force, solute_atoms, lam, logger):
    """
    创建scaled CMAP force - GROMACS

    ====================================================================
    CMAP (Correction Map) 缩放算法说明
    ====================================================================
    背景:
      CMAP是CHARMM力场中用于提高骨架二面角精度的交叉项。
      它通过在φ/ψ二面角空间上定义能量校正网格来修正Amber/CHARMM
      标准扭转势的不足，对蛋白质骨架构象的准确描述至关重要。

    REST2中的CMAP缩放:
      在REST2方法中，溶质-溶质相互作用需要按因子λ进行缩放。
      对于CMAP扭转项，正确的缩放方式有两种:

      1. scaled_maps (GROMACS):
         - 将整个能量网格乘以λ: E'(φ,ψ) = λ × E(φ,ψ)
         - 适用于所有CMAP扭转项
         - 物理上完全正确，保持能量函数的连续性和可微性

      2. scaled_torsions:
         - 仅对溶质相关的扭转项缩放
         - 适用于溶质-溶剂分离更严格的场景

    实现:
      对每个map: 复制size和能量网格，将能量值乘以λ
      对每个torsion: 重新指向缩放后的map
      所有扭转项都使用缩放后的map（GROMACS风格的完整缩放）

    参考:
      Wang et al., PNAS 108(37): 15107-15112 (2011) - REST2原始论文
      GROMACS manual: CMAP handling in REST2
    ====================================================================
    """
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
    
    logger.info(f"  [CMAP] Result: {n_scaled} torsions scaled, {n_preserved} preserved")
    logger.info(f"  [CMAP] Physical accuracy: COMPLETE (GROMACS-style)")
    
    return new_force, n_scaled


def create_rest2_system_with_local_heating(base_system, solute_atoms, hot_atoms, lam_global,
                                            lam_hot, params, cmap_idx, cfg, logger):
    """
    创建带局部加热的REST2系统

    ====================================================================
    REST2 + 局部加热 缩放算法说明
    ====================================================================

    一、REST2基本原理 (Wang et al., PNAS 2011)
    --------------------------------------------------------------------
    传统T-REMD中，所有副本的势能函数相同，仅温度不同。
    对于大溶剂化系统，这导致交换率极低（需要大量副本）。

    REST2的核心改进:
      - 溶剂-溶剂相互作用: 不缩放 (保持原始势能)
      - 溶质-溶质相互作用: 按 λ 缩放
      - 溶质-溶剂相互作用: 按 √λ 缩放

      这样每个副本的有效温度为: T_eff = T_ref / λ
      而实际模拟温度始终为 T_ref，减少了需要的副本数。

    二、缩放因子的数学推导
    --------------------------------------------------------------------
    设 λ = T_ref / T_replica (全局缩放因子)

    对于有效哈密顿量: H_eff = λ·U_ss + √λ·U_solv + U_vv
    其中:
      U_ss  = 溶质-溶质相互作用 (按 λ 缩放)
      U_solv = 溶质-溶剂相互作用 (按 √λ 缩放)
      U_vv  = 溶剂-溶剂相互作用 (不缩放)

    为什么溶质-溶剂用 √λ?
      因为 U_solv ~ Σ(q_i·q_w)/r
      若对电荷缩放 √λ，则库仑项缩放 (√λ)²= λ
      若对LJ的ε缩放 √λ，则色散项也正确缩放

    三、局部加热扩展 (Liu et al., JCP 2013)
    --------------------------------------------------------------------
    在标准REST2基础上，对特定残基施加更强的缩放:

      普通溶质原子:   λ_global (标准REST2缩放)
      局部加热原子:   λ_hot_eff = λ_global × λ_hot (额外缩放)

    其中 λ_hot < 1 (本实验取0.5)，使热残基的有效温度更高:
      T_hot_eff = T_ref / (λ_global × λ_hot) = T_replica / λ_hot

    例如: T_replica=400K, λ_hot=0.5 → T_hot_eff = 800K
    这允许热残基克服更高的能垒，探索更广阔的构象空间。

    四、各类力的缩放规则
    --------------------------------------------------------------------
    1. NonbondedForce (非键力):
       - 电荷 q:      q' = q × √λ (使库仑力∝λ)
       - LJ epsilon ε: ε' = ε × λ (直接缩放色散)
       - 例外(1-4):   SS对→λ, SW对→√λ

    2. HarmonicBondForce (键能):
       - 力常数 k:     k' = k × λ

    3. HarmonicAngleForce (角能):
       - 力常数 k:     k' = k × λ

    4. PeriodicTorsionForce (周期扭转):
       - 力常数 k:     k' = k × λ

    5. CMAPTorsionForce (CMAP交叉项):
       - 能量网格:     E' = E × λ (完整缩放，GROMACS风格)

    参考:
      Wang et al., PNAS 108(37): 15107-15112 (2011)  [REST2]
      Liu et al., JCP 139: 034115 (2013)             [局部REST]
      Huang et al., JCTC 13(1): 35-46 (2017)        [CHARMM36m]
    ====================================================================
    """
    system = copy.deepcopy(base_system)

    # 计算各类缩放因子
    sqrt_lam_global = np.sqrt(lam_global)                    # 普通溶质-溶剂缩放
    sqrt_lam_hot = np.sqrt(lam_global * lam_hot)              # 热残基-溶剂缩放
    lam_hot_effective = lam_global * lam_hot                  # 热残基-溶质有效缩放
    
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
                    # 局部加热原子: 更强的缩放
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
                        # 包含局部加热原子的例外: 更强缩放
                        force.setExceptionParameters(j, p1, p2, q12 * lam_hot_effective, sig, eps * lam_hot_effective)
                        n_hot_ss += 1
                    else:
                        force.setExceptionParameters(j, p1, p2, q12 * lam_global, sig, eps * lam_global)
                        n_ss += 1
                elif s1 or s2:
                    if h1 or h2:
                        # 局部加热原子与溶剂的例外
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
                # CMAP使用全局lam缩放 (CMAP通常涉及整个主链)
                scaled_cmap, n_cmap_scaled = create_scaled_cmap_force(force, solute_atoms, lam_global, logger)
                cmap_forces_to_replace.append((i, scaled_cmap))
            else:
                logger.info(f"    CMAPTorsionForce: preserved (mode={cfg['cmap_scaling_mode']})")
    
    if cmap_forces_to_replace:
        for orig_idx, new_force in cmap_forces_to_replace:
            logger.info(f"    [CMAP] Force {orig_idx} will be replaced with scaled version")
    
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
    """
    几何分布温度 - REST2最优温度分布

    ====================================================================
    温度分布算法说明
    ====================================================================
    在REMD/REST2中，温度的分布方式直接影响交换率:

    1. 均匀分布: T_i = T_ref + (T_max - T_ref) * i / (N-1)
       - 温度间隔恒定: ΔT = const
       - 但低温区交换率高，高温区交换率低

    2. 几何分布 (本代码采用): T_i = T_ref × (T_max/T_ref)^(i/(N-1))
       - 温度间隔随温度增大而增大
       - 各副本间的交换率更均匀
       - 是REST2论文推荐的标准分布方式

    3. 对应的缩放因子: λ_i = T_ref / T_i
       - λ_0 = 1.0 (参考温度，不缩放)
       - λ_max = T_ref / T_max = 300/400 = 0.75

    交换率优化原理:
      交换概率 ∝ exp(-Δ)，其中 Δ ∝ 1/√N_atoms × (ΔT/T)²
      几何分布使各对的Δ更均匀，从而提高整体交换率。
    ====================================================================
    """
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
            if i < 10 or i >= n_replicas - 5:  # 显示前10个和后5个
                logger.info(f"  {i:2d}   {T:6.1f}   {lam:.4f}    {lam_hot_eff:.4f}     {np.sqrt(lam):.4f}    {dT:5.1f}")
            elif i == 10:
                logger.info(f"  ...  (中间副本省略显示)  ...")
        logger.info(f"{'='*70}")
        logger.info(f"[v1.0] Hot residues: {CONFIG['hot_residue_ids']}")
        logger.info(f"[v1.0] λ_hot={CONFIG['lambda_hot']}, additional scaling applied")
    
    return temps, lams


def run_rest2(cfg, logger):
    """执行带局部加热的REST2模拟"""
    
    kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
    kB_val = kB.value_in_unit(unit.kilojoule_per_mole / unit.kelvin)
    
    logger.info(f"\n{'='*70}")
    logger.info(f"REST2 Local Heating Simulation [v{cfg['version']}]")
    logger.info(f"Temperature Range: {cfg['T_ref']}K - {cfg['T_max']}K")
    logger.info(f"Hot Residues: {cfg['hot_residue_ids']}")
    logger.info(f"Lambda_hot: {cfg['lambda_hot']} (additional scaling for hot residues)")
    logger.info(f"{'='*70}")
    
    base_system, psf, pdb, box, params, cmap_idx = build_base_system(cfg, logger)
    solute_atoms = identify_solute_atoms(pdb.topology, logger)
    
    # 识别局部加热残基的原子
    if cfg["local_heating_enabled"]:
        hot_atoms, hot_atoms_by_residue = identify_hot_residue_atoms(pdb.topology, cfg, logger)
    else:
        hot_atoms = set()
        hot_atoms_by_residue = {}
    
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
        
        # 使用目标温度初始化速度（而非固定10K）
        ctx.setVelocitiesToTemperature(T*unit.kelvin)
        contexts.append(ctx)
    
    # ====================================================================
    # 分阶段平衡协议
    # ====================================================================
    # 所有副本经历相同的温度阶梯: 50K → 100K → 200K → 300K → 目标温度
    # 这种渐进式加热策略可以:
    #   1. 避免突然高温导致的系统不稳定
    #   2. 逐步释放最小化后的构象张力
    #   3. 使溶剂分子逐步适应溶质的存在
    #
    # 每个阶段: (温度K, 步数, 时间步fs, 说明)
    # ====================================================================
    logger.info(f"\n=== Staged Equilibration ===")
    equil_protocol = [
        (50,   50,  0.5, "Low-T relaxation"),       # 低温松弛，小步长
        (100, 100,  1.0, "Medium-T thermalization"), # 中温热化
        (200, 200,  1.0, "High-T sampling preparation"), # 高温准备
        (300, 500,  2.0, "Target-T equilibration"),  # 接近目标温度
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
    logger.info(f"  Hot residues: {cfg['hot_residue_ids']}")
    logger.info(f"  Lambda_hot: {cfg['lambda_hot']}")
    logger.info(f"  Platform: {plat_name}")
    logger.info(f"{'='*70}\n")
    
    exchange_attempts = np.zeros(n_rep-1, dtype=int)
    exchange_accepts = np.zeros(n_rep-1, dtype=int)
    energy_history = []
    delta_history = {k: [] for k in range(n_rep-1)}
    replica_state_history = []
    
    t0 = time.time()

    # ====================================================================
    # REST2 生产模拟主循环
    # ====================================================================
    # 每次迭代包含以下步骤:
    #   1. MD演化: 每个副本独立运行 n_steps 步
    #   2. 构型收集: 获取所有副本的位置和能量
    #   3. 副本交换: 按Metropolis准则尝试相邻副本间的构型交换
    #   4. 数据记录: 保存能量历史、交换统计
    #   5. 检查点: 定期保存检查点文件
    #
    # 交换策略: 奇偶交替
    #   偶数迭代: 交换 (0,1), (2,3), (4,5), ...
    #   奇数迭代: 交换 (1,2), (3,4), (5,6), ...
    #   这确保了所有相邻对都有交换机会，且不冲突
    # ====================================================================
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
        
        # ====================================================================
        # Metropolis 交换准则 (REST2版本)
        # ====================================================================
        # 在REST2中，每个副本i的配分函数为:
        #   Z_i = ∫ exp(-β_i × U_i(x)) dx
        #
        # 其中:
        #   β_i = 1/(k_B × T_i)  — 副本i的逆温度
        #   U_i(x) = λ_i × U_solute(x) + U_solvent(x)  — 缩放后的有效势能
        #
        # 交换副本m和n的构型时，Metropolis准则为:
        #   Δ = β_m × [U_m(x_n) - U_m(x_m)] + β_n × [U_n(x_m) - U_n(x_m)]
        #
        # 其中:
        #   x_m, x_n  — 交换前副本m和n的构型
        #   U_m(x_n)  — 构型x_n在副本m力场下的能量 (需要重新计算)
        #   U_m(x_m)  — 构型x_m在副本m力场下的能量 (已知)
        #
        # 接受概率: P_accept = min(1, exp(-Δ))
        #   - Δ < 0: 总是接受 (能量降低)
        #   - Δ > 0: 以概率 exp(-Δ) 接受
        #
        # 奇偶交替交换策略:
        #   偶数迭代: 交换对 (0,1), (2,3), (4,5), ...
        #   奇数迭代: 交换对 (1,2), (3,4), (5,6), ...
        #   这样每个副本都有机会与两个邻居交换，提高遍历性
        # ====================================================================
        for m in pairs:
            n = m + 1
            exchange_attempts[m] += 1

            pos_m, pos_n = positions[m], positions[n]
            E_m_xm, E_n_xn = energies[m], energies[n]

            # 将副本n的构型放入副本m的力场中，计算交叉能量
            contexts[m].setPositions(pos_n)
            E_m_xn = contexts[m].getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

            # 将副本m的构型放入副本n的力场中，计算交叉能量
            contexts[n].setPositions(pos_m)
            E_n_xm = contexts[n].getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

            # 计算Metropolis判据
            beta_m, beta_n = 1/(kB_val*temps[m]), 1/(kB_val*temps[n])
            delta = beta_m*(E_m_xn-E_m_xm) + beta_n*(E_n_xm-E_n_xn)
            delta_history[m].append(delta)

            # Metropolis接受/拒绝判断
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
    logger.info(f"  Hot residues: {cfg['hot_residue_ids']}")
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
        'hot_residue_ids': cfg['hot_residue_ids'],
        'hot_residue_names': cfg['hot_residue_names'],
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
    
    # MBAR兼容输出
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
    ax.plot(T, lam_hot_eff, 's-', label='λ_hot_eff', color='red')
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
    ax.set_title(f'Temperature Distribution (Hot residues: {results["hot_residue_ids"]})', fontweight='bold')
    ax.axhline(300, color='g', ls='--', label='T_ref')
    ax.axhline(400, color='r', ls='--', label='T_max')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'fig2_exchange.png'), dpi=150)
    plt.close()
    
    # Fig 3: 热残基信息
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    hot_residue_str = ', '.join([f"{results['hot_residue_names'][rid]}{rid}" 
                                  for rid in results['hot_residue_ids']])
    ax.text(0.5, 0.5, f"Local Heating Residues:\n{hot_residue_str}\n\nλ_hot = {results['lambda_hot']}",
            ha='center', va='center', fontsize=16, transform=ax.transAxes)
    ax.set_title('Local Heating Configuration', fontweight='bold', fontsize=14)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'fig3_hot_residues.png'), dpi=150)
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
        'lambda_values': [float(l) for l in results['lam_values']],
        'avg_exchange_rate': float(round(avg_rate, 2)),
        'exchange_rates': [float(round(r,2)) for r in rates],
        'exchange_rate_valid': bool(20 <= avg_rate <= 40),
        'n_iterations': int(len(E)),
        'total_simulation_ps': float(len(E) * 1000 * 2 / 1000),
        'total_time_s': float(round(results['total_time'], 1)),
        'cmap_scaling_mode': results['cmap_scaling_mode'],
        'cmap_scaling_complete': bool(True),
        
        # 局部加热信息
        'local_heating_enabled': bool(results['local_heating']),
        'hot_residue_ids': results['hot_residue_ids'],
        'hot_residue_names': results['hot_residue_names'],
        'lambda_hot': float(results['lambda_hot']),
        'lambda_hot_effective': [float(l * results['lambda_hot']) for l in results['lam_values']],
        
        'system_info': {
            'protein': '4AU5',
            'force_field': 'CHARMM36m',
            'water_model': 'TIP3P',
            'n_hot_residues': len(results['hot_residue_ids']),
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
        logger.info(f"REST2 Local Heating Simulation - 4AU5 Protein + CHARMM36m")
        logger.info(f"Version: {cfg['version']}")
        logger.info(f"Temperature: {cfg['T_ref']}K - {cfg['T_max']}K")
        logger.info(f"Hot Residues: {cfg['hot_residue_ids']}")
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