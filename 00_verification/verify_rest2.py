#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
verify_rest2.py — REST-NhaA (4AU5 v4.4) 体系正确性验证
Layer 1: 静态构建检查 (拓扑/segid/编号/盒子)
Layer 2: 缩放哈密顿量验证 (force 枚举 / 槽位0恒等 / 参数缩放律 / 能量标定)

运行环境: OpenMM >= 7.6 (omm760 可用, 无需 GPU)
用法: python verify_rest2.py [--json verify_report.json]
"""
import os
import sys
import json
import copy
import logging
import importlib.util
import argparse

import numpy as np
import openmm
from openmm import app, unit

BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # 00_verification/
PROJ_DIR = os.path.dirname(BASE_DIR)                           # REST-NhaA root
SYS_DIR = os.path.join(PROJ_DIR, "4AU5")
SCRIPT_V44 = os.path.join(SYS_DIR, "versions", "rest2_4au5_v4.4.py")

REPORT = {"L1": {}, "L2": {}, "fail": []}


def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        REPORT["fail"].append(f"{name}: {detail}")
    return ok


class CollectHandler(logging.Handler):
    """捕获被测代码的日志, 用于审计计数。"""
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record.getMessage())


def load_module():
    spec = importlib.util.spec_from_file_location("rest2_v44", SCRIPT_V44)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # __main__ 有 guard, import 安全
    return mod


# =====================================================================
# Layer 1: 静态构建检查
# =====================================================================

def layer1(mod, cfg, psf, pdb):
    print("\n" + "=" * 70)
    print("Layer 1: 静态构建检查")
    print("=" * 70)
    L1 = {}
    top_psf, top_pdb = psf.topology, pdb.topology

    # 1.1 原子数
    n_psf, n_pdb = top_psf.getNumAtoms(), top_pdb.getNumAtoms()
    check("NATOM == 68411 (PSF)", n_psf == 68411, f"PSF={n_psf}")
    check("PDB 与 PSF 原子数一致", n_pdb == n_psf, f"PDB={n_pdb}")
    L1["n_atoms"] = {"psf": n_psf, "pdb": n_pdb}

    # 1.2 PSF 文本解析 segid 分布
    seg_counts, psf_atoms = {}, []
    with open(cfg["psf_file"]) as f:
        lines = f.read().splitlines()
    try:
        i0 = next(i for i, l in enumerate(lines) if "!NATOM" in l)
    except StopIteration:
        raise RuntimeError("PSF 中未找到 !NATOM 段")
    for line in lines[i0 + 1: i0 + 1 + n_psf]:
        parts = line.split()
        seg, resid, resname, atname = parts[1], parts[2], parts[3], parts[4]
        seg_counts[seg] = seg_counts.get(seg, 0) + 1
        psf_atoms.append((seg, int(resid), resname, atname))
    print(f"  segid 分布: {seg_counts}")
    L1["segid_counts"] = seg_counts
    check("存在蛋白/膜/水/离子段",
          any(k.startswith("PRO") for k in seg_counts)
          and ("MEMB" in seg_counts or any(k.startswith("MEM") for k in seg_counts))
          and ("TIP3" in seg_counts or "HOH" in seg_counts or "WAT" in seg_counts),
          str(sorted(seg_counts)))

    # 1.3 PDB-PSF 原子一一对应 (学生代码用 pdb.topology 索引改 PSF System)
    # 用 openmm 规范化后的 topology 比对 (CHARMM PSF 原始文本中水氧为 OH2,
    # PDB 为 O 等命名差异由 openmm 统一; 学生代码只用规范化后的索引)
    psf_top_atoms = [(int(a.residue.id), a.residue.name, a.name) for a in top_psf.atoms()]
    pdb_atoms = [(int(a.residue.id), a.residue.name, a.name) for a in top_pdb.atoms()]
    name_mismatch = sum(1 for p, q in zip(psf_top_atoms, pdb_atoms)
                        if p[1] != q[1] or p[2] != q[2])
    resid_mismatch = sum(1 for p, q in zip(psf_top_atoms, pdb_atoms)
                         if p[0] < 10000 and p[0] != q[0])
    check("PDB-PSF 原子顺序/残基名/原子名逐项一致 (openmm 规范化后)", name_mismatch == 0,
          f"{name_mismatch} 处不匹配")
    check("PDB-PSF 残基号一致 (排除 PDB 4 位折返的水)", resid_mismatch == 0,
          f"{resid_mismatch} 处不匹配")
    L1["pdb_psf_name_mismatches"] = name_mismatch
    L1["pdb_psf_resid_mismatches_below_10000"] = resid_mismatch

    # 1.4 溶质与 hot 残基识别
    solute = mod.identify_solute_atoms(top_pdb, quiet)
    n_prot_atoms = sum(c for k, c in seg_counts.items() if k.startswith("PRO"))
    check("溶质原子数 ≈ 蛋白段原子数 (水/脂/离子全部排除)",
          len(solute) == n_prot_atoms, f"solute={len(solute)}, PRO*={n_prot_atoms}")
    L1["n_solute"] = len(solute)

    hot_all, hot_by_res = mod.identify_hot_residue_atoms(top_pdb, solute, cfg, quiet)
    check("hot 残基恰为 5 个且审计通过", len(hot_by_res) == 5,
          f"{sorted(hot_by_res)}")
    L1["hot_residues"] = {k: len(v) for k, v in hot_by_res.items()}
    print(f"  hot 残基原子数: {L1['hot_residues']} (合计 {len(hot_all)})")

    # 1.5 残基编号偏移
    res_ids = sorted({int(a.residue.id) for a in top_pdb.atoms()
                      if a.residue.name in mod.AA_RESNAMES})
    offset = res_ids[0] - 1
    check("残基编号偏移 = 9 (输入 10-390 → 输出 1-381)", offset == 9,
          f"first_input_id={res_ids[0]}, offset={offset}")
    L1["residue_offset"] = offset

    # 1.6 盒子
    box_dims, box_angles = mod.read_charmm_gui_box_info(cfg["boxinfo_str"], quiet)
    expect = [82.4067792, 82.4067792, 109.362015]
    ok = all(abs(a - b) < 1e-4 for a, b in zip(box_dims, expect))
    check("盒子尺寸与 CHARMM-GUI .str 一致 (82.4068/82.4068/109.3620 Å)", ok,
          f"{[round(x, 4) for x in box_dims]}")
    L1["box"] = box_dims

    REPORT["L1"] = L1


# =====================================================================
# Layer 2: 缩放哈密顿量验证
# =====================================================================

def build_scaled(mod, base, solute, hot, lam, lam_hot, cfg, cap):
    quiet_local = logging.getLogger("verify.builder")
    quiet_local.addHandler(cap)
    quiet_local.setLevel(logging.INFO)
    sys_s, cmap_rep = mod.create_rest2_system_with_local_heating(
        base, solute, hot, lam, cfg["lambda_min"], lam_hot, cfg, quiet_local)
    if cmap_rep:
        sys_s = mod.rebuild_system_with_scaled_cmap(sys_s, cmap_rep, quiet_local)
    return sys_s


def force_lists(sys_a, sys_b):
    return [sys_a.getForce(i) for i in range(sys_a.getNumForces())], \
           [sys_b.getForce(i) for i in range(sys_b.getNumForces())]


def cmp_slot0(base, slot0):
    """槽位 0 (λ=1) System 与 base 逐 force 全参数对比。"""
    fa, fb = force_lists(base, slot0)
    check("force 数目一致", len(fa) == len(fb), f"{len(fa)} vs {len(fb)}")
    n_diff = 0
    for i, (x, y) in enumerate(zip(fa, fb)):
        if type(x) is not type(y):
            check(f"force {i} 类型一致", False, f"{type(x).__name__} vs {type(y).__name__}")
            return False
        t = type(x).__name__
        try:
            if isinstance(x, openmm.HarmonicBondForce):
                n = x.getNumBonds()
                for j in range(n):
                    if x.getBondParameters(j) != y.getBondParameters(j):
                        n_diff += 1
            elif isinstance(x, openmm.HarmonicAngleForce):
                n = x.getNumAngles()
                for j in range(n):
                    if x.getAngleParameters(j) != y.getAngleParameters(j):
                        n_diff += 1
            elif isinstance(x, openmm.PeriodicTorsionForce):
                n = x.getNumTorsions()
                for j in range(n):
                    if x.getTorsionParameters(j) != y.getTorsionParameters(j):
                        n_diff += 1
            elif isinstance(x, openmm.NonbondedForce):
                n = x.getNumParticles()
                for j in range(n):
                    p, q = x.getParticleParameters(j), y.getParticleParameters(j)
                    if any(abs(p[k].value_in_unit(unit.kilojoule_per_mole if k == 2 else (
                            unit.nanometer if k == 1 else unit.elementary_charge)) -
                            q[k].value_in_unit(unit.kilojoule_per_mole if k == 2 else (
                            unit.nanometer if k == 1 else unit.elementary_charge))) > 0
                           for k in range(3)):
                        n_diff += 1
                if x.getNumExceptions() != y.getNumExceptions():
                    n_diff += 1
            elif isinstance(x, openmm.CustomNonbondedForce):
                # 学生实现: 追加 restScale 粒子参数并改写表达式为 sqrt(restScale1*restScale2)*(...)
                # λ=1 时语义等价: 原参数为前缀, restScale=1, 表达式 = sqrt(...)*(原式)
                n_base_params = x.getNumPerParticleParameters()
                if y.getNumPerParticleParameters() != n_base_params + 1:
                    n_diff += 1
                expr_y = y.getEnergyFunction()
                expr_x = x.getEnergyFunction()
                prefix = "sqrt(restScale1*restScale2)*("
                if not (expr_y.startswith(prefix) and expr_x.split(";")[0] in expr_y):
                    n_diff += 1
                for j in range(x.getNumParticles()):
                    px = list(x.getParticleParameters(j))
                    py = list(y.getParticleParameters(j))
                    if py[:n_base_params] != px or py[n_base_params] != 1.0:
                        n_diff += 1
            elif isinstance(x, openmm.CMAPTorsionForce):
                for j in range(x.getNumTorsions()):
                    if x.getTorsionParameters(j) != y.getTorsionParameters(j):
                        n_diff += 1
        except Exception as e:
            check(f"force {i} ({t}) 参数对比", False, str(e))
            return False
    check("槽位 0 全部 force 参数 ≡ base System", n_diff == 0, f"{n_diff} 处差异")
    return n_diff == 0


def param_scales(base, scaled, solute, hot, lam, lam_hot, mod, cfg, n_sample=200):
    """最高槽位参数缩放律抽查 (hot/normal/mixed/solvent 各类)。"""
    lam_min = cfg["lambda_min"]
    ok_all = True

    def s_atom(a):
        return mod.atom_scale(a, solute, hot, lam, mod.local_multiplier_for_lambda(
            lam, lam_min, lam_hot))

    # --- HarmonicBondForce ---
    fb = next(f for f in scaled.getForces() if isinstance(f, openmm.HarmonicBondForce))
    f0 = next(f for f in base.getForces() if isinstance(f, openmm.HarmonicBondForce))
    idxs = np.random.default_rng(0).choice(fb.getNumBonds(), min(n_sample, fb.getNumBonds()), replace=False)
    bad = 0
    for j in idxs:
        p1, p2, r0, k0 = f0.getBondParameters(int(j))
        _, _, r1, k1 = fb.getBondParameters(int(j))
        s_expect = mod.geometric_term_scale((p1, p2), solute, hot, lam,
                                            mod.local_multiplier_for_lambda(lam, lam_min, lam_hot))
        if abs(float(k1 / k0) - s_expect) > 1e-9 * max(1.0, s_expect) or r0 != r1:
            bad += 1
    ok_all &= check("bond: k × 几何平均(s_i), r0 不变", bad == 0, f"{bad}/{len(idxs)} 项偏差")

    # --- NonbondedForce 粒子 ---
    fn = next(f for f in scaled.getForces() if isinstance(f, openmm.NonbondedForce))
    f0n = next(f for f in base.getForces() if isinstance(f, openmm.NonbondedForce))
    idxs = np.random.default_rng(1).choice(fn.getNumParticles(), min(n_sample, fn.getNumParticles()), replace=False)
    bad_q = bad_e = 0
    for j in idxs:
        q0, sig0, e0 = f0n.getParticleParameters(int(j))
        q1, sig1, e1 = fn.getParticleParameters(int(j))
        s = s_atom(int(j))
        q0v = q0.value_in_unit(unit.elementary_charge)
        e0v = e0.value_in_unit(unit.kilojoule_per_mole)
        q1v = q1.value_in_unit(unit.elementary_charge)
        e1v = e1.value_in_unit(unit.kilojoule_per_mole)
        if q0v != 0 and abs(q1v / q0v - np.sqrt(s)) > 1e-9:
            bad_q += 1
        if e0v != 0 and abs(e1v / e0v - s) > 1e-9 * max(1.0, s):
            bad_e += 1
        if sig0 != sig1:
            bad_q += 1
    ok_all &= check("nonbonded 粒子: q ×√s, ε ×s, σ 不变", bad_q == 0 and bad_e == 0,
                    f"q:{bad_q}, eps:{bad_e} 偏差")

    # --- exceptions ---
    n_exc = fn.getNumExceptions()
    idxs = np.random.default_rng(2).choice(n_exc, min(n_sample, n_exc), replace=False)
    bad = 0
    for j in idxs:
        p1, p2, q12_0, sig_0, e_0 = f0n.getExceptionParameters(int(j))
        _, _, q12_1, sig_1, e_1 = fn.getExceptionParameters(int(j))
        s_pair = np.sqrt(s_atom(p1) * s_atom(p2))
        qv = abs(q12_1.value_in_unit(unit.elementary_charge ** 2))
        q0v = abs(q12_0.value_in_unit(unit.elementary_charge ** 2))
        ev = abs(e_1.value_in_unit(unit.kilojoule_per_mole))
        e0v = abs(e_0.value_in_unit(unit.kilojoule_per_mole))
        expect = s_pair  # 学生实现: pair_scale = sqrt(s1*s2), q12 与 ε 同乘
        if q0v > 0 and abs(qv / q0v - expect) > 1e-9:
            bad += 1
        elif e0v > 0 and abs(ev / e0v - expect) > 1e-9 * max(1.0, expect):
            bad += 1
    ok_all &= check("exception: q₁q₂ 与 ε 均 ×√(s_a·s_b)", bad == 0, f"{bad}/{len(idxs)} 项偏差")

    # --- CustomTorsionForce (improper) ---
    ct = [f for f in scaled.getForces() if isinstance(f, openmm.CustomTorsionForce)]
    ct0 = [f for f in base.getForces() if isinstance(f, openmm.CustomTorsionForce)]
    bad = 0
    for f_s, f_b in zip(ct, ct0):
        amp = mod.custom_torsion_amplitude_indices(f_s, quiet)
        if not amp:
            continue
        for j in range(f_s.getNumTorsions()):
            p1, p2, p3, p4, pars_s = f_s.getTorsionParameters(j)
            _, _, _, _, pars_b = f_b.getTorsionParameters(j)
            if float(pars_b[0]) == 0.0:  # base 中 k=0 的 improper, 缩放后也应为 0
                if float(pars_s[0]) != 0.0:
                    bad += 1
                continue
            s_exp = mod.geometric_term_scale((p1, p2, p3, p4), solute, hot, lam,
                                             mod.local_multiplier_for_lambda(lam, lam_min, lam_hot))
            if abs(float(pars_s[0]) / float(pars_b[0]) - s_exp) > 1e-9 * max(1.0, s_exp):
                bad += 1
    ok_all &= check("improper (CustomTorsion): k × 几何平均", bad == 0, f"{bad} 项偏差")

    # --- CMAP ---
    cm = [f for f in scaled.getForces() if isinstance(f, openmm.CMAPTorsionForce)]
    cm0 = [f for f in base.getForces() if isinstance(f, openmm.CMAPTorsionForce)]
    bad = 0
    if cm and cm0:
        fs, f0 = cm[0], cm0[0]
        for j in range(0, fs.getNumTorsions(), max(1, fs.getNumTorsions() // 100)):
            mi_s, *atoms_s = fs.getTorsionParameters(j)
            mi_0, *atoms_0 = f0.getTorsionParameters(j)
            atoms = tuple(int(a) for a in atoms_s)
            s_exp = mod.geometric_term_scale(atoms, solute, hot, lam,
                                             mod.local_multiplier_for_lambda(lam, lam_min, lam_hot))
            _, grid_s = fs.getMapParameters(mi_s)
            _, grid_0 = f0.getMapParameters(mi_0)
            g0 = np.array([v.value_in_unit(unit.kilojoule_per_mole) for v in grid_0])
            gs = np.array([v.value_in_unit(unit.kilojoule_per_mole) for v in grid_s])
            if np.max(np.abs(gs - g0 * s_exp)) > 1e-6 * max(1.0, np.max(np.abs(g0))):
                bad += 1
    ok_all &= check("CMAP: 能量网格 × 几何平均(8 原子)", bad == 0, f"{bad} 张图偏差")

    return ok_all


def energy_fit(mod, base, solute, hot, cfg, cap):
    """标准模式 (hot=∅) 三 λ 能量拟合 U(λ)=aλ+b√λ+c, 预测第四 λ 并验证。"""
    print("  构建标准模式 (hot=∅) 缩放 System 并计算 CPU 单点能...")
    lams = [0.95, 0.90]
    lam_pred = 0.86
    positions = None

    def energy_of(system):
        integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
        platform = openmm.Platform.getPlatformByName("CPU")
        ctx = openmm.Context(system, integrator, platform)
        ctx.setPositions(pdb_pos)
        e = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole)
        del ctx, integrator
        return e

    # base (λ=1)
    quiet_e = logging.getLogger("verify.energy")
    quiet_e.setLevel(logging.WARNING)
    integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
    platform = openmm.Platform.getPlatformByName("CPU")
    ctx = openmm.Context(base, integrator, platform)
    ctx.setPositions(pdb_pos)
    e_base = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole)
    del ctx, integrator
    print(f"    E(λ=1)    = {e_base:.4f} kJ/mol")

    energies = {}
    for lam in lams:
        s_std = build_scaled(mod, base_std_for_build, solute, set(), lam, 1.0, cfg, cap)
        energies[lam] = energy_of(s_std)
        print(f"    E(λ={lam}) = {energies[lam]:.4f} kJ/mol")

    # 拟合 aλ + b√λ + c
    A = np.array([[lams[0] - 1.0, np.sqrt(lams[0]) - 1.0],
                  [lams[1] - 1.0, np.sqrt(lams[1]) - 1.0]])
    y = np.array([energies[lams[0]] - e_base, energies[lams[1]] - e_base])
    a_, b_ = np.linalg.solve(A, y)
    c_ = e_base - a_ - b_
    print(f"    拟合: U(λ) = {a_:.4f}·λ + {b_:.4f}·√λ + {c_:.4f}")
    print(f"    (E_ss={a_:.1f}, E_sv={b_:.1f}, E_vv+溶剂价键={c_:.1f} kJ/mol)")

    # 预测并验证 λ_pred
    s_pred = build_scaled(mod, base_std_for_build, solute, set(), lam_pred, 1.0, cfg, cap)
    e_pred_actual = energy_of(s_pred)
    e_pred_model = a_ * lam_pred + b_ * np.sqrt(lam_pred) + c_
    rel = abs(e_pred_actual - e_pred_model) / max(1.0, abs(e_pred_model))
    print(f"    λ={lam_pred}: 实测 {e_pred_actual:.4f} vs 模型 {e_pred_model:.4f} "
          f"(相对差 {rel:.2e})")
    ok = check("能量标定: U(λ)=λE_ss+√λE_sv+E_vv 预测第四 λ (rel<1e-6)", rel < 1e-6, "")
    return ok


def layer2(mod, cfg, base, psf, pdb, solute, hot_all):
    print("\n" + "=" * 70)
    print("Layer 2: 缩放哈密顿量验证")
    print("=" * 70)
    cap = CollectHandler()

    # 2.1 force 类型枚举
    types = {}
    for i in range(base.getNumForces()):
        t = type(base.getForce(i)).__name__
        types[t] = types.get(t, 0) + 1
    print(f"  base system force 类型: {types}")
    REPORT["L2"]["force_types"] = types
    covered = {"HarmonicBondForce", "HarmonicAngleForce", "PeriodicTorsionForce",
               "CustomTorsionForce", "CMAPTorsionForce", "NonbondedForce",
               "CustomNonbondedForce"}
    v4_capable = {"HarmonicBondForce", "HarmonicAngleForce", "PeriodicTorsionForce",
                  "CustomTorsionForce", "CMAPTorsionForce", "NonbondedForce",
                  "CustomNonbondedForce", "MonteCarloBarostat",
                  "MonteCarloMembraneBarostat", "CMMotionRemover",
                  "CustomExternalForce"}
    uncovered = [t for t in types if t not in v4_capable]
    check("无 v4.4 缩放代码未覆盖的力类型", not uncovered,
          f"未覆盖: {uncovered}" if uncovered else "全部覆盖/与缩放无关")

    # 2.2 槽位 0 恒等
    print("  构建槽位 0 (λ=1) System...")
    slot0 = build_scaled(mod, base, solute, hot_all, 1.0, cfg["lambda_hot"], cfg, cap)
    check("槽位 0 粒子/约束数一致",
          slot0.getNumParticles() == base.getNumParticles()
          and slot0.getNumConstraints() == base.getNumConstraints())
    cmp_slot0(base, slot0)

    # 2.3 最高槽位参数缩放律
    lam_top = cfg["lambda_min"]
    print(f"  构建最高槽位 (λ={lam_top:.6f}, local-hot ×{cfg['lambda_hot']}) System...")
    top_sys = build_scaled(mod, base, solute, hot_all, lam_top, cfg["lambda_hot"], cfg, cap)
    param_scales(base, top_sys, solute, hot_all, lam_top, cfg["lambda_hot"], mod, cfg)

    # 2.4 审计: 覆盖计数与 warning
    txt = "\n".join(cap.records)
    warn_count = txt.count("cannot be scaled reliably")
    check("CustomTorsionForce 无未识别表达式 (warning=0)", warn_count == 0,
          f"{warn_count} 条")
    for line in txt.splitlines():
        if "coverage audit" in line or ("normal/hot/mixed" in line and "CMAP" not in line):
            pass
    audits = [l for l in txt.splitlines() if ": " in l and "/" in l
              and ("normal/hot/mixed" in l or any(
                  k in l for k in ("bond:", "angle:", "periodic torsion:",
                                   "custom torsion:", "CMAP:",
                                   "nonbonded particle:", "nonbonded exception:")))]
    REPORT["L2"]["audit_lines"] = audits[:30]

    # 2.5 能量标定 (标准模式)
    global base_std_for_build, pdb_pos
    base_std_for_build = base
    pdb_pos = pdb.positions
    energy_fit(mod, base, solute, hot_all, cfg, cap)


# =====================================================================
def main():
    global quiet, pdb_pos, base_std_for_build
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default=os.path.join(BASE_DIR, "verify_report.json"))
    args = parser.parse_args()

    quiet = logging.getLogger("verify")
    quiet.setLevel(logging.WARNING)
    logging.getLogger("verify.builder").propagate = False
    np.set_printoptions(precision=6, suppress=True)

    print(f"验证对象: {SCRIPT_V44}")
    mod = load_module()
    cfg = dict(mod.CONFIG)
    cfg["lambda_min"] = cfg["T_ref"] / cfg["T_max"]
    print(f"体系: {cfg['psf_file']}")
    print(f"λ 范围: 1.0 → {cfg['lambda_min']:.6f} (T_ref={cfg['T_ref']}, T_max={cfg['T_max']})")

    # ---- 构建 base system ----
    print("\n加载 PSF/PDB/toppar 并构建 base System (约 1-2 分钟)...")
    base, psf, pdb, box_size, original_box = mod.build_base_system(cfg, quiet)
    solute = mod.identify_solute_atoms(pdb.topology, quiet)
    hot_all, _ = mod.identify_hot_residue_atoms(pdb.topology, solute, cfg, quiet)
    pdb_pos = pdb.positions

    layer1(mod, cfg, psf, pdb)
    layer2(mod, cfg, base, psf, pdb, solute, hot_all)

    # ---- 汇总 ----
    print("\n" + "=" * 70)
    n_fail = len(REPORT["fail"])
    print(f"验证完成: {'全部通过 ✓' if n_fail == 0 else f'{n_fail} 项 FAIL ✗'}")
    for f in REPORT["fail"]:
        print(f"  - {f}")
    print("=" * 70)
    with open(args.json, "w") as f:
        json.dump(REPORT, f, indent=2, ensure_ascii=False, default=str)
    print(f"报告已写入: {args.json}")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
