#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
l4_traj_analysis.py — L4 轨迹层分析: 16 槽位 OF-gate Rg / RMSD 温度响应
输入: outputs/replica_stride100/replica_{0..15}_stride100.dcd (槽位轨迹, stride=100 迭代)
拓扑: charmm-gui/4au5-aa-k300n/openmm/step5_charmm2omm.psf (保留 segid/resid 原编号)
输出: l4_rg_results.json + l4_rg_plot.png/pdf
"""
import json
import os
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import align

BASE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(BASE)
SYS = os.path.join(PROJ, "4AU5")
PSF = os.path.join(SYS, "charmm-gui", "4au5-aa-k300n", "openmm", "step5_charmm2omm.psf")
DCD_DIR = os.path.join(SYS, "outputs", "replica_stride100")

# MDAnalysis 2.0.0 的 "type H*" 通配符选择会段错误 (numpy in1d bug) — 用 numpy 手工过滤等价口径
RES_SEL = "segid PROA and (resid 344 or resid 345 or resid 167 or resid 168 or resid 72)"
BB_NAMES = {"N", "CA", "C", "O", "OC1", "OC2", "OT1", "OT2", "OXT"}
CA_SEL = "protein and name CA"
T_REF, T_MAX, LAMBDA_HOT = 300.0, 350.0, 1.0  # 数据来自 v4.3 标准模式 (用户确认), 无 local-hot
SKIP_SLOTS = [0]  # replica_0 在微信传输源头损坏 (bad CRC), 待学生重传后补算


def heavy_sidechain_mask(ag):
    names = np.array([str(n) for n in ag.names])
    return np.array([n not in BB_NAMES and not n.startswith("H") for n in names])


results = {"slots": [], "gate_sel": RES_SEL + " + 侧链重原子(numpy过滤)", "skipped_slots": SKIP_SLOTS}
for rep in range(16):
    if rep in SKIP_SLOTS:
        print(f"slot {rep}: 跳过 (DCD 源头损坏, 待重传)")
        continue
    dcd = os.path.join(DCD_DIR, f"replica_{rep}_stride100.dcd")
    u = mda.Universe(PSF, dcd)
    res = u.select_atoms(RES_SEL)
    gate = res[heavy_sidechain_mask(res)]
    ca = u.select_atoms(CA_SEL)
    n_frames = len(u.trajectory)
    if rep == 1:
        print(f"每槽位帧数: {n_frames}")
    rg_list, rmsd_list = [], []
    ref_ca = ca.positions.copy()
    ref_ca_com = ref_ca.mean(0)
    for ts in u.trajectory:
        if np.isnan(u.atoms.positions).any():
            print(f"  slot {rep} frame {ts.frame}: NaN!")
            continue
        rg_list.append(gate.radius_of_gyration())  # MDAnalysis 坐标为 Å, 直接用
        # Kabsch 对齐到首帧再算 CA RMSD (消除平移+旋转漂移)
        cur = ca.positions - ca.center_of_mass()
        ref = ref_ca - ref_ca_com
        U, S, Vt = np.linalg.svd(cur.T @ ref)
        d = np.sign(np.linalg.det(U @ Vt))
        D = np.diag([1.0, 1.0, d])
        rot = U @ D @ Vt
        rmsd_list.append(np.sqrt(((cur @ rot - ref) ** 2).sum(1).mean()))
    rg = np.array(rg_list)
    rmsd = np.array(rmsd_list)
    lam = (T_REF / T_MAX) ** (rep / 15.0)
    mult = LAMBDA_HOT ** (np.log(lam) / np.log(T_REF / T_MAX)) if lam < 1 else 1.0
    t_eff = T_REF / lam
    t_hot = T_REF / (lam * mult)
    results["slots"].append({
        "slot": rep, "n_frames": n_frames, "lambda": lam,
        "T_eff_solute": t_eff, "T_eff_hot": t_hot,
        "rg_mean_A": float(rg.mean()), "rg_std_A": float(rg.std()),
        "rg_sem_A": float(rg.std() / np.sqrt(len(rg))),
        "rmsd_mean_A": float(rmsd.mean()), "rmsd_std_A": float(rmsd.std()),
    })
    print(f"slot {rep:2d}: λ={lam:.4f}  T_eff={t_eff:6.1f}K  T_hot={t_hot:6.1f}K  "
          f"Rg={rg.mean():.3f}±{rg.std():.3f} Å  RMSD={rmsd.mean():.2f}±{rmsd.std():.2f} Å  (n={len(rg)})")

# 拟合: Rg vs T_eff_solute (普通溶质温标) 与 Rg vs T_eff_hot (门区有效温标)
rg = np.array([s["rg_mean_A"] for s in results["slots"]])
rgw = np.array([s["rg_sem_A"] for s in results["slots"]])
for key, xname in [("T_eff_solute", "Rg vs 溶质有效温标"),
                   ("T_eff_hot", "Rg vs 门区有效温标")]:
    x = np.array([s[key] for s in results["slots"]])
    A = np.vstack([x, np.ones_like(x)]).T
    W = np.diag(1.0 / rgw ** 2)
    coef = np.linalg.solve(A.T @ W @ A, A.T @ W @ rg)
    pred = A @ coef
    ss_res = np.sum(1/rgw**2 * (rg - pred) ** 2)
    ss_tot = np.sum(1/rgw**2 * (rg - np.average(rg, weights=1/rgw**2)) ** 2)
    r2 = 1 - ss_res / ss_tot
    results[key] = {"slope_A_per_K": float(coef[0]), "intercept_A": float(coef[1]),
                    "R2": float(r2)}
    print(f"{xname}: slope={coef[0]*1000:.3f} mÅ/K, R²={r2:.3f}")

results["assumption"] = ("stride=100 迭代/帧; 每迭代 n_steps_per_move×2 fs; "
                         "时间轴与总时长需与学生的 run config (rest2_results.npz) 核对")

with open(os.path.join(BASE, "l4_rg_results.json"), "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# 绘图
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 10})
fig, axes = plt.subplots(1, 2, figsize=(8.27 * 0.4 * 2, 11.69 * 0.4 / 3 * 1.6))
for ax, key, lab in [(axes[0], "T_eff_solute", "Effective solute T (K)"),
                     (axes[1], "T_eff_hot", "Effective gate T (K)")]:
    x = np.array([s[key] for s in results["slots"]])
    y = rg
    e = rgw
    ax.errorbar(x, y, yerr=e * 1.96, fmt="o", ms=4, lw=0.8, color="C2",
                capsize=2, label="OF-gate Rg (95% CI)")
    fit = results[key]
    xx = np.linspace(x.min(), x.max(), 50)
    ax.plot(xx, fit["slope_A_per_K"] * xx + fit["intercept_A"], "k--", lw=0.8,
            label=f"slope={fit['slope_A_per_K']*1000:.2f} mÅ/K, R²={fit['R2']:.2f}")
    ax.set_xlabel(lab)
    ax.set_ylabel("Rg (Å)")
    ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(os.path.join(BASE, f"l4_rg_plot.{ext}"), dpi=300)
print("图与 JSON 已写入 00_verification/")
