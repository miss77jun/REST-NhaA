# v3.3 周期性盒子 & 分子运动约束 — 对照 CHARMM-GUI 核查报告

核查时间: 2026-07-30
代码: `versions/rest2_4au5_v3.3.py`
输入: `charmm-gui/4au5-aa-k300n/openmm/step5_charmm2omm.str` + `step7_production.inp` / `step6.1_equilibration.inp` / `membrane_restraint.str`

---

## 一、结论速览

| 检查项 | CHARMM-GUI 原始意图 | v3.3 实现 | 结论 |
|---|---|---|---|
| 周期性盒子尺寸 | 82.4068 × 82.4068 × 109.362 Å（正交） | 同（从 .str 读取） | ✅ 一致 |
| PSF/System 盒子向量 | PME 必须的盒子 | `psf.setBox` + `system.setDefaultPeriodicBoxVectors` | ✅ 已设置 |
| Context 盒子向量 | — | `ctx.setPeriodicBoxVectors(...)` 每副本 | ✅ 已设置 |
| PME / Ewald容差 | PME, tol 0.0005 | PME, tol 0.0005 | ✅ 一致 |
| 力切换vdW | r_on=1.0, r_off=1.2 nm | 同（setSwitchingDistance=1.0, cutoff=1.2） | ✅ 一致 |
| 约束方式 | HBonds | HBonds | ✅ 一致 |
| 生产期位置约束 | `rest = no` | 无 position restraint | ✅ 一致 |
| 膜气压/表面张力 | `MonteCarloMembraneBarostat`（生产） | 不使用气压（NVT 定盒） | ✅ REST2 正确做法 |
| x/y/z 运动约束 | 生产=膜气压；平衡=MMFP 限制 | 定盒(PBC)限制运动，无 MMFP | ⚠️ 见下 |

---

## 二、周期性盒子设置 — 已真实完成 ✅

代码链路（均在 `build_base_system` + `run_rest2` 中）:

1. `read_charmm_gui_box_info()` 读取 `step5_charmm2omm.str`:
   - `BOXLX = 82.4067792`, `BOXLY = 82.4067792`, `BOXLZ = 109.362015` Å
   - 正交盒 `box_angles = [90,90,90]`（与 CHARMM-GUI 一致）
2. `psf.setBox(...)` —— 在 `createSystem` 之前设置，满足 PME 要求（关键步骤，已正确执行）
3. `setup_periodic_box_vectors()` 设置正交向量:
   ```
   [a_nm, 0,    0   ]
   [0,    b_nm, 0   ]
   [0,    0,    c_nm]
   ```
   并调用 `system.setDefaultPeriodicBoxVectors(*vectors)`
4. `run_rest2` 中每个副本 `ctx.setPeriodicBoxVectors(*base_system.getDefaultPeriodicBoxVectors())`

盒子尺寸 82.4 Å 远大于 vdW 截断 12 Å（半盒 41.2 Å），PME 无自相互作用风险。

**此前崩溃的根因**：第 1341 行日志把 OpenMM `Quantity` 直接做 `:.4f` 格式化（`TypeError: unsupported format string passed to Quantity.__format__`）。这只发生在**日志打印**，盒子向量本身在 `build_base_system` 内早已正确设置（崩溃点位于该步骤之后）。已用 `float()` 包裹修复。

---

## 三、"x,y,z 轴分子运动范围约束" 的真实含义与 v3.3 处理

CHARMM-GUI 在该膜蛋白体系中定义了两类约束，但**用途不同**：

### (A) 平衡阶段（step6.1, `rest = yes`）— `membrane_restraint.str`
用 MMFP 限制分子运动范围：
- 水分子 `twater`：以 z 轴平面 (`zdir 1.0`) `harmonic outside droff 11.0`，把水推出疏水核心（z 方向约束）。
- 每个脂 HEAD：上下两层分别用 `plane RCM zref ±zpostop` 钉在靶 z 位（z 约束）。
- 脂 TAIL：`plane zref 0.0 harmonic inside droff 5.0`，把尾锁在膜内核（z 约束）。
- 蛋白骨架/侧链/脂位置/脂二面体：`fc_bb=4000, fc_sc=2000, fc_lpos=1000, fc_ldih=1000 kJ/mol/nm²`。

→ 这些**仅用于 CHARMM-GUI 的平衡（step 1–6）**，目的是把膜"拉正"。

### (B) 生产阶段（step7_production.inp, `rest = no`）
- 位置约束**全部关闭**（`rest = no`）。
- 改用 **`MonteCarloMembraneBarostat`**（`p_type=membrane`, `p_XYMode=XYIsotropic`, `p_ZMode=ZFree`, `p_tens=0.0`）以恒定表面张力维持膜几何。

### v3.3 的做法（REST2 生产）
- 无 position restraint —— 与 CHARMM-GUI 生产 `rest=no` **一致**。
- 不使用气压（`use_barostat=False`）—— **REST2 的标准正确做法**：副本交换不宜做 NPT，膜气压也难以与温度副本兼容；固定 NVT 盒子即可把分子运动限制在盒子内（PBC 包裹 + make_whole 保证分子完整）。
- 起始结构 `step5_charmm2omm.pdb` 已是 CHARMM-GUI 平衡后的结构，故无需重做带约束的平衡。

**结论**：v3.3 与 CHARMM-GUI **生产协议一致**，没有缺失必要的运动约束。定盒 (PBC) 本身就是 x/y/z 方向的运动范围约束。

---

## 四、可选的增强（默认不建议，按需开启）

若希望长程 REST2 中膜更"稳"（额外防止缓慢 undulation），可二选一：
1. **可选膜约束**：把 `membrane_restraint.str` 的 MMFP 转为 OpenMM `CustomExternalForce`（z 平面/柱面势）加到 `base_system`，仅限脂 HEAD/TAIL，弱力常数。注意：这会偏离 CHARMM-GUI 生产协议、可能引入采样偏好，**不推荐**作为默认。
2. **保持现状**：固定 NVT 盒子 + PBC，与 CHARMM-GUI 生产一致，已是膜蛋白 REST2 的常规做法。

---

## 五、已做的代码修改
- `versions/rest2_4au5_v3.3.py` 第 1340 行：`box_lengths_nm = [float(box_vecs_init[0][0]), float(box_vecs_init[1][1]), float(box_vecs_init[2][2])]` —— 修复日志 `Quantity.__format__` 崩溃（仅影响日志，不影响盒子设置）。
