# REST2 体系正确性验证方案 (v4.4)

制定日期: 2026-10-19
对象: `versions/rest2_4au5_v4.4.py`(local-hot 默认) + `--no_local_heating`(标准 REST2)
体系: `charmm-gui/4au5-aa-k300n/` (68,411 原子, CHARMM36m, PME/1.2 nm, HBonds)

---

## 0. 前置盘点 (2026-10-19 实测)

### 0.1 文件可用性

| 文件 | 状态 | 用途 |
|---|---|---|
| `charmm-gui/4au5-aa-k300n/openmm/step5_charmm2omm.psf` (17.2 MB) | ✅ | Layer 1/2 全部 |
| `charmm-gui/4au5-aa-k300n/openmm/step5_charmm2omm.pdb` (5.3 MB) | ✅ | Layer 1/2/3 |
| `charmm-gui/4au5-aa-k300n/openmm/step5_charmm2omm.crd` (9.6 MB) | ✅ | 仅 CHARMM 重跑需要 |
| `charmm-gui/4au5-aa-k300n/openmm/step5_charmm2omm.str` | ✅ | 盒子尺寸 |
| `charmm-gui/4au5-aa-k300n/toppar.str` + `toppar/` (50 文件, 21 MB) | ✅ | Layer 2 重建体系 |
| `versions/rest2_4au5_v4.4.py` / `v4.3.py` | ✅ | 验证对象 |
| `environment_export.yml` / `requirements_export.txt` | ✅ | 环境重建 |
| **v4.3 标准 REST2 对照实验产物** (`outputs/experiment_20260919_002619/`) | ❌ **缺失** | Layer 4 的 T₀ 分布交叉核对 |
| **v4.4 预实验产物** (`outputs/experiment_20261012_101652/`) | ❌ **缺失** | Layer 4 复核 |
| `experiment.log` (历史运行日志) | ❌ **缺失** | Layer 2 的 improper/CMAP warning 计数 |

**缺失文件处理**: 前两项须向实验执行者取回(或从 v4.4 checkpoint 恢复段找); 若无法取回, Layer 4 改为"新跑短实验自建基线"。`experiment.log` 缺失不阻塞——Layer 2 的验证脚本可现场重算 warning 计数。

### 0.2 环境可用性

| 环境 | 状态 | 可承担的层 |
|---|---|---|
| `omm760` (OpenMM 7.6, 现存) | ✅ 可用 | Layer 1/2 (参数检查+CPU 能量评估, 不需 LangevinMiddleIntegrator) |
| `rest2_4au5` (OpenMM 8.5.1) | ❌ **未创建** | Layer 3/4 必需 |
| GPU | ✅ slurm `gpu` 分区 (16 mixed / 4 idle); 登录节点无 GPU | Layer 3/4 经 sbatch |
| 磁盘 | ✅ home 剩余 3.4 T | 充裕 |

---

## Layer 1: 静态构建检查 — 可行性 ★★★★★

**目标**: 拓扑/原子数/segid/编号映射正确。

1. 解析 PSF: NATOM=68,411; segid 分布 (PROA/PROB/MEMB/TIP3/SOD/CLA) 与 README 一致
2. PDB-PSF 原子一一对应 (学生代码用 `pdb.topology` 索引改 PSF System, 必须断言)
3. 残基编号偏移 +9: 五个 hot 残基 (输入 72/167/168/344/345) 在拓扑中唯一、名称匹配、全部为溶质原子
4. 盒子 82.4068×82.4068×109.362 Å 与 `step5_charmm2omm.str` 一致

**可行性**: 完全可行, omm760 CPU 即可, ~1 分钟。
**缺失**: 无。

## Layer 2: 单点能量学验证 — 可行性 ★★★★☆ (核心层)

**目标**: 缩放哈密顿量数学正确。

1. **force 类型枚举**: 列出 base system 全部 force 类型, 对照 v4.4 代码覆盖清单 (HarmonicBond/Angle/PeriodicTorsion/CustomTorsion/CMAP/Nonbonded/CustomNonbonded)。**重点查有无 CustomBondForce 或其他未覆盖类型**——若存在, v4.4 的 if/elif 会静默跳过
2. **槽位 0 恒等**: `--no_local_heating` 与 local-hot 两种模式的 λ=1 System 逐 force 逐参数 ≡ base system
3. **缩放律标定**: 最高槽位抽查 bond k×s、angle k×s、torsion k×s、电荷 ×√s、ε×s、exception ×√(s_a·s_b); hot/normal/mixed 三类项各抽 3 项
4. **CMAP/improper 覆盖审计**: 现场重跑缩放函数, 统计 "CustomTorsionForce expression cannot be scaled reliably" warning 数; CMAP normal/hot/mixed 计数
5. **能量标定**: 同一构象在 base vs 最高槽位 System 下的 CPU 单点能, 验证 E(λ) = λE_pp + √λE_pw + E_ww 的分解关系 (按原子对归属分组)

**可行性**: 高。全部可在 omm760 CPU 上完成; 唯一风险是 v4.4 脚本为运行时设计, import 时需绕开 `__main__` 与 CUDA cache 环境变量 (已确认 import 副作用安全)。
**缺失**: 无。预计 ~30 分钟 (PSF 解析 + 16 System 重建)。

## Layer 3: 短程动力学冒烟 — 可行性 ★★★☆☆

**目标**: 数值稳定。

1. 先建 `rest2_4au5` 环境 (OpenMM 8.5.1): `conda env create -f environment_export.yml`; 若 yml 解析失败, fallback = 新建 env + pip 装 openmm/numpy/mdanalysis
2. 最高槽位 (slot 15) 单副本 1–2 ns, 各自哈密顿量下最小化后起跑; 查 NaN/温度/密度/膜完整
3. 用 CLI 缩短平衡: `--npt_equilibration_steps 100000 --nvt_equilibration_steps 50000 --unrestrained_equilibration_steps 50000`

**可行性**: 中高。瓶颈 = 环境创建 (yml 跨平台解析可能失败, 已备 fallback) + GPU 排队 (gpu 分区 4 卡 idle)。单副本 2 ns 在 A40 约 1–3 小时。
**缺失**: 无 (输入齐全)。

## Layer 4: 交换与分布统计 — 可行性 ★★☆☆☆ (受文件缺失制约)

**目标**: 交换质量 + T₀ 分布无偏。

1. 16 副本 × 5–10 ns 短实验: 相邻接受率 20–40%、每副本遍历全阶梯
2. T₀ 分布交叉核对: 对比 slot 0 与**标准 REST2 对照**的 CA RMSD / 门区距离分布
3. hot-heavy RMSD 警戒线 (3/5 Å) 全程无触发

**可行性**: 中。Layer 4.1 可行 (16×A40, 约 1–2 天墙钟); **Layer 4.2 的原计划依赖缺失的 v4.3 对照产物**——两个替代: (a) 取回缺失文件; (b) 用 v4.4 `--no_local_heating` 新跑 5 ns 标准基线 (成本 = 再占 16 副本 × 5 ns)。
**缺失**: v4.3 对照产物 (见 0.1)。

## 执行顺序与判定

```
Layer 1 (必过) → Layer 2 (必过, 核心判据) → [建 env] → Layer 3 (slot 15 稳定)
→ Layer 4.1 短实验 (接受率/游走) → Layer 4.2 分布核对 (等 v4.3 产物或自建基线)
```

**一票否决项**: Layer 2 的 force 枚举发现未覆盖力类型、槽位 0 ≠ base、缩放律偏差 >1%——三者任一出现, 后续全部暂停并回退修代码。

## 交付物

- `verify_rest2.py`: Layer 1+2 自动化脚本 (产出 JSON 报告)
- Layer 3/4 沿用 `rest2_4au5_v4.4.py` 的 `experiment_evaluation.json` 分维度评价
- 本文档随 README 提交 git, 验证结果追加为附录

---

*制定: Hermes Agent, 2026-10-19; 依据: README v4.4 (2026-09-20), 代码审查 (3147 行), zimei 环境实测*
