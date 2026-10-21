# 4AU5 REST2 / local-hot REST2 模拟

本项目使用 OpenMM 对 4AU5 膜蛋白体系执行 REST2（Replica Exchange with Solute Tempering 2）模拟。当前推荐脚本为 `versions/rest2_4au5_v4.4.py`。v4.4 默认启用一个针对五个残基的 **non-standard local-hot REST2 方法学扩展**；使用 `--no_local_heating` 可回到标准 REST2。

> 重要：所有副本的物理恒温热浴均为 300 K。文中的 300–350 K 是 REST2 溶质有效温标，不是各副本的物理热浴温度。局部 437.5 K 同样是 Hamiltonian 缩放对应的有效温标，不是实际恒温器温度。

## 1. 项目概述

### 1.1 模拟体系

| 项目 | 当前体系 |
|---|---|
| 输入目录 | `charmm-gui/4au5-aa-k300n/openmm` |
| 总原子数 | 68,411 |
| 蛋白残基数 | 381 |
| 蛋白力场 | CHARMM36m |
| 溶剂与离子 | TIP3P，0.15 M NaCl |
| 环境 | 膜体系，周期性边界条件 |
| 非键处理 | PME，1.2 nm cutoff；vdW force-switch 1.0–1.2 nm |
| 约束 | HBonds，刚性水 |
| 目标残基 | PHE72、ALA167、ILE168、PHE344、ILE345 |

输入文件为：

- `charmm-gui/4au5-aa-k300n/openmm/step5_charmm2omm.psf`
- `charmm-gui/4au5-aa-k300n/openmm/step5_charmm2omm.pdb`
- `charmm-gui/4au5-aa-k300n/openmm/step5_charmm2omm.str`
- `charmm-gui/4au5-aa-k300n/toppar.str`

输入 PDB 中蛋白残基编号为 10–390；OpenMM 写出的 final PDB 会重编号为 1–381，偏移量为 9。分析目标残基时，应确认使用的是输入编号还是输出 PDB 编号。

### 1.2 v4.4 的定位

标准 REST2 对整个溶质 Hamiltonian 进行缩放。v4.4 在此基础上，默认对 PHE72、ALA167、ILE168、PHE344、ILE345 再施加端点锚定的局部缩放。该 local-hot 方案是本项目的方法学扩展，**不是 REST2 原始论文定义的标准算法**。

- 默认：REST2 + non-standard five-residue local-hot 扩展。
- `--no_local_heating`：禁用局部扩展，运行标准 REST2。
- 参考端严格无额外局部偏置。
- 最高端默认 `lambda_hot=0.8`，五残基局部有效温标为 437.5 K。

## 2. 安装与运行环境

建议使用独立 Conda 环境，并优先使用 CUDA GPU。脚本按 CUDA、OpenCL、CPU 的顺序选择平台，GPU 平台使用 mixed precision。

核心依赖包括：

- Python 3.10+；当前导出环境使用的核心包可见 `environment_export.yml` 和 `requirements_export.txt`
- OpenMM 8.5.1
- NumPy
- MDAnalysis 2.9.0（轨迹验证与 OF-gate Rg 分析）
- Matplotlib（绘图）
- SciPy（Rg 线性趋势）

从已有环境文件创建环境：

```bash
conda env create -n rest2_4au5 -f environment_export.yml
conda activate rest2_4au5
```

若导出环境因平台差异无法直接解析，应在新的 Conda 环境中至少安装上述核心依赖，并先执行短测试。`requirements_export.txt` 是完整环境快照，包含与模拟无关的包，不建议不加检查地覆盖现有环境。

在项目根目录检查 CLI：

```bash
python versions/rest2_4au5_v4.4.py --help
```

## 3. 默认正式方案

| 参数 | v4.4 默认值 | 含义 |
|---|---:|---|
| 副本数 | 16 | Hamiltonian 槽位数 |
| `T_ref` / `T_max` | 300 / 350 K | 溶质有效温标范围 |
| 物理热浴 | 300 K | 所有副本相同 |
| 时间步长 | 2 fs | LangevinMiddleIntegrator |
| 每轮步数 | 1,000 | 每 2 ps 尝试一次交换 |
| 生产轮数 | 50,000 | 100 ns/副本 |
| 轨迹间隔 | 5 轮 | 每 10 ps 一帧，另写初始帧 |
| full checkpoint 间隔 | 500 轮 | 每 1 ns |
| NPT 平衡 | 2,500,000 步 | 5 ns，共同盒、半各向同性膜 NPT |
| 受约束 NVT | 250,000 步/副本 | 0.5 ns，各 Hamiltonian、300 K 热浴 |
| 无约束 NVT | 250,000 步/副本 | 0.5 ns，释放 CA 约束后 |
| 平衡期 CA 位置约束 | 100 kJ mol⁻¹ nm⁻² | NPT 和第一段 NVT 使用，生产前移除 |
| `lambda_hot` | 0.8 | 最高 REST2 端点的额外局部倍率 |
| 随机种子 | 20260920 | NumPy 与派生 OpenMM seeds |

16 个副本各生产 100 ns，因此总积分量为 1.6 μs；这表示 **16 条副本积分时间的总和**，不等于一条连续 1.6 μs 轨迹，也不能按单轨迹方式解释动力学时间。

## 4. 方法

### 4.1 标准 REST2 与温标

第 `i` 个槽位的有效溶质温度采用几何梯度：

```text
T_eff,i = T_ref × (T_max / T_ref)^(i / (N - 1))
lambda_i = T_ref / T_eff,i
```

默认 `N=16`、`T_ref=300 K`、`T_max=350 K`，因此 `lambda` 从 1 下降到 `300/350 = 0.857142...`。标准 REST2 可概括为：

```text
U_i(X) = lambda_i U_ss(X) + sqrt(lambda_i) U_sv(X) + U_vv(X)
```

其中 `ss`、`sv`、`vv` 分别表示溶质–溶质、溶质–环境和环境–环境贡献。所有槽位仍由 300 K 恒温器推进。

### 4.2 v4.4 端点锚定 local-hot 缩放

令：

```text
lambda_min = T_ref / T_max
progress = log(lambda_global) / log(lambda_min)
multiplier = lambda_hot^progress
s_hot = lambda_global × multiplier
```

`progress` 在实现中限制到 `[0, 1]`。普通溶质原子的原子尺度为 `s=lambda_global`，五个目标残基原子的尺度为 `s=s_hot`，环境原子的尺度为 `s=1`。

这一定义保证：

- 参考端：`lambda_global=1`、`progress=0`、`multiplier=1`，因此完全无偏；
- 最高端：`lambda_global=300/350`、`progress=1`、`multiplier=lambda_hot=0.8`；
- 最高端目标残基原子尺度为 `(300/350)×0.8=0.685714...`；
- 局部有效温标为 `350/0.8=437.5 K`。

### 4.3 力项的原子尺度与几何平均

v4.4 的 `endpoint_anchored_geometric_atom_scale_v2` 先为每个原子定义尺度 `s_a`，再按力项所含原子的几何平均缩放该项：

```text
S_term = (Π_a s_a)^(1/n)
```

具体规则：

- **bond**：两个原子的尺度几何平均，缩放力常数 `k`；平衡键长不变。
- **angle**：三个原子的尺度几何平均，缩放力常数 `k`；平衡角不变。
- **periodic torsion**：四个原子的尺度几何平均，缩放振幅 `k`；周期数和相位不变。
- **CHARMM improper / CustomTorsionForce**：对已识别的 CHARMM36m improper 振幅 `k` 使用四原子几何平均；无法可靠识别的表达式保持不变并记录警告。
- **CMAP**：对八个参与原子取几何平均，为实际尺度缓存并缩放 CMAP map。
- **NonbondedForce 粒子参数**：电荷乘 `sqrt(s_a)`，epsilon 乘 `s_a`，sigma 不变。
- **普通非键原子对和 CHARMM CustomNonbondedForce LJ**：使用 `sqrt(s_a s_b)`，自然恢复标准 REST2 中的溶质–溶质 `lambda` 和溶质–环境 `sqrt(lambda)`。
- **nonbonded exception**：`chargeProd` 与 epsilon 均乘 `sqrt(s_a s_b)`，sigma 不变。
- **混合项**：若一个 bonded、CMAP 或非键项同时含 hot、普通溶质或环境原子，按上述原子尺度几何平均连续过渡，不进行全有或全无分类缩放。
- **constraints**：约束由 OpenMM `System` 原样复制，约束距离不缩放。

### 4.4 固定物理温度下的交换判据

相邻 Hamiltonian 槽位 `m`、`n` 分别含构型 `X_m`、`X_n`。每次交换需要四次势能组合：

```text
E_m(X_m), E_n(X_n), E_m(X_n), E_n(X_m)
Delta = beta_ref × [E_m(X_n) + E_n(X_m) - E_m(X_m) - E_n(X_n)]
P_accept = min(1, exp(-Delta))
beta_ref = 1 / (k_B T_ref)
```

由于所有槽位的物理温度相同，判据只使用 `beta_ref`。接受交换时移动构型及其周期盒，并交换 walker 标签；各槽位保留自己的速度，不做温度重标定。

## 5. CLI 与可复现实例

### 5.1 命令行参数

调用方式：

```bash
python versions/rest2_4au5_v4.4.py [OPTIONS]
```

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--n_iterations` | 50000 | 总生产轮数；不是“新增轮数” |
| `--n_steps_per_move` | 1000 | 每轮积分步数 |
| `--n_replicas` | 16 | 副本/Hamiltonian 槽位数 |
| `--T_ref` | 300 | 参考温度与所有副本物理热浴温度，K |
| `--T_max` | 350 | 最高溶质有效温度，K |
| `--lambda_hot` | 0.8 | 最高端局部倍率 |
| `--local_heating` | 默认已启用 | 显式启用 non-standard local-hot 扩展 |
| `--no_local_heating` | false | 禁用局部扩展，使用标准 REST2 |
| `--equilibration_restraints` | 默认已启用 | 显式启用平衡期 CA 位置约束 |
| `--no_equilibration_restraints` | false | 禁用平衡期位置约束 |
| `--checkpoint_interval` | 500 | full checkpoint 间隔，轮 |
| `--npt_equilibration_steps` | 2500000 | 共同盒 NPT 步数 |
| `--nvt_equilibration_steps` | 250000 | 受约束 NVT 步数/副本 |
| `--unrestrained_equilibration_steps` | 250000 | 无约束 NVT 步数/副本 |
| `--trajectory_interval` | 5 | DCD 保存间隔，轮 |
| `--rmsd_monitor_interval` | 50 | RMSD 监控间隔，轮，即默认 100 ps |
| `--restore` | 无 | 要恢复的 `checkpoint_*_full.npz` |
| `--output_dir` | 新运行自动创建 | 恢复时必需，且应指向原实验目录 |
| `--start_iteration` | 0 | 兼容参数；恢复时实际迭代号由 checkpoint 决定 |

随机种子、2 fs 步长和摩擦系数等配置目前没有 CLI 参数；正式复现应保留源码默认值并保存生成的 `config.json`。

### 5.2 1 ns 预实验

```bash
python versions/rest2_4au5_v4.4.py --n_iterations 500
```

这会执行完整默认平衡，然后为每个副本生产 1 ns。建议在新硬件、驱动或环境上先做更短的功能测试，再做 1 ns 预实验，检查数值稳定性、交换率、RMSD、DCD/PBC 和 checkpoint。

### 5.3 默认正式运行

```bash
python versions/rest2_4au5_v4.4.py
```

默认输出到 `outputs/experiment_YYYYMMDD_HHMMSS/`，生产长度为 100 ns/副本。

### 5.4 标准 REST2 对照

```bash
python versions/rest2_4au5_v4.4.py --no_local_heating
```

此时五残基不再接受额外局部倍率，但全溶质仍按标准 REST2 缩放。

### 5.5 从 full checkpoint 恢复

例如从第 500 轮恢复，并继续到总计 50,000 轮：

```bash
python versions/rest2_4au5_v4.4.py \
  --restore outputs/experiment_YYYYMMDD_HHMMSS/checkpoint_500_full.npz \
  --output_dir outputs/experiment_YYYYMMDD_HHMMSS \
  --n_iterations 50000
```

注意：

1. 只有 `checkpoint_*_full.npz` 可用于 v4.4 恢复。
2. 使用 `--restore` 时 `--output_dir` 必需。
3. 恢复迭代号、Hamiltonian 配置、副本数、RNG 状态及关键输入哈希由 checkpoint 读取并验证。
4. 不要为恢复而随意指定 `--start_iteration`；若其与 checkpoint 不同，脚本会忽略它并采用 checkpoint 迭代号。
5. 恢复后 DCD 写入新的 `replica_<i>_cont_from_<iteration>.dcd`，不会向旧 DCD 直接追加。
6. OpenMM 的 NPZ 状态恢复不保证 bitwise 完全一致，但保存了继续运行所需的位置、速度、盒子、统计、walker 历史及 NumPy RNG 状态。

## 6. 最新实验结果

以下结论仅来自项目内对应的 `scientific_report.json`、`rg_analysis.json` 和 `experiment.log`。两个实验的方法不同，不能混为同一个连续实验。

### 6.1 A：`experiment_20260919_002619`，v4.3 标准 REST2 对照

- 方法：v4.3 标准 REST2，local heating 关闭。
- 长度：25,000 轮，50 ns/副本。
- 相邻交换率：47.75–52.61%，均值 50.17%；交换连续且良好。
- walker mixing：总计 94 次完整端点 round trip；16 个 walker 中 15 个至少完成一次。
- 后半段 CA RMSD：槽位 0、7、15 分别约 2.26、2.28、2.09 Å，结构稳定。
- OF-gate Rg：各槽位均值约 4.67–4.69 Å；线性斜率 `-0.000145 Å/K`，`R²=0.185`。
- 结论：该标准 REST2 对照在 50 ns/副本内结构稳定、交换和 walker 混合良好，但 Rg 对有效温标没有明确响应。50 ns 对照是有价值的质量基线，不替代 v4.4 默认 100 ns/副本正式实验。

### 6.2 B：`experiment_20261012_101652`，v4.4 local-hot 1 ns 预实验

- 方法：v4.4 non-standard local-hot，`lambda_hot=0.8`，参考端无偏。
- 长度：500 轮，1 ns/副本。
- 相邻交换率：40.0–50.4%，均值 45.28%；全部高于实际 pass 阈值 20%。
- 后半段 CA RMSD：槽位 0、7、15 分别约 1.41、1.53、1.40 Å。
- 后半段 hot-heavy RMSD：槽位 0、7、15 分别约 0.79、0.98、0.94 Å；所有记录均未触及 3 Å 警戒线。
- walker mixing：0 round trip。1 ns 对跨越 16 个 Hamiltonian 槽位明显过短，因此该结果表示时长不足，不能据此判定采样错误。
- OF-gate Rg：端点约从 4.70 Å 降至 4.60 Å；对局部有效温标的线性斜率为 `-0.00277 Å/K`，`R²=0.809`。
- Rg 限制：每个槽位仅 81 个分析帧（101 个 DCD 帧去掉前 20%），时间序列存在自相关，各 Hamiltonian 槽位也不是独立重复实验。因此结果只能标记为 **exploratory**，不得声称统计显著、因果机制或稳定构象转换。
- 综合结论：核心安全检查通过，`production_readiness = ready`，`overall_stage = pilot_completed_ready_for_production`。这表示可以进入正式生产，不表示采样已成熟或科学机制已证实。

## 7. 实验评价体系

v4.4 将评价拆成独立维度，避免用一个总分掩盖风险：

| 维度 | 评价内容与解释 |
|---|---|
| `numerical_stability` | 能量必须全部有限；首末 20% 均值差/全程标准差用于漂移筛查，但不是严格的平衡或收敛检验。 |
| `exchange_quality` | 逐一检查全部相邻 pair。20–60% 可视为良好经验范围；程序的实际 pass 条件是所有 pair 的最小交换率 `>=20%`。10–20% 为 warning，低于 10% 为 fail。 |
| `structural_safety` | 检查全部已记录 CA RMSD 与 local-hot 重原子 RMSD；hot-heavy 3 Å 为 warning、5 Å 为 severe/fail 参考线，CA 使用 4/6 Å 分级。 |
| `walker_mixing` | 基于 `slot -> walker` 历史统计槽位覆盖、端点访问和完整 round trip。0 round trip 必须结合时长：短于 10 ns 是时长不足，不直接判错；长运行仍无往返才构成风险。 |
| `sampling_maturity` | 区分 pilot、preliminary 和 production mixing。项目的默认正式设计是 100 ns/副本；只有完成这一级别的 v4.4 正式实验，才用于最终采样成熟度判断。短预实验只判断能否进入生产。 |
| `rg_evidence` | 报告均值、端点差、线性斜率、R²、帧数和限制。生产短于 10 ns或任一槽位少于 200 帧时，只能作 exploratory 描述；同时始终考虑时间自相关和槽位非独立性。 |
| `production_readiness` | 仅当数值稳定、交换质量和结构安全均通过且无 fatal issue 时为 `ready`。短预实验中的 walker/Rg 未成熟不会单独阻止生产。 |
| `overall_stage` | 汇总实验所处阶段，例如 `pilot_completed_ready_for_production`；它不是“机制成立”或“采样收敛”的同义词。 |

正式分析还应进行分块统计、自相关时间/有效样本量、重复实验和适当的再加权。当前输出不包含可直接用于 MBAR 的完整交叉态矩阵。

## 8. 输出文件

每次新运行创建 `outputs/experiment_YYYYMMDD_HHMMSS/`。主要内容如下：

| 文件 | 内容 |
|---|---|
| `config.json` | 实际运行配置快照 |
| `experiment.log` | 参数、平衡、交换、RMSD、分析与评价日志 |
| `checkpoint_<iteration>_full.npz` | 唯一可恢复的完整 checkpoint；含位置、速度、盒子、统计、walker/RNG 状态及一致性元数据 |
| `rest2_results.npz` | 能量历史、交换统计、lambda/温标、RMSD、walker 混合和评价 JSON |
| `scientific_report.json` | 便于审阅的科学报告摘要 |
| `experiment_evaluation.json` | 分维度实验质量评价 |
| `walker_label_history.npy` | 每轮 Hamiltonian 槽位到 walker 标签的映射 |
| `rmsd_history.npy` | 槽位 0、7、15 的 CA 和 local-hot 重原子 RMSD 历史 |
| `rg_analysis.json` | OF-gate Rg 均值、标准差和趋势 |
| `mbar_analysis/own_state_energies.npz` | 每帧在其自身 Hamiltonian 下的势能；**不是**完整 MBAR `u_kn/N_k`，不能直接用于 MBAR |
| `trajectories/replica_<i>.dcd` | 各 Hamiltonian 槽位轨迹；恢复段使用 `*_cont_from_*.dcd` |
| `replica_<i>_final.pdb` | 各槽位最终结构 |
| `figures/*.png` | 能量/交换、RMSD 和 Rg 图 |

DCD 表示 Hamiltonian 槽位轨迹，而 walker 的温标迁移应结合 `walker_label_history.npy` 重建。

## 9. VMD / 轨迹加载与 PBC

DCD 本身不含完整拓扑。必须先加载与轨迹 **完全相同的 68,411 原子全体系** PSF/PDB，再加载 DCD。不要先加载仅蛋白结构或旧的 75,840 原子拓扑。

VMD 示例：

```tcl
mol new charmm-gui/4au5-aa-k300n/openmm/step5_charmm2omm.psf type psf waitfor all
mol addfile charmm-gui/4au5-aa-k300n/openmm/step5_charmm2omm.pdb type pdb waitfor all
mol addfile outputs/experiment_YYYYMMDD_HHMMSS/trajectories/replica_0.dcd type dcd waitfor all
```

v4.4 已将 PBC unit-cell 写入 DCD。写帧前会对大分子执行 make-whole；“分子完整”不等于“整个分子必须位于主盒内”，因此完整蛋白或脂质可能跨主盒边界显示。若需要把片段重新包回可视主盒，可在 VMD/PBCTools 中使用：

```tcl
package require pbctools
pbc wrap -all -compound fragment -center com -centersel "protein"
```

在做距离、Rg 或构象分析前，应明确所需的 unwrap/make-whole/wrap 语义，避免把周期镜像跳跃误认为结构变化。

## 10. 项目结构

```text
4AU5/
├── charmm-gui/4au5-aa-k300n/
│   ├── openmm/                  # 当前 PSF/PDB、盒信息及 CHARMM-GUI OpenMM 文件
│   ├── toppar/                  # CHARMM36m、脂质、水和离子参数
│   └── toppar.str               # 参数入口
├── versions/
│   ├── rest2_4au5_v4.4.py       # 当前推荐版本
│   ├── rest2_4au5_v4.3.py       # 标准 REST2 对照版本
│   └── rest2_4au5_v4.0-v4.2.py  # 历史版本
├── outputs/
│   └── experiment_*/            # 实验、轨迹、报告、checkpoint 与图
├── scripts/                     # 辅助分析脚本
├── docs/                        # 补充说明
├── environment_export.yml       # 导出的 Conda 环境
├── requirements_export.txt      # 完整 Python 环境快照
└── README.md
```

## 11. 版本历史

| 版本 | 摘要 | 状态 |
|---|---|---|
| v4.0 | 引入局部加热控制、残基映射、RMSD 监控、make-whole 和平衡约束等基础设施。 | 历史 |
| v4.1 | 修正标准 REST2 Hamiltonian 与交换流程。 | 历史 |
| v4.2 | 16 副本 300–350 K 正式参数；标准 REST2 默认；改进 PBC/DCD、平衡流程和 full checkpoint。 | 历史 |
| v4.3 | 增加可靠 walker 标签、round-trip 分析、续算一致性和实验质量诊断；用于 50 ns 标准 REST2 对照。 | 对照 |
| **v4.4** | 默认五残基 non-standard local-hot；参考端无偏；bond/angle/torsion/improper/CMAP/nonbonded/exception 的完整原子尺度几何平均；增加 hot-heavy RMSD、统一评价、哈希化 full checkpoint、RNG/速度续算、DCD 时间与 PBC unit-cell 修复。 | **当前推荐** |

v0–v2 为早期历史版本，包含旧副本数、旧输入路径和已被修正的恢复/缩放实现，不再推荐用于新实验。

## 12. 常见问题

### 12.1 相邻交换率为 0 或很低，是否因为温差太小？

通常不是。“温差太小”一般会提高而不是压低相邻接受率。低交换率更常见的原因是相邻 Hamiltonian 差异过大、势能分布重叠不足、局部缩放过强、温标/副本布置不合适、体系尚未平衡，或交换能量计算/配置不一致。

先检查：

1. 所有相邻 pair 是否都有交换尝试；
2. 数值能量是否有限；
3. 最低交换率对应哪个 pair；
4. `T_max`、副本数和 `lambda_hot` 是否使相邻 Hamiltonian 梯度过陡；
5. 是否错误混用了不同版本、输入或 checkpoint。

不要仅通过减少副本数来“修复”低交换率；减少副本通常会增大相邻 Hamiltonian 间隔，可能使问题更严重。

### 12.2 1 ns 没有 round trip，是否失败？

不是。对 16 个槽位，1 ns 预实验主要用于安全和运行质量检查。v4.4 明确把短于 10 ns 且 0 round trip 标为时长不足。应在正式长度上结合槽位覆盖、端点访问、往返次数和占用均匀性评价混合。

### 12.3 能否把 16×100 ns 写成 1.6 μs 轨迹？

只能称为 1.6 μs 聚合积分量，不能称为一条连续 1.6 μs 轨迹。每个 Hamiltonian 槽位是 100 ns，walker 会在槽位间迁移。

### 12.4 为什么 VMD 只显示一部分体系或报原子数不匹配？

最常见原因是先加载了仅蛋白 PDB、旧 CHARMM-GUI 拓扑或原子数不同的结构。请使用当前 `step5_charmm2omm.psf`/`step5_charmm2omm.pdb`，确认拓扑与 DCD 均为 68,411 原子，再加载轨迹。

### 12.5 为什么 make-whole 后分子仍在盒外？

make-whole 只保证同一分子的共价连续性，不保证整个分子被放在主盒内。显示需要时再执行 `pbc wrap -compound fragment`。

### 12.6 哪个 checkpoint 能恢复？

只使用 v4.4 生成的 `checkpoint_*_full.npz`。旧的非 full checkpoint 或早期版本 checkpoint 不应作为 v4.4 正式续算起点。

### 12.7 运行需要多久？

实际耗时高度依赖 GPU、驱动、OpenMM 平台、并行上下文、磁盘和分析阶段，README 不给出跨硬件的固定估时。先运行短功能测试或 1 ns 预实验，用本机 `experiment.log` 的实测墙钟时间和吞吐规划正式任务。

### 12.8 `own_state_energies.npz` 能否直接做 MBAR？

不能。它只保存每帧在自身 Hamiltonian 下的势能，没有完整的交叉态 reduced potential `u_kn` 和样本计数 `N_k`。需要额外计算每个样本在所有目标 Hamiltonian 下的能量，并正确处理相关性和采样来源后才能进行 MBAR。

## 13. 参考文献

1. Wang, L.; Friesner, R. A.; Berne, B. J. **Replica Exchange with Solute Scaling: A More Efficient Version of Replica Exchange with Solute Tempering (REST2).** *J. Phys. Chem. B* **2011**, *115*, 9431–9438.
2. Eastman, P. et al. **OpenMM 7: Rapid Development of High Performance Algorithms for Molecular Dynamics.** *PLoS Comput. Biol.* **2017**, *13*, e1005659.
3. Huang, J. et al. **CHARMM36m: An Improved Force Field for Folded and Intrinsically Disordered Proteins.** *Nat. Methods* **2017**, *14*, 71–73.

五残基 local-hot 方案应在论文或报告中明确写作“本项目在 REST2 基础上的 non-standard 方法学扩展”，并给出本 README 第 4.2–4.3 节的端点锚定公式和逐力项缩放规则；不得将其表述为 Wang、Friesner 与 Berne 原始 REST2 算法本身。

## 14. 许可

本项目仅供学术研究使用。使用实验结果时应保留版本、输入文件、`config.json`、随机种子和完整评价限制，以支持复现与审计。
