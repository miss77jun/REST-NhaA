# v4.0 变更日志 (2026-07-30)

## 背景

用户反馈实验后蛋白结构与输入文件差异过大。经分析 `experiment_20260724` 输出:
- 三副本 RMSD ~5 Å (蛋白确实变形)
- Rg 仅变化 2% (尺寸未变, 但构象重排)
- 主要原因: `lambda_hot=0.5` 在 T_max=350K 时等效温度达 700K, 过于激进

## 核心改动

### 1. [关键] lambda_hot 默认值 0.5 → 0.75

| lambda_hot | T_eff @ 300K | T_eff @ 350K | 评价 |
|---|---|---|---|
| 0.5 (v3.3) | 600K | **700K** | 过于激进, 蛋白过度变形 |
| 0.75 (v4.0) | 400K | **467K** | 适中, 增强采样与稳定平衡 |
| 1.0 | 300K | 350K | 无局部加热 |

- 新增 `--lambda_hot` CLI 参数
- 添加安全检查: `lambda_hot < 0.5` 时发出警告

### 2. [新增] 残基编号映射记录

OpenMM 加载 PDB 时保留原始残基 ID (10-390), 但写出 PDB 时自动重编号为 1-381。
v4.0 在以下位置记录映射关系:
- `experiment.log` (启动时)
- `config.json` (`residue_offset` 字段)
- `replica_X_final.pdb` 的 REMARK 记录
- `scientific_report.json`

热残基映射:
```
输入 P72  -> 输出 63
输入 A167 -> 输出 158
输入 I168 -> 输出 159
输入 F344 -> 输出 335
输入 I345 -> 输出 336
```

### 3. [新增] RMSD 实时监控

- 每 100 次迭代计算蛋白 CA 原子 RMSD (Kabsch 对齐后)
- 监控副本: 0 (300K), 15 (中温), 31 (350K)
- 输出:
  - `rmsd_history.npy` - 完整 RMSD 历史
  - `figures/fig_rmsd_evolution.png` - 演化图
  - `experiment.log` - 实时日志
  - `scientific_report.json` - 最终统计

### 4. [优化] make_whole 性能

- 跳过小分子 (水、离子, <10 原子)
- 仅对蛋白和脂质执行 make_whole
- 预期性能提升 3-5 倍

### 5. [新增] CLI 参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--lambda_hot` | 0.75 | 局部加热 lambda |
| `--no_local_heating` | False | 禁用局部加热 |
| `--equilibration_restraints` | False | 平衡阶段位置约束 |

### 6. [改进] 输出 PDB REMARK

`replica_X_final.pdb` 头部新增:
```
CRYST1   82.407   82.407  109.362  90.00  90.00  90.00 P 1           1
REMARK   1 CREATED WITH OPENMM (REST2 v4.0)
REMARK   2 RESIDUE MAPPING: output_resid = input_resid - 9
REMARK   3 HOT RESIDUES (input -> output): F72->63, A167->158, ...
REMARK   4 LAMBDA_HOT=0.75, T_EFF_MAX=467K
```

## 使用示例

```bash
# 默认 (lambda_hot=0.75)
python versions/rest2_4au5_v4.0.py

# 更温和的局部加热
python versions/rest2_4au5_v4.0.py --lambda_hot 0.8

# 无局部加热 (纯全局 REST2)
python versions/rest2_4au5_v4.0.py --no_local_heating

# 带平衡约束
python versions/rest2_4au5_v4.0.py --equilibration_restraints

# 从检查点恢复
python versions/rest2_4au5_v4.0.py --restore outputs/exp/checkpoint_500_full.npz \
    --output_dir outputs/exp --start_iteration 500
```

## 兼容性

- v4.0 可恢复 v3.3 的检查点 (向后兼容)
- v3.3 不能恢复 v4.0 的检查点 (rmsd_history 字段)
- 配置文件格式兼容
