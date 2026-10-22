# L4 轨迹层分析报告 (2026-10-22)

数据: 学生提供的 replica_stride100.zip (16 槽位 DCD, 每条 51 帧, stride=100 迭代)
传输: zip MD5 两端一致 (fc1d7ecf), 但 replica_0_stride100.dcd 在微信传输源头损坏 (bad CRC), 已删除待重传
分析: 00_verification/l4_traj_analysis.py, 拓扑 step5_charmm2omm.psf, omm760/MDAnalysis 2.0.0

## 口径
- OF-gate Rg: segid PROA resid {72,167,168,344,345} 侧链重原子 (23 个, 与学生 com_rg_dist.py 等价;
  MDAnalysis 2.0.0 "type H*" 通配符选择触发 segfault, 改 numpy 手工过滤)
- RMSD: 全蛋白 CA, 逐帧 Kabsch 对齐到各自首帧
- 每槽位帧数 51; 时间轴假设 stride=100 迭代 × 1000 步 × 2 fs = 200 ps/帧 → ~10.2 ns/槽位 (待学生确认)

## 结果 (slot 1-15, slot 0 缺失)

| 指标 | 结果 |
|---|---|
| Rg 范围 | 4.60-4.70 Å (各槽位 SEM ~0.03 Å) |
| Rg vs 溶质有效温标 (300-350 K) | slope = +0.18 mÅ/K, R² = 0.012 |
| Rg vs 门区有效温标 (300-437.5 K) | slope = +0.06 mÅ/K, R² = 0.012 |
| CA RMSD (对齐后) | 1.8-2.0 Å, 各槽位相近 |

## 主要发现: 门区 Rg 无温度响应

15 个 Hamiltonian 槽位的 OF-gate Rg 统计上不可区分 (R²=0.012, 斜率为正且远小于 pilot 的 -2.77 mÅ/K)。
这与 v4.4 1 ns pilot 的 R²=0.809 (Rg 4.70→4.60 Å) 明显矛盾。

解释 (按可能性排序):
1. pilot 的 Rg-T 相关是短时瞬态/自相关假象 (学生评估笔记已注明该风险: 槽位间非独立重复)
2. 平衡态下该五残基侧链堆积对 300→437.5 K 不敏感 — Rg 不是门区构象变化的敏感标记物
3. 本数据可能来自 v4.3 标准模式 (无 local-hot) — 待学生确认 run 配置

## L4 缺口 (需学生补充)
- replica_0_stride100.dcd 重传 (300 K 参考端)
- rest2_results.npz + experiment.log: 交换接受率/round trip/walker mixing 与 run 版本配置
- 确认 stride 与 n_steps_per_move, 锁定每帧时间与总时长

## 结论
轨迹层统计质量足够 (15×51 帧), 门区 Rg 温度响应在平衡数据中不存在。
在拿到交换统计 (npz) 之前, 无法给出 REST2 采样效率的最终判定。
