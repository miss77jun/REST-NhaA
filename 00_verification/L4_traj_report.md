# L4 轨迹层分析报告 (2026-10-22, 修订版)

数据: 学生提供的 replica_stride100.zip — **确认来自 v4.3 标准模式** (无 local-hot)
传输: zip MD5 两端一致 (fc1d7ecf), 但 replica_0_stride100.dcd 在微信传输源头损坏 (bad CRC), 待重传
分析: 00_verification/l4_traj_analysis.py, 拓扑 step5_charmm2omm.psf, omm760/MDAnalysis 2.0.0

## 口径
- OF-gate Rg: segid PROA resid {72,167,168,344,345} 侧链重原子 (23 个, 与 com_rg_dist.py 等价;
  MDAnalysis 2.0.0 "type H*" 通配符选择触发 segfault, 改 numpy 按名过滤)
- RMSD: 全蛋白 CA, 逐帧 Kabsch 对齐到各自首帧
- 每槽位 51 帧; 时间轴假设 stride=100 迭代 × 1000 步 × 2 fs = 200 ps/帧 → ~10.2 ns/槽位 (待确认)

## 结果 (slot 1-15, slot 0 缺失)

| 指标 | 结果 |
|---|---|
| Rg 范围 | 4.60-4.70 Å (各槽位 SEM ~0.03 Å) |
| Rg vs 溶质有效温标 (300-350 K) | slope = +0.18 mÅ/K, R² = 0.012 |
| CA RMSD (对齐后) | 1.8-2.0 Å, 各槽位相近 |

## 解读: v4.3 无响应 — 与既有 v4.3 结论一致 ✓

标准 REST2 下门区只扫 300→350 K, Rg 无温度响应 (首尾槽位 4.649 vs 4.654 Å, 差 0.005 Å << SEM 0.03 Å)。
这与学生 v4.3 50 ns 对照的结论定量吻合 (slope -0.145 mÅ/K, R²=0.185 — 同样无响应):
两次独立数据 (50 ns 产物 vs 本次 ~10 ns/槽位) 互相印证, v4.3 标准模式对该门区观测量无响应是稳健结论。

**注意: 不可与 v4.4 local-hot pilot 的 R²=0.809 直接对比** — 那是不同变体
(门区有效温标拉到 437.5 K)。本结果不否定 v4.4 扩展, 反而与其设计动机自洽:
标准 REST2 的温标窗口不足以调制门区, local-hot 把窗口扩大 3 倍才出现响应。

## 对验证的意义

L4 的"哈密顿量响应"半层在 v4.3 体系上不构成放行/否决项 (无响应是已知且可解释的行为)。
最终判定仍取决于交换统计 (npz) — 待学生补充:
- replica_0_stride100.dcd 重传
- rest2_results.npz + experiment.log (交换接受率 / round trip / walker mixing / run 配置)
- 确认 stride 与 n_steps_per_move

## 初版报告的更正说明

初版误将本数据与 v4.4 pilot 对照并得出"矛盾"结论; 经用户确认数据来自 v4.3 后修正。
方法学教训: pilot 对照必须先核对变体/配置, 不同变体的 Rg-T 响应不可比。
