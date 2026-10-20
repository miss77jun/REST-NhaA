# Debug Session: vmd-empty-protein
- **Status**: [OPEN]
- **Issue**: VMD加载DCD后蛋白质不可见且周期盒中央为空
- **Debug Server**: http://127.0.0.1:7777/event
- **Log File**: .dbg/trae-debug-log-vmd-empty-protein.ndjson

## Reproduction Steps
1. 使用当前v4.0脚本生成DCD。
2. 在VMD中载入结构拓扑和对应replica DCD。
3. 观察蛋白质不可见、盒中央为空。

## Hypotheses & Verification
| ID | Hypothesis | Likelihood | Effort | Evidence |
|----|------------|------------|--------|----------|
| A | make-whole成键拓扑或原子顺序不匹配 | High | Low | Pending |
| B | enforcePeriodicBox预包裹与make-whole组合错误 | High | Low | Pending |
| C | 蛋白整体位于主盒外而非坐标丢失 | High | Low | Pending |
| D | VMD使用的拓扑与DCD原子顺序/数量不一致 | Medium | Low | Pending |
| E | 写入前蛋白坐标已有NaN、重合或异常跨度 | Medium | Low | Pending |

## Log Evidence
Pending.

## Verification Conclusion
Pending.
