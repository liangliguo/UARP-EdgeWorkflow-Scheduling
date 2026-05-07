# UARP 论文复现 TODO

> 论文：Xu X. et al. *Dynamic resource provisioning for workflow scheduling under uncertainty in edge computing environment*. CCPE 2020, e5674. DOI: 10.1002/cpe.5674
> 本文档跟踪复现进度。完成项打勾即可。

## 0. 技术栈与决策

- 语言：**Python 3.11+**
- 包管理：**uv**（`uv init` + `uv add ...`）
- DAG 来源：**随机 DAG**（`networkx.gnp_random_graph` + 强制有向无环 + 固定 seed）
- 多目标优化：**pymoo 的 NSGA-III**（不重写参考点机制；Algorithm 1 仅作为对照实现）
- 绘图：matplotlib
- 单测：pytest

---

## 1. 项目骨架（阶段 1，预计 0.5 天）

- [ ] `uv init` 初始化项目；写 `pyproject.toml`，Python 锁定 3.11+
- [ ] `uv add numpy networkx pymoo matplotlib pandas`
- [ ] `uv add --dev pytest pytest-cov ruff`
- [ ] 目录结构
  - [ ] `src/uarp/model/`        — workflow / edge node / cost 模型
  - [ ] `src/uarp/scheduler/`    — NSGA-III 包装、SAW+MCDM、Algorithm 1/2
  - [ ] `src/uarp/uncertainty/`  — 性能退化 / 服务失败 / 新节点加入事件
  - [ ] `src/uarp/baselines/`    — Benchmark / FF / WF
  - [ ] `experiments/`           — 复现 §4.2、§4.3 的脚本
  - [ ] `figs/`                  — 输出 Figure 5–9
  - [ ] `tests/`                 — 公式单元测试
  - [ ] `notation.md`            — 论文 Table 1 符号对照（中英）
- [ ] `.gitignore`、`README.md`（仅写"复现说明"骨架）

## 2. 系统模型与公式（阶段 2，预计 1 天）

参考论文 §2.2、Table 1。

- [ ] `Workflow`：DAG 容器，含 `predecessors(v)` / `successors(v)`（公式 1–2），任务大小 `b_{m,i}`
- [ ] `EdgeNode`：处理能力 `B_k`、单位执行能耗 `ce_k`、起停/空闲/运行功率 θ_s/θ_id/θ_op
- [ ] `Topology`：移动设备到节点的距离矩阵 `d_{m,i}`，控制器同步矩阵 `h_{ij}`、`cs_{ij}`
- [ ] `Schedule`：决策变量 `x_{m,i,k} ∈ {0,1}`（公式 3）
- [ ] **时间分析**
  - [ ] `OT(v) = b · d / BA`                     (Eq.4)
  - [ ] `ET(v) = b / B_k`                         (Eq.5)
  - [ ] `ST(first) = 0`                           (Eq.6)
  - [ ] `ST(v) = ST(p) + OT(p) + ET(p)`           (Eq.7) — 按拓扑序
  - [ ] `WT(v) = ST(p) + ET(p)`                   (Eq.8)
  - [ ] `WT(V_m)` = 末任务的 WT
- [ ] **能耗分析**
  - [ ] `CMT = ΣΣ x · ct(v, z)`                   (Eq.9)
  - [ ] `CME = ΣΣ x · b · ce(z)`                  (Eq.10)
  - [ ] `CMS = ΣΣ h · cs`                         (Eq.11)
  - [ ] `CM  = CMT + CME + CMS`                   (Eq.12)
- [ ] **Deadline / 成功率**
  - [ ] `DT(V_m) = α · WT(v_M)`                   (Eq.13)
  - [ ] `DT(v_i)` 关键路径插值                    (Eq.14)
  - [ ] `k(af) = 1[rt(af) < DT(v)]`               (Eq.15)
  - [ ] `SUC(V_m) = (1/T) Σ k(af)`                (Eq.16)
- [ ] 单元测试：手工构造 5 任务串行 / 5 任务并行两个小 DAG，验算所有公式

## 3. 编码与适应度（阶段 3，预计 0.5 天）

- [ ] 染色体：长度 `M` 整数向量，第 i 位 ∈ `[0, N-1]` 表示分配节点（§3.1.1）
- [ ] `fit1 = WT(v_M)`、`fit2 = CM(V_m)`
- [ ] 约束 `WT ≤ DT`：在 pymoo 里以 `G(x)` 不等式约束传入

## 4. NSGA-III + SAW/MCDM（阶段 4，预计 1.5 天）

- [ ] `pymoo` 接入：`NSGA3(ref_dirs=das_dennis)`，`crossover=SBX`、`mutation=PolynomialMutation`
- [ ] **复现选择的超参**（论文未给，写进 README 声明）：
  - `pop_size = 100`
  - `n_gen = 100`
  - `p_c = 0.9`
  - `p_m = 1 / M`
- [ ] SAW + MCDM
  - [ ] 归一化（Eq.19–22）：`WT^t = WT - WT_min`、`CM^t = CM - CM_min`，再除极值 σ
  - [ ] 效用 `W = 0.5·WT^n + 0.5·CM^n`（Eq.23），取最小者为最优
- [ ] **Algorithm 1**：完整严格实现一份（与 pymoo 结果对照），主实验仍用 pymoo
- [ ] 输出 Pareto 前沿与候选解效用值（Figure 5 的源数据）

## 5. 基线（阶段 5，预计 0.5 天）

- [ ] **Benchmark**：固定一次分配，不确定性发生后**不**重排（§4.1(1)）
- [ ] **FF (First Fit)**：任务顺序遍历，挑第一个满足资源要求的节点；不确定性后重排
- [ ] **WF (Worst Fit)**：挑剩余资源最多的节点；不确定性后重排
- [ ] 与 UARP 共享同一 DAG / 拓扑 / 不确定性事件 seed，保证公平比较

## 6. 不确定性注入与 Algorithm 2（阶段 6，预计 1 天）

- [ ] `Event` 类型
  - [ ] `performance_degradation`：节点 `B_k *= 随机系数 ∈ [0.3, 0.7]`
  - [ ] `service_failure`：节点不可用直到工作流结束
  - [ ] `new_node_join`：新增一个 z，参与剩余任务调度
- [ ] 触发器：工作流推进到 30%–50% 进度时按 seed 随机触发
- [ ] **Algorithm 2** 流程
  1. SDN 框架捕获异常，通知所有节点
  2. 重算 `DT(V_m)`（基于 `X_n`）
  3. 受影响 + 未执行任务 → 新工作流
  4. 重新跑 Algorithm 1（NSGA-III + SAW/MCDM）→ `X_new`
  5. 计算 `SUC(V_m)`（Eq.16）
- [ ] 事件可重放（保存 seed + 事件序列 JSON）

## 7. 实验复现（阶段 7，预计 1.5 天）

### 7.1 参数（论文 Table 2）

| 参数 | 值 |
|---|---|
| BA | 1000 Mbps |
| VM 处理能力 | 2000 MHz |
| 边缘节点数 N | 20 |
| θ_s（VM 启动） | 0.2 kW |
| θ_id（VM 空闲） | 0.03 kW |
| θ_op（VM 运行） | 0.05 kW |

### 7.2 复现的图

- [ ] **Figure 5**（A–F）：6 种工作流规模（10/15/20/25/30/35）下 UARP Pareto 前沿候选解的效用值柱状图
- [ ] **Figure 6**：completion time vs 任务数，4 种方法
- [ ] **Figure 7**：total energy consumption vs 任务数，4 种方法
- [ ] **Figure 8**：execution energy consumption vs 任务数，4 种方法
- [ ] **Figure 9**：success rate vs deadline 系数 α ∈ {1.1, 1.2, 1.3, 1.4}，4 种方法

### 7.3 实验执行

- [ ] 每个配置跑 ≥ 30 次取均值（论文未明示，复现需写明）
- [ ] 输出 CSV 到 `experiments/results/`，绘图脚本读 CSV
- [ ] `experiments/run_all.py` 一键复现全部图

---

## 8. 已知风险与说明（写进最终 README）

1. **DAG 形态未公布**：使用随机 DAG，固定 seed；非论文官方设定
2. **缺失超参**：`pop_size, n_gen, p_c, p_m, b_{m,i}, d_{m,i}, ct/ce/cs` 均自定义，README 声明
3. **公式 14 的写法歧义**：以小型关键路径手算结果为准
4. **Algorithm 1 与 pymoo 数值可能不一致**：保留两份实现，互相校验
5. **无作者源码**：目标是**定性结论一致**
   - 完成时间 / 总能耗 / 执行能耗：UARP < FF ≈ WF < Benchmark
   - 成功率：UARP > FF > WF > Benchmark（α 越小差距越大）

---

## 9. 进度跟踪

| 阶段 | 状态 | 说明 |
|---|---|---|
| 1 项目骨架 | ☐ | |
| 2 系统模型 | ☐ | |
| 3 编码与适应度 | ☐ | |
| 4 NSGA-III + SAW | ☐ | |
| 5 基线 | ☐ | |
| 6 不确定性 + Alg.2 | ☐ | |
| 7 实验复现 | ☐ | |

合计预估：**5–6 工作日**。
