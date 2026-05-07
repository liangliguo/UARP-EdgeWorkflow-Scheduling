# 符号对照表（论文 Table 1）

| 符号 | 含义 | 代码字段 |
|---|---|---|
| V | 工作流中的计算任务集合 | `Workflow.tasks` |
| M | 任务数量 | `len(Workflow.tasks)` |
| N | 边缘节点数量 | `len(Topology.nodes)` |
| v_{m,i} | 第 m 个工作流的第 i 个任务 | `Task` |
| p(v) | 前驱任务集合（公式 1） | `Workflow.predecessors(v)` |
| s(v) | 后继任务集合（公式 2） | `Workflow.successors(v)` |
| x_{m,i,k} | 调度策略：任务 v_{m,i} 是否分配到节点 z_k（公式 3） | `Schedule.assignment` |
| WT(V_m) | 工作流完成时间（公式 8） | `cost.completion_time(...)` |
| CM(V_m) | 工作流总能耗（公式 12） | `cost.total_energy(...)` |
| SUC(V_m) | 工作流成功率（公式 16） | `cost.success_rate(...)` |
| DT(V_m) | 工作流截止时间（公式 13） | `cost.deadline(...)` |

## 派生量

| 符号 | 含义 | 公式 |
|---|---|---|
| OT(v_{m,i}) | 传输时间 | (4) `b · d / BA` |
| ET(v_{m,i}) | 执行时间 | (5) `b / B_k` |
| ST(v_{m,i}) | 开始时间 | (6)(7) |
| CMT(V_m) | 传输能耗 | (9) |
| CME(V_m) | 执行能耗 | (10) |
| CMS(V_m) | 控制器同步能耗 | (11) |
| α | deadline 系数 | (13) |

## 实验参数（论文 Table 2）

| 参数 | 值 |
|---|---|
| BA | 1000 mbps |
| VM 处理能力 | 2000 MHz |
| 边缘节点数 N | 20 |
| θ_s（VM 启动） | 0.2 kW |
| θ_id（VM 空闲） | 0.03 kW |
| θ_op（VM 运行） | 0.05 kW |
