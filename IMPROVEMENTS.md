# UARP 复现的改进规划

本文件记录在 `improvements` 分支上计划补强的内容,起因是原论文(Xu et al. 2020, CCPE e5674)的算法/模型在真实移动边缘场景下存在若干局限。每条改动均给出**问题来源、解决思路、代码改动点、验证方式、工作量**,按落地优先级排序。

> 与 `TODO.md` 的区别:`TODO.md` 是论文原始复现的阶段计划;本文是**超出论文范围**的扩展与修正。

---

## 问题分类总览

| 编号 | 问题 | 类别 | 优先级 |
|---|---|---|---|
| #1 | 节点忙等约束缺失(同节点任务可并行) | 模型 bug | P1 |
| #2 | NSGA-III 用于 2 目标 | 算法选型 | P1 |
| #3 | SAW MCDM 假设线性偏好 | 决策 | P1 |
| #4 | 设备移动性未建模 | 模型 | P2 |
| #5 | 事件序列退化为单次扰动 | 模型 | P2 |
| #6 | 事件类型缺失(带宽抖动 / 队列拥塞 / 任务体积漂移) | 模型 | P2 |
| #7 | 确定性成本 vs 随机现实 | 算法 | P3 |
| #8 | α-deadline 是粗粒度 buffer | 算法 | P3 |
| #9 | Eq. 14 任务级 deadline 公式 ill-posed | 论文公式硬伤 | P3 |
| #10 | 仅做一次重调度,无 rolling horizon | 算法 | P4 |
| #11 | 数据再传输代价未计 | 模型 | P4 |
| #12 | 服务失效靠 `capacity*=0.05` 近似 | 工程妥协 | P4 |
| #13 | 单链路共享带宽,无每链路差异 | 模型 | P4 |
| #14 | 能耗模型缺 idle / 设备侧 / SNR 项 | 模型 | P5 |
| #15 | 单工作流假设,无跨工作流调度 | 范围 | P5 |
| #16 | 纯 reactive,无失效预测 | 算法 | P5 |

---

## P1:调度模型修正 + 决策改进(半天内可完成)

### #1 节点忙等约束

**问题**。`src/uarp/model/cost.py:30` 的 `schedule_times()` 只考虑 DAG 前驱完成 + 数据传输完成,**没有"同一节点上两个任务必须串行"的约束**。等价于把每个 edge node 视为无限并行,会系统性低估 WT,且让"全部压到最快节点"的策略不公平地占便宜。

**方案**。改为 list-scheduling,维护每节点的下一次空闲时刻:

```python
def schedule_times(wf, topo, sched):
    node_free = np.zeros(topo.N)
    ST = np.zeros(wf.M); WT = np.zeros(wf.M)
    for i in wf.topo_order():
        k = sched.node_of(i)
        ot_i = transmission_time(wf, topo, sched, i)
        et_i = execution_time(wf, topo, sched, i)
        pred_ready = max((WT[p] for p in wf.predecessors(i)), default=0.0)
        ST[i] = max(pred_ready, node_free[k]) + ot_i
        WT[i] = ST[i] + et_i
        node_free[k] = WT[i]
    return ST, WT
```

**代码改动**:
- `src/uarp/model/cost.py:30-54`
- 受影响测试:`tests/test_model.py::test_serial_schedule_times`, `test_parallel_schedule_times`(需更新预期值)

**新增测试**:同节点 + 无依赖的 2 任务,断言两任务串行而非并行完成。

**预期影响**。Figure 6/7/8 的绝对数值变化,但相对排序(UARP < FF < WF < Benchmark)应保留。需重跑 `experiments/run_all.py` 并更新 README"复现趋势"段落。

---

### #2 NSGA-II 替换 NSGA-III(2 目标场景)

**问题**。NSGA-III 的 reference direction 机制是为 ≥3 目标设计的;2 目标场景下 NSGA-II 在收敛速度和 Pareto front 多样性上都更优。

**方案**。

```python
# src/uarp/scheduler/uarp.py
from pymoo.algorithms.moo.nsga2 import NSGA2
algo = NSGA2(pop_size=pop_size, sampling=..., crossover=..., mutation=...)
```

**约束**。引入 #14(device energy 第 3 目标)后切回 NSGA-III。

**验证**。`tests/test_scheduler.py` 确认 Pareto front 大小 ≥ 2(异构拓扑上)。

---

### #3 Knee-point 替换 SAW

**问题**。`src/uarp/scheduler/saw_mcdm.py` 用线性加权,对非凸 Pareto front 容易选端点而漏掉 knee;权重又需要外部指定,缺乏依据。

**方案**。增加 knee-point selector(到归一化 ideal point 的最近 L2 距离):

```python
def knee_point(F: np.ndarray) -> int:
    F_n = (F - F.min(0)) / (F.ptp(0) + 1e-9)
    return int(np.argmin(np.linalg.norm(F_n, axis=1)))
```

`solve()` 暴露 `selector="saw" | "knee" | "topsis"`,默认 `"knee"`;旧调用方保留兼容。

---

## P2:移动性 + 不确定性扩展(1-2 天)

### #4 时变距离(MobilityModel)

**问题**。`Topology.distances[k]` 是常量,但移动 MEC 场景下距离随时间变化,直接影响 OT(Eq. 4)和 CMT(Eq. 9)。

**方案**。

- 新建 `src/uarp/uncertainty/mobility.py`:
  ```python
  @dataclass
  class MobilityTrace:
      times: np.ndarray            # (T,)
      distances: np.ndarray        # (T, N)
      def distance_at(self, t: float, k: int) -> float: ...
  ```
- 提供两种生成器:`linear_walk`(沿直线匀速)、`random_waypoint`(RWP 模型,经典 mobility model)。
- `Topology` 增加可选字段 `mobility: MobilityTrace | None`;若为 `None`,退化为常量距离。
- `cost.transmission_time(wf, topo, sched, i, t_start)` 增加 `t_start` 形参;`schedule_times()` 内部传入。

**新建测试** `tests/test_mobility.py`:两节点 + 直线运动,断言 OT(t=0) < OT(t=终点) 当设备远离节点时。

---

### #5 + #6 NHPP 事件序列 + 新事件类型

**问题**。`generate_events(n_events=1)` 把一个 stochastic process 退化成单次扰动;且只覆盖 3 种节点侧事件。

**方案**。

- 改造 `generate_events()`:接受 `rate_fn: Callable[[float], float]` 和 `T_horizon`,按 NHPP(非齐次泊松)采样事件时间戳;旧的 `n_events` 接口保留为捷径。
- `Event` 加 `time: float` 字段,按时间排序后再 apply。
- 新增事件类型:
  - `bandwidth_jitter`: `BA *= factor`,持续 `duration`,过期自动回滚(用 `expiry: float`)
  - `queue_contention`: `capacity *= factor`,带 `duration`(代表邻居工作流抢占)
  - `task_size_drift`: 给 `Workflow.sizes` 注入乘性噪声(执行前才显现)

**代码改动**:`src/uarp/uncertainty/events.py`,`Event.apply` 改为支持带时序的 apply/revert。

---

## P3:鲁棒优化与正确的 deadline 公式(2-3 天)

### #7 + #8 Scenario-based CVaR 第一阶段

**问题**。论文 NSGA-III 只在确定性 topology 上优化,把"应对不确定性"留给事后 reschedule——这是 reactive,不是题目暗示的 *provisioning*。

**方案**(不重写 NSGA,只改 evaluation):

```python
# src/uarp/scheduler/problem.py
def _evaluate(self, X, out, *args, **kwargs):
    S = self.n_scenarios            # 默认 10-20
    wt_per_x = np.zeros((len(X), S))
    cm_per_x = np.zeros((len(X), S))
    for s in range(S):
        topo_s = perturb(self.topology, self.rng_scenarios[s])
        for j, x in enumerate(X):
            sched = Schedule(np.round(x).astype(int))
            wt_per_x[j, s] = completion_time(self.workflow, topo_s, sched)
            cm_per_x[j, s] = total_energy(self.workflow, topo_s, sched)
    # 用 CVaR_0.9 作为 robust 目标
    out["F"] = np.stack([cvar(wt_per_x, 0.9), cvar(cm_per_x, 0.9)], axis=1)
```

`perturb()` 复用 `events.generate_events` 的样本路径。S 维度可在 `experiments/config.py` 暴露。

**额外**。`DT = quantile_0.9(WT_scenarios)` 取代 Eq. 13 的 `α·WT_ideal`,消去 α 这一魔术参数。Figure 9 横轴改为"风险偏好分位数"。

---

### #9 重写 Eq. 14 任务级 deadline

**问题**。论文 Eq. 14 写的 `(WT(v_q)-ST(v_p))·(WT(v_i)-ST(v_p)) / (WT(v_i)-ST(v_p))` 分子分母同形,ill-posed。复现版只能猜一个解释(见 `cost.py:111-124`)。

**方案**。改用经典 **Slack-Time Allocation**(Hwang et al. 1989):

```
slack = α · WT_max - WT_critical_path
DT[i] = WT_ideal[i] + slack · (path_len_through_i / total_path_len)
```

**代码改动**:`src/uarp/model/cost.py:103-124`。
**测试**:线性 DAG 上 slack 在各任务间均匀分配。

---

## P4:Rolling horizon + 数据迁移 + 链路差异化(3-5 天)

### #10 Rolling-horizon Algorithm 2

**问题**。`reschedule()` 在 `progress_frac=0.4` 处仅触发一次。

**方案**。

- 把 `reschedule()` 改为事件驱动循环:`for ev in events_sorted_by_time: replan(now=ev.time, state=...)`。
- `state` 维护:已完成任务集合、每节点已缓存的 output 数据、当前 mobility 位置。
- `replan` 用缩小的 NSGA(`pop_size=30, n_gen=20`),并计入 `replan_latency` 作为新指标——动作生效前旧分配仍在跑,这部分时延要扣到总 WT。
- 实验:新增 `figure10_rolling.py`,横轴 = 事件率 λ,曲线 = {UARP-once, UARP-RH} 的 success rate / replan_cost。

---

### #11 数据再传输代价

**问题**。事件触发后只重排剩余任务的节点;若已完成任务的输出原本落在现失效/降级节点上,后续任务的 OT 应重新评估或加迁移代价。

**方案**。

- `Schedule` 隐式 + 显式维护 `output_location: dict[task_id, node_idx]`。
- 重调度时:若后继任务的某前驱输出在失效节点,该数据需重传——计入额外 OT,起点是 fallback 节点。
- 体现在 `cost.transmission_time()` 中:OT(i) 改为对**每个前驱**单独累加(不再只用 i 的 size),反映"取数据"的真实代价。

---

### #12 真正的服务失效语义

**问题**。当前用 `capacity *= 0.05`(`src/uarp/uncertainty/events.py:47`)规避 inf。

**方案**。

- `service_failure` 恢复为 `available=False`。
- Benchmark 改为报告 **"违反 deadline 的任务比例"** 而非 `WT`,从而允许 inf 任务存在而不破坏对比。
- `reschedule_benchmark` 里的 fallback 修复(`algorithm2.py:191-197`)删除——Benchmark 字面意义就是"不修复",fallback 反而给它送了免费红利,污染 Figure 9。

---

### #13 每链路带宽 + 拥塞

**问题**。`Topology.BA` 是全局标量。

**方案**。

- `Topology` 增加 `link_bw: ndarray(N,)`,默认初始化为 `BA`。
- `schedule_times` 维护 `link_busy[k]`:OT 占用链路 k 期间,其他面向 k 的 OT 必须排队。
- 这是 fluid 近似——比 packet-level 仿真便宜 2 个数量级,够分辨"热门链路 vs 闲置链路"。

---

## P5:能耗 + 多工作流 + 预测(1-2 周,可独立成稿)

### #14 能耗补全

- `EdgeNode.p_idle: float`(W);执行能耗加 `p_idle · (节点 last_busy - first_arrive)`。
- 新增**设备侧能耗**(终端电池):`device_energy = Σ tx_power · OT(i)`,作为 NSGA 的**第 3 个目标**。
- 第 3 目标后切回 NSGA-III(见 #2)。
- 传输能耗从 `size·distance·const` 改为 `size · d^γ · const`(γ ∈ [2,4],path-loss 指数)。

---

### #15 多工作流共享 edge

- 新建 `src/uarp/multitenant/`:
  - `MultiWorkflowSchedule`:扩展 `Schedule`,记录 (workflow_id, task_id) → node_idx。
  - `schedule_times_mt()`:跨工作流共享 `node_free` 与 `link_busy`(直接复用 #1 和 #13 的实现)。
- 外层 priority queue 按 deadline 紧迫度调度;内层每个 workflow 调用单 workflow UARP。
- 新增 `figure11_multitenant.py`:横轴 = 并发工作流数,曲线 = {UARP-MT, 静态 FCFS, EDF} 的成功率。

---

### #16 失效率预测器

- 维护历史事件日志 `experiments/results/events_history.jsonl`。
- 用滑动窗口 MLE 估计每节点失效率 `λ_fail(k)`(或 1 阶 HMM)。
- NSGA 目标加软惩罚 `Σ_i λ_fail(node_of(i)) · ET(i)`(被高失效率节点占用越久越糟)。
- 这是 *proactive provisioning* 的最简形式,呼应论文标题里的 "dynamic resource provisioning"。

---

## 落地时间线建议

| 阶段 | 内容 | 产出 |
|---|---|---|
| P1(0.5d) | #1 + #2 + #3 | 重跑 Figure 6/7/8,数字更可信;README 更新趋势段 |
| P2(1-2d) | #4 + #5 + #6 | 新 Figure 10:success vs. mobility speed |
| P3(2-3d) | #7 + #8 + #9 | 替换 α-deadline,Figure 9 改为风险分位数曲线 |
| P4(3-5d) | #10 + #11 + #12 + #13 | 新 Figure 11:replan_cost vs. event_rate |
| P5(1-2w) | #14 + #15 + #16 | 论文级扩展(3 目标 + 多工作流 + 预测),可独立投稿 |

---

## 工作流约定

- 每个 P 阶段独立 PR 合入 `improvements` 分支;P1 全部 merge 后再开 P2 子分支。
- 每条改动伴随:**单元测试 + 受影响 figure 重跑 + README 对应段落更新**。
- 所有新引入的可调参数集中在 `experiments/config.py`,严禁散落。
- 模型层(`src/uarp/model/`)的语义变化需要在 `notation.md` 同步更新。
