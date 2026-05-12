# UARP 复现的改进规划

本文件记录在 `improvements` 分支上计划补强的内容,起因是原论文(Xu et al. 2020, CCPE e5674)的算法/模型在真实移动边缘场景下存在若干局限。本轮聚焦两大方向——**不确定性建模过于简化**(对"移动"场景最致命)与**确定性成本模型 vs 随机现实**。每条改动均给出问题来源、解决思路、代码改动点、验证方式、工作量。

> 与 `TODO.md` 的区别:`TODO.md` 是论文原始复现的阶段计划;本文是**超出论文范围**的扩展与修正。

---

## 总体问题与解决方案

**问题类型**字段说明:

- **建模简化** — 物理/系统层面的过度理想化(节点、链路、能耗、数据迁移)
- **不确定性薄弱** — 对随机事件、风险尾部的处理过于粗糙或缺失
- **论文/实现缺陷** — 论文公式硬伤或复现时为绕开问题做的工程妥协

| 编号 | 问题类型 | 问题描述 | 解决方案(一句话) | 类别 | 优先级 | 改动位置 |
|---|---|---|---|---|---|---|
| 一.1 | 建模简化 | 没有移动性建模。`Topology.distances[k]` 是每节点一个常量(`edge.py:30`),设备—节点的距离从头到尾不变。真实移动场景下距离连续变化(OT、ct 都该是时间的函数),跨基站会发生 handover / service migration(状态搬运、缓存重建有迁移成本);且论文只考虑节点侧三种事件,没有用户侧的任何事件——而 mobile edge 的主要 uncertainty 恰恰来自用户侧。 | 引入 `MobilityTrace`(`linear_walk` + `random_waypoint` 两种生成器);`transmission_time()` 接受 `t_start`;增加 handover 触发与迁移代价。 | 模型 | P1 | `uncertainty/mobility.py`(新) / `model/cost.py` / `model/edge.py` |
| 一.2 | 不确定性薄弱 | 每次仅注入一个事件,且时刻固定。`generate_events` 默认 `n_events=1`,`reschedule()` 在 `progress_frac=0.4` 处一次性触发(`algorithm2.py:88`)。现实里事件是连续到达的点过程(Poisson / 突发),论文相当于把一个 stochastic process 退化成一次性扰动。 | 用 NHPP(非齐次泊松)采样事件时间戳;`Event` 增加 `time` 字段;`reschedule()` 改为按时间顺序逐事件触发。 | 模型 | P1 | `uncertainty/events.py` / `uncertainty/algorithm2.py` |
| 一.3 | 不确定性薄弱 | 事件类型缺失。链路带宽 BA 抖动、节点队列拥塞、邻居工作流抢占、电池耗尽、无线 PHY 速率下降、任务体积/计算量本身的不确定(`Workflow.size(i)` 也是常量)——都未建模。 | 增加 `bandwidth_jitter`、`queue_contention`、`task_size_drift` 三类事件,支持 `duration` 与到期自动回滚;`Workflow.size` 改为可附带噪声。 | 模型 | P2 | `uncertainty/events.py` / `model/workflow.py` |
| 一.4 | 论文/实现缺陷 | `service_failure` 的工程妥协。README 第 46–50 行直接承认:若真把节点设成 `available=False`,Benchmark 一旦命中失效节点就 inf,无法对比,所以把失效改写成 `capacity *= 0.05`(`events.py:47`)。这暴露 UARP 框架本身无法处理硬失效——既然算法叫"动态资源供给",对硬不可用就该有迁移/恢复策略,而不是降级近似。 | 恢复 `available=False` 真失效语义;Benchmark 改为报告"违反 deadline 任务比例"而非 WT;`reschedule()` 增加"已完成任务的中间结果迁移"分支,即明确的恢复策略。 | 工程妥协 | P2 | `uncertainty/events.py` / `uncertainty/algorithm2.py` |
| 二.5 | 不确定性薄弱 | 所有代价(OT/ET/CMT/CME)都是点估计。论文不是 robust / stochastic optimization,而是"先用确定模型求 NSGA-III,事件触发后再求一次"——这是 reactive,不是题目暗示的 *provisioning*。真正的供给应在第一次调度时就对不确定性建模并加入约束(例如 chance-constrained 或 CVaR)。 | scenario-based 鲁棒评估:每次 `_evaluate` 在 S 个扰动 topology 上算 WT/CM,以 CVaR₀.₉ 作为目标。 | 算法 | P3 | `scheduler/problem.py` |
| 二.6 | 不确定性薄弱 | α-deadline(Eq. 13)是个粗粒度 buffer。`DT = α · WT_ideal`,α 取 1.1–1.4(Figure 9 横轴),完全不区分任务的尾部风险。两个完成时间相同的方案,若一个方差大、一个小,这里给同样的 buffer——这违反 mobile 场景的直觉。 | 用 `quantile_0.9(WT_scenarios)` 取代 α-deadline;Figure 9 横轴换成风险分位数。 | 算法 | P3 | `model/cost.py` / `experiments/figure9_*.py` |
| 二.7 | 论文/实现缺陷 | Eq. 14 任务级 deadline 本身写错了。`cost.py:111-124` 的注释和公式都点出:论文写的 `(WT(v_q)-ST(v_p))·(WT(v_i)-ST(v_p)) / (WT(v_i)-ST(v_p))` 分子分母自相消,是 ill-posed 的;复现只能猜一个"沿关键路径线性分配 slack"的解释。这是论文形式化层面的硬伤。 | 改用经典 Slack-Time Allocation(Hwang et al. 1989):`DT[i] = WT_ideal[i] + slack · (path_len_through_i / total_path_len)`。 | 论文公式硬伤 | P1 | `model/cost.py:103-124` |

---

## 一、不确定性建模过于简化(对"移动"场景最致命)

### 一.1 时变距离与 handover(MobilityModel)

**方案**。

- 新建 `src/uarp/uncertainty/mobility.py`:
  ```python
  @dataclass
  class MobilityTrace:
      times: np.ndarray            # (T,)
      distances: np.ndarray        # (T, N)
      def distance_at(self, t: float, k: int) -> float: ...
  ```
- 提供两种生成器:`linear_walk`(沿直线匀速)、`random_waypoint`(经典 RWP 模型)。
- `Topology` 增加可选字段 `mobility: MobilityTrace | None`;为 `None` 时退化为常量距离,保持旧测试兼容。
- `cost.transmission_time(wf, topo, sched, i, t_start)` 增加 `t_start` 形参;`schedule_times()` 内部按任务起始时刻传入。
- handover:当设备所在小区切换时,若当前任务仍在执行,触发一次"状态搬运 + 缓存重建"代价(参考 #11 的 `output_location` 思路,计入额外 OT)。

**新建测试** `tests/test_mobility.py`:两节点 + 直线运动,断言设备远离节点时 OT 单调增。

**预期影响**。Figure 6/7/8 在加入 mobility 后,UARP 与 FF/WF 的差距会拉大(UARP 知道距离会变,FF 不会)。

---

### 一.2 NHPP 事件点过程

**方案**。

- 改造 `generate_events()`:
  ```python
  def generate_events(topo, rate_fn: Callable[[float], float], T_horizon: float, *, seed=0):
      """NHPP sampling — thinning algorithm."""
      ...
  ```
  旧 `n_events=N` 接口保留为捷径(等价于均匀 rate)。
- `Event` 加 `time: float`;事件按 `time` 排序后顺序 apply。
- `reschedule()` 改为遍历事件列表:每个事件 fire 时调用一次 `replan(now, state)`(详见 #10 但本轮不展开)。

---

### 一.3 新事件类型

**新增事件**:

| 类型 | 语义 | 关键字段 |
|---|---|---|
| `bandwidth_jitter` | 链路带宽乘性扰动 | `factor`, `duration` |
| `queue_contention` | 节点 capacity 临时缩放(代表邻居工作流抢占) | `factor`, `duration` |
| `task_size_drift` | `Workflow.sizes` 乘性噪声(执行前才显现) | `task_id`, `factor` |

`Event.apply()` 支持带 `expiry` 的扰动,过期自动 revert。

`Workflow.size(i)` 与 `Workflow.sizes` 改成"基础值 + 当前噪声乘子",调度算法用基础值规划,实际执行时用噪声后的值算 ET。

---

### 一.4 真正的服务失效语义

**方案**。

- `service_failure` 恢复为 `available=False`(`events.py:47` 改回)。
- `reschedule()` 增加"迁移分支":失效节点上的已完成任务的中间结果,在新拓扑中需要重传到 fallback 节点,这部分 OT 计入总 WT(对应 `output_location` 机制)。
- Benchmark 不再做 fallback 修复(删除 `algorithm2.py:191-197` 的偷偷迁移逻辑),改为输出 **"违反 deadline 任务比例"** 作为评估指标——这样允许 inf 任务存在而不破坏对比,也消除现在 Figure 9 中给基线送的免费红利。

---

## 二、确定性成本模型 vs 随机现实

### 二.5 Scenario-based 鲁棒评估(CVaR)

**方案**(不重写 NSGA-III,只改 evaluation):

```python
# src/uarp/scheduler/problem.py
def _evaluate(self, X, out, *args, **kwargs):
    S = self.n_scenarios            # 默认 10–20
    wt_per_x = np.zeros((len(X), S))
    cm_per_x = np.zeros((len(X), S))
    for s in range(S):
        topo_s = perturb(self.topology, self.rng_scenarios[s])
        for j, x in enumerate(X):
            sched = Schedule(np.round(x).astype(int))
            wt_per_x[j, s] = completion_time(self.workflow, topo_s, sched)
            cm_per_x[j, s] = total_energy(self.workflow, topo_s, sched)
    out["F"] = np.stack([cvar(wt_per_x, 0.9), cvar(cm_per_x, 0.9)], axis=1)
```

`perturb()` 直接复用 `events.generate_events` 抽样得到的扰动拓扑;S 在 `experiments/config.py` 暴露。

**预期影响**。UARP 从 reactive 升级为真正的 *provisioning*——第一次调度就考虑了不确定性,而非依赖事后 reschedule。

---

### 二.6 分位数 deadline

**方案**。

- `cost.deadline(wf, topo, sched, alpha)` 增加重载 `deadline_quantile(wf, scenarios, q)`:
  ```python
  DT = np.quantile(WT_scenarios, q)
  ```
- `experiments/figure9_success_rate.py` 横轴从 α ∈ [1.1, 1.4] 改为 q ∈ [0.5, 0.99](风险分位数)。
- α-deadline 接口保留兼容,新接口默认调用。

---

### 二.7 重写 Eq. 14 任务级 deadline

**方案**。改用经典 Slack-Time Allocation:

```python
def task_deadline_v2(wf, topo, sched, i, critical_path, alpha=1.2):
    ST, WT = schedule_times(wf, topo, sched)
    slack = alpha * WT[critical_path[-1]] - (WT[critical_path[-1]] - ST[critical_path[0]])
    path_len_through_i = path_length_via(wf, i, critical_path)
    total_path_len = WT[critical_path[-1]] - ST[critical_path[0]]
    return WT[i] + slack * (path_len_through_i / total_path_len)
```

**代码改动**。`src/uarp/model/cost.py:103-124` 整段替换;旧公式保留为 `task_deadline_legacy` 供回归对比。
**测试**。`tests/test_model.py` 增加线性 DAG 上 slack 均匀分配的断言。

---

## 落地时间线

| 阶段 | 内容 | 工作量 | 产出 |
|---|---|---|---|
| **P1** ✅ | 一.1 移动性 + 一.2 NHPP 事件 + 二.7 Eq.14 重写 | 2–3 天 | 新 Figure 10:`success rate vs. mobility speed` |
| **P2** | 一.3 新事件类型 + 一.4 真失效语义 | 1–2 天 | Figure 9 重跑,移除基线偷跑红利 |
| **P3** | 二.5 CVaR 鲁棒评估 + 二.6 分位数 deadline | 2–3 天 | Figure 9 横轴改为风险分位数 |

### P1 落地记录(本次提交)

- `src/uarp/model/cost.py` — `transmission_time/transmission_energy/schedule_times` 接受 `t_start`,新增 `task_deadlines_slf` 取代 ill-posed Eq.14(旧实现保留为 `task_deadline_legacy`)。
- `src/uarp/model/edge.py` — `Topology` 加 `mobility` 字段与 `distance_at(t,k)` 方法,新加入节点自动 fallback 到常量距离。
- `src/uarp/uncertainty/mobility.py`(新) — `MobilityTrace` + `linear_walk` + `random_waypoint`。
- `src/uarp/uncertainty/events.py` — `Event.time` 字段 + `generate_events_nhpp()`(thinning 采样);`apply_events` 按 `time` 排序后应用。
- `src/uarp/uncertainty/algorithm2.py` — `_per_task_deadlines` 改用 SLF。
- `tests/test_model.py` — 新增 3 个测试(SLF total slack / SLF 按 ET 比例 / mobility 拉长 OT),全部 32 个 tests pass。
- `experiments/figure10_mobility.py`(新) — UARP/FF/WF/Benchmark 在 velocity ∈ {0, 0.25, 0.5, 1, 2, 4} 上的 success rate;`run_all.py` 接入。
- `experiments/figure_compare.py`(新) — 读 baseline_*.csv + 新 csv 输出 5 张 side-by-side 对比图。

**复跑命令**:`uv run python -m experiments.run_all && uv run python -m experiments.figure_compare`

---

## 工作流约定

- 每个 P 阶段独立 PR 合入 `improvements` 分支;P1 全部 merge 后再开 P2。
- 每条改动伴随:**单元测试 + 受影响 figure 重跑 + README 对应段落更新**。
- 所有新引入的可调参数集中在 `experiments/config.py`,严禁散落。
- 模型层(`src/uarp/model/`)的语义变化需要在 `notation.md` 同步更新。
