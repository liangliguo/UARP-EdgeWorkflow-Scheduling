"""Workflow DAG model — paper §2.2.1, Eqs. (1)(2)."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np


@dataclass
class Task:
    idx: int
    size: float  # b_{m,i} — task data size in MB-equivalent units


@dataclass
class Workflow:
    tasks: list[Task]
    graph: nx.DiGraph  # node = task idx, edge = precedence

    @property
    def M(self) -> int:
        return len(self.tasks)

    def predecessors(self, i: int) -> list[int]:
        return list(self.graph.predecessors(i))

    def successors(self, i: int) -> list[int]:
        return list(self.graph.successors(i))

    def topo_order(self) -> list[int]:
        return list(nx.topological_sort(self.graph))

    def size(self, i: int) -> float:
        return self.tasks[i].size


def subset(workflow: "Workflow", task_ids: list[int]) -> tuple["Workflow", list[int]]:
    """Return a workflow induced by the given task ids, plus the global->local map.

    Used by Algorithm 2 to reschedule only the un-finished tasks.
    The returned workflow uses local indices 0..len(task_ids)-1 in topo order.
    """
    keep = set(task_ids)
    ordered = [i for i in workflow.topo_order() if i in keep]
    global_to_local = {g: l for l, g in enumerate(ordered)}
    new_g = nx.DiGraph()
    new_g.add_nodes_from(range(len(ordered)))
    for u, v in workflow.graph.edges():
        if u in keep and v in keep:
            new_g.add_edge(global_to_local[u], global_to_local[v])
    new_tasks = [Task(idx=l, size=workflow.tasks[g].size) for l, g in enumerate(ordered)]
    return Workflow(tasks=new_tasks, graph=new_g), ordered


def random_dag(
    n_tasks: int,
    edge_prob: float = 0.3,
    size_range: tuple[float, float] = (50.0, 200.0),
    seed: int | None = None,
) -> Workflow:
    """Generate a random DAG with n_tasks nodes.

    Strategy: order nodes 0..n-1, only add edge i->j when i<j (guarantees acyclic).
    Then ensure connectivity by chaining the topological roots.
    """
    rng = np.random.default_rng(seed)
    g = nx.DiGraph()
    g.add_nodes_from(range(n_tasks))
    for i in range(n_tasks):
        for j in range(i + 1, n_tasks):
            if rng.random() < edge_prob:
                g.add_edge(i, j)
    # connect any orphans into the chain so there's a single source/sink path
    for i in range(1, n_tasks):
        if g.in_degree(i) == 0:
            g.add_edge(i - 1, i)
    for i in range(n_tasks - 1):
        if g.out_degree(i) == 0:
            g.add_edge(i, i + 1)
    sizes = rng.uniform(size_range[0], size_range[1], size=n_tasks)
    tasks = [Task(idx=i, size=float(sizes[i])) for i in range(n_tasks)]
    return Workflow(tasks=tasks, graph=g)
