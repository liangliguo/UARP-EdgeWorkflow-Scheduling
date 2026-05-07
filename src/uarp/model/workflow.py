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
