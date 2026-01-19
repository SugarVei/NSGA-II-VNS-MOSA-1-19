# -*- coding: utf-8 -*-
"""
统一算子库
Unified Operators Library

为所有多目标优化算法提供基于四矩阵编码的交叉、变异和邻域生成算子。
"""

import numpy as np
from typing import List, Tuple, Optional, TYPE_CHECKING
from copy import deepcopy
import random

if TYPE_CHECKING:
    from models.problem import SchedulingProblem
    from models.solution import Solution
    from models.decoder import Decoder



def four_matrix_sx_crossover(
    parent1: 'Solution',
    parent2: 'Solution',
    rng: np.random.Generator,
    problem: 'SchedulingProblem',
    decoder: 'Decoder'
) -> Tuple['Solution', 'Solution']:
    """
    四矩阵交换序列交叉 (SX Crossover, paper-aligned)

    核心思想（与论文描述一致）：
    1) 随机选择一个工件子集块 \u2112 = {l, l+1, ..., u}（按工件索引形成的连续块）。
    2) 以 sequence_priority 的“排序签名”作为序列表示，得到两个父代在该块上的排列 π1、π2。
    3) 构造 swap-path：将 π1 变换到 π2 所需的一系列交换。
    4) 沿 swap-path 逐步交换“整行四矩阵”，得到过渡子代集合 C，并用 NSGA-II 准则挑选两个输出。

    说明：
    - 这里的“整行交换”指对同一工件 i 的四矩阵行同时交换（M,Q,V,W），保持编码语义一致。
    - 为降低复杂度，候选集来自两条路径：parent1→parent2 与 parent2→parent1。

    Returns:
        两个子代 (child1, child2)，已 repair 且已 decode。
    """
    n_jobs = int(problem.n_jobs)
    if n_jobs <= 1:
        c1, c2 = parent1.copy(), parent2.copy()
        if c1.repair(problem) is None:
            c1 = parent1.copy()
            c1.repair(problem)
        if c2.repair(problem) is None:
            c2 = parent2.copy()
            c2.repair(problem)
        decoder.decode(c1); decoder.decode(c2)
        return c1, c2

    # ---- block selection ----
    l = int(rng.integers(0, n_jobs - 1))
    u = int(rng.integers(l + 1, n_jobs))
    block = list(range(l, u + 1))

    def signature(sol, job):
        # 以该 job 在所有阶段的 priority 向量作为签名
        return tuple(int(x) for x in sol.sequence_priority[job, :])

    def permutation(sol):
        return sorted(block, key=lambda j: (signature(sol, j), j))

    def swap_path(pi_from, pi_to):
        cur = pi_from.copy()
        pos = {job: idx for idx, job in enumerate(cur)}
        swaps = []
        for k in range(len(cur)):
            target_job = pi_to[k]
            if cur[k] == target_job:
                continue
            i = k
            j = pos[target_job]
            a, b = cur[i], cur[j]
            # swap in cur
            cur[i], cur[j] = cur[j], cur[i]
            pos[a], pos[b] = j, i
            swaps.append((a, b))
        return swaps

    def apply_row_swap(sol, a, b):
        sol.machine_assign[[a, b], :] = sol.machine_assign[[b, a], :]
        sol.sequence_priority[[a, b], :] = sol.sequence_priority[[b, a], :]
        sol.speed_level[[a, b], :] = sol.speed_level[[b, a], :]
        sol.worker_skill[[a, b], :] = sol.worker_skill[[b, a], :]

    def crowding_distance(sols):
        if len(sols) <= 2:
            return {id(s): float('inf') for s in sols}
        objs = np.array([s.objectives for s in sols], dtype=float)
        cd = np.zeros(len(sols), dtype=float)
        for m in range(objs.shape[1]):
            order = np.argsort(objs[:, m])
            cd[order[0]] = np.inf
            cd[order[-1]] = np.inf
            fmin, fmax = objs[order[0], m], objs[order[-1], m]
            if fmax - fmin == 0:
                continue
            for k in range(1, len(sols) - 1):
                if np.isinf(cd[order[k]]):
                    continue
                cd[order[k]] += (objs[order[k + 1], m] - objs[order[k - 1], m]) / (fmax - fmin)
        return {id(sols[i]): float(cd[i]) for i in range(len(sols))}

    def phi(sol, ref):
        objs = np.array([s.objectives for s in ref], dtype=float)
        mn = objs.min(axis=0)
        mx = objs.max(axis=0)
        rng_ = np.where(mx - mn == 0, 1.0, mx - mn)
        z = (np.array(sol.objectives, dtype=float) - mn) / rng_
        w = np.array([1/3, 1/3, 1/3], dtype=float)
        return float(np.dot(w, z))

    # ---- generate candidates ----
    candidates = []

    def gen_path(p_from, p_to):
        pi1, pi2 = permutation(p_from), permutation(p_to)
        swaps = swap_path(pi1, pi2)
        cur = p_from.copy()
        # include the starting solution as candidate
        cur.objectives = None
        if cur.repair(problem) is None:
            return
        decoder.decode(cur)
        candidates.append(cur)
        for (a, b) in swaps:
            nxt = cur.copy()
            apply_row_swap(nxt, a, b)
            nxt.objectives = None
            if nxt.repair(problem) is None:
                break
            decoder.decode(nxt)
            candidates.append(nxt)
            cur = nxt

    gen_path(parent1, parent2)
    gen_path(parent2, parent1)

    # 去重（按目标+矩阵哈希的轻量近似）
    uniq = []
    seen = set()
    for s in candidates:
        key = (tuple(round(float(x), 6) for x in s.objectives), s.machine_assign.tobytes(), s.sequence_priority.tobytes())
        if key in seen:
            continue
        seen.add(key)
        uniq.append(s)
    candidates = uniq

    # ---- select 2 offspring ----
    def nondominated_set(sols):
        nd = []
        for s in sols:
            dominated = False
            for t in sols:
                if t is s:
                    continue
                if t.dominates(s):
                    dominated = True
                    break
            if not dominated:
                nd.append(s)
        return nd

    nd = nondominated_set(candidates)

    if len(nd) >= 2:
        cd = crowding_distance(nd)
        nd_sorted = sorted(nd, key=lambda s: cd.get(id(s), 0.0), reverse=True)
        return nd_sorted[0], nd_sorted[1]

    # nd 只有 0 或 1
    if len(nd) == 1:
        first = nd[0]
        rest = [s for s in candidates if s is not first]
        if not rest:
            # 极端情况：只有一个候选
            return first.copy(), first.copy()
        second = min(rest, key=lambda s: phi(s, candidates))
        return first, second

    # 无非支配：按 φ 最小选前二
    candidates_sorted = sorted(candidates, key=lambda s: phi(s, candidates))
    return candidates_sorted[0], candidates_sorted[1]



def mutation_single(
    solution: 'Solution',
    rng: np.random.Generator,
    problem: 'SchedulingProblem'
) -> 'Solution':
    """
    单点变异
    
    随机选择一个 (job, stage) 位置，然后随机选择四矩阵中的一个进行变异。
    
    Args:
        solution: 待变异的解
        rng: numpy 随机数生成器
        problem: 调度问题实例
        
    Returns:
        变异后的解（已修复，未评估）
    """
    mutant = solution.copy()
    
    job = rng.integers(0, problem.n_jobs)
    stage = rng.integers(0, problem.n_stages)
    
    # 随机选择变异类型
    mutation_type = rng.choice(['machine', 'priority', 'speed', 'skill'])
    
    if mutation_type == 'machine':
        n_machines = problem.machines_per_stage[stage]
        if n_machines > 1:
            current = mutant.machine_assign[job, stage]
            new_machine = rng.integers(0, n_machines)
            while new_machine == current and n_machines > 1:
                new_machine = rng.integers(0, n_machines)
            mutant.machine_assign[job, stage] = new_machine
            
    elif mutation_type == 'priority':
        # 与另一个随机工件交换优先级
        if problem.n_jobs > 1:
            other_job = rng.integers(0, problem.n_jobs)
            while other_job == job:
                other_job = rng.integers(0, problem.n_jobs)
            mutant.sequence_priority[job, stage], mutant.sequence_priority[other_job, stage] = \
                mutant.sequence_priority[other_job, stage], mutant.sequence_priority[job, stage]
                
    elif mutation_type == 'speed':
        current = mutant.speed_level[job, stage]
        delta = rng.choice([-1, 1])
        new_speed = max(0, min(problem.n_speed_levels - 1, current + delta))
        mutant.speed_level[job, stage] = new_speed
        
    else:  # skill
        current = mutant.worker_skill[job, stage]
        delta = rng.choice([-1, 1])
        new_skill = max(0, min(problem.n_skill_levels - 1, current + delta))
        mutant.worker_skill[job, stage] = new_skill
    
    mutant.objectives = None
    mutant2 = mutant.repair(problem)
    if mutant2 is None:
        # 修复失败：回退到原解（等价于本次不变异）
        mutant2 = solution.copy()
    mutant2.objectives = None
    return mutant2


def mutation_multi(
    solution: 'Solution',
    rng: np.random.Generator,
    problem: 'SchedulingProblem',
    k: int = 3
) -> 'Solution':
    """
    多点变异
    
    连续执行 k 次单点变异。
    
    Args:
        solution: 待变异的解
        rng: numpy 随机数生成器
        problem: 调度问题实例
        k: 变异次数
        
    Returns:
        变异后的解（已修复，未评估）
    """
    mutant = solution.copy()
    
    for _ in range(k):
        mutant = mutation_single(mutant, rng, problem)
    
    return mutant


def mutation_inversion(
    solution: 'Solution',
    rng: np.random.Generator,
    problem: 'SchedulingProblem'
) -> 'Solution':
    """
    反转变异
    
    对某个阶段的序列进行片段反转，同步应用到四矩阵。
    
    Args:
        solution: 待变异的解
        rng: numpy 随机数生成器
        problem: 调度问题实例
        
    Returns:
        变异后的解（已修复，未评估）
    """
    mutant = solution.copy()
    
    stage = rng.integers(0, problem.n_stages)
    n_jobs = problem.n_jobs
    
    if n_jobs < 3:
        return mutant
    
    # 按当前 sequence_priority 得到排序
    priorities = mutant.sequence_priority[:, stage]
    order = np.argsort(priorities)
    
    # 随机选择反转区间 [i, j]
    i = rng.integers(0, n_jobs - 1)
    j = rng.integers(i + 1, n_jobs)
    
    # 反转区间
    reversed_segment = order[i:j+1][::-1]
    new_order = np.concatenate([order[:i], reversed_segment, order[j+1:]])
    
    # 根据新顺序重新赋值 priority
    for rank, job_idx in enumerate(new_order):
        mutant.sequence_priority[job_idx, stage] = rank
    
    mutant.objectives = None
    mutant2 = mutant.repair(problem)
    if mutant2 is None:
        mutant2 = solution.copy()
    mutant2.objectives = None
    return mutant2


def simple_neighbor(
    solution: 'Solution',
    rng: np.random.Generator,
    problem: 'SchedulingProblem',
    mode: Optional[str] = None
) -> 'Solution':
    """
    简单邻域生成
    
    为 MOSA(use_vns=False) 提供邻域候选。随机选择一种变异方式生成邻居。
    
    Args:
        solution: 当前解
        rng: numpy 随机数生成器
        problem: 调度问题实例
        mode: 变异模式 ('single', 'multi', 'inversion')，None 则随机选择
        
    Returns:
        邻域解（已修复，未评估）
    """
    if mode is None:
        mode = rng.choice(['single', 'multi', 'inversion'])
    
    if mode == 'single':
        return mutation_single(solution, rng, problem)
    elif mode == 'multi':
        return mutation_multi(solution, rng, problem, k=3)
    else:  # inversion
        return mutation_inversion(solution, rng, problem)


def apply_crossover_with_probability(
    parent1: 'Solution',
    parent2: 'Solution',
    crossover_prob: float,
    rng: np.random.Generator,
    problem: 'SchedulingProblem',
    decoder: 'Decoder'
) -> Tuple['Solution', 'Solution']:
    """
    带概率的交叉操作
    
    Args:
        parent1: 父代1
        parent2: 父代2
        crossover_prob: 交叉概率
        rng: numpy 随机数生成器
        problem: 调度问题实例
        decoder: 解码器
        
    Returns:
        (child1, child2): 两个子代
    """
    if rng.random() > crossover_prob:
        c1 = parent1.copy()
        c2 = parent2.copy()
        if c1.objectives is None:
            decoder.decode(c1)
        if c2.objectives is None:
            decoder.decode(c2)
        return c1, c2
    
    return four_matrix_sx_crossover(parent1, parent2, rng, problem, decoder)


def apply_mutation_with_probability(
    solution: 'Solution',
    mutation_prob: float,
    rng: np.random.Generator,
    problem: 'SchedulingProblem',
    decoder: 'Decoder'
) -> 'Solution':
    """
    带概率的变异操作
    
    Args:
        solution: 待变异的解
        mutation_prob: 变异概率
        rng: numpy 随机数生成器
        problem: 调度问题实例
        decoder: 解码器
        
    Returns:
        变异后的解
    """
    if rng.random() > mutation_prob:
        return solution
    
    # 随机选择变异类型
    mutation_type = rng.choice(['single', 'multi', 'inversion'])
    
    if mutation_type == 'single':
        mutant = mutation_single(solution, rng, problem)
    elif mutation_type == 'multi':
        mutant = mutation_multi(solution, rng, problem)
    else:
        mutant = mutation_inversion(solution, rng, problem)
    
    decoder.decode(mutant)
    return mutant
