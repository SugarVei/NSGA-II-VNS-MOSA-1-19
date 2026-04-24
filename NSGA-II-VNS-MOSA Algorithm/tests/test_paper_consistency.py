# -*- coding: utf-8 -*-
"""
论文一致性快速测试
Paper Consistency Smoke Tests

验证 Solution.repair 产生的解严格满足论文中定义的人机耦合、技能-速度
向下兼容及工人可用性等约束；也覆盖 NSGA2_VNS_MOSA 端到端可运行性。
"""

import pytest

from models.problem import SchedulingProblem
from models.solution import Solution
from algorithms.nsga2_vns_mosa import NSGA2_VNS_MOSA


def test_repair_produces_paper_feasible_solution():
    problem = SchedulingProblem.generate_random(
        n_jobs=10,
        n_stages=4,
        n_speed_levels=3,
        n_skill_levels=3,
        seed=42,
    )
    sol = Solution.generate_random(problem, seed=123)
    sol2 = sol.repair(problem)
    assert sol2 is not None, f"Repair failed: {getattr(sol, 'feasibility_violations', None)}"
    ok, details = sol2.check_paper_constraints(problem)
    assert ok, details


def test_algorithm_archive_solutions_satisfy_paper_constraints():
    problem = SchedulingProblem.generate_random(
        n_jobs=10,
        n_stages=4,
        n_speed_levels=3,
        n_skill_levels=3,
        seed=7,
    )
    algo = NSGA2_VNS_MOSA(
        problem,
        pop_size=40,
        n_generations=6,
        mosa_layers=3,
        vns_max_iters=2,
        audit_enabled=False,
        seed=7,
    )
    ap = algo.run()
    assert ap, "Archive should not be empty"

    for s in ap:
        ok, details = s.check_paper_constraints(problem)
        assert ok, details
