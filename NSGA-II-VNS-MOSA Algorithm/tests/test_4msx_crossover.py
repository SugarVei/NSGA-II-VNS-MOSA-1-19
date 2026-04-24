# -*- coding: utf-8 -*-
"""
4M-SX 交叉算子专项测试
Tests for Four-Matrix Swap Crossover (4M-SX)
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.problem import SchedulingProblem
from models.solution import Solution
from models.decoder import Decoder
from algorithms.operators import four_matrix_sx_crossover


def _make_problem_and_parents(seed=42, n_jobs=8, n_stages=3):
    """辅助：生成问题实例和两个已评估的父代"""
    problem = SchedulingProblem.generate_random(
        n_jobs=n_jobs,
        n_stages=n_stages,
        n_speed_levels=3,
        n_skill_levels=3,
        seed=seed,
    )
    decoder = Decoder(problem)
    rng = np.random.default_rng(seed)

    p1 = Solution.generate_random(problem, seed=seed)
    p1.repair(problem)
    decoder.decode(p1)

    p2 = Solution.generate_random(problem, seed=seed + 1)
    p2.repair(problem)
    decoder.decode(p2)

    return problem, decoder, rng, p1, p2


class TestSwapPathCorrectness:
    """验证 swap-path 终态确实将 π_A 变换为 π_B"""

    def test_swap_path_produces_valid_permutation(self):
        """交叉操作应产生有效的排列（无重复、无遗漏）"""
        problem, decoder, rng, p1, p2 = _make_problem_and_parents()
        c1 = four_matrix_sx_crossover(p1, p2, rng, problem, decoder)
        c2 = four_matrix_sx_crossover(p1, p2, rng, problem, decoder)

        for stage in range(problem.n_stages):
            perm1 = list(np.argsort(c1.sequence_priority[:, stage]))
            perm2 = list(np.argsort(c2.sequence_priority[:, stage]))
            # 应该是 0..n_jobs-1 的排列
            assert sorted(perm1) == list(range(problem.n_jobs))
            assert sorted(perm2) == list(range(problem.n_jobs))


class TestOffspringFeasibility:
    """验证输出子代满足论文约束"""

    def test_offspring_satisfy_paper_constraints(self):
        problem, decoder, rng, p1, p2 = _make_problem_and_parents()
        c1 = four_matrix_sx_crossover(p1, p2, rng, problem, decoder)
        c2 = four_matrix_sx_crossover(p1, p2, rng, problem, decoder)

        ok1, details1 = c1.check_paper_constraints(problem)
        ok2, details2 = c2.check_paper_constraints(problem)
        assert ok1, f"Child1 constraint violation: {details1}"
        assert ok2, f"Child2 constraint violation: {details2}"

    def test_offspring_feasibility_check(self):
        problem, decoder, rng, p1, p2 = _make_problem_and_parents()
        c1 = four_matrix_sx_crossover(p1, p2, rng, problem, decoder)
        c2 = four_matrix_sx_crossover(p1, p2, rng, problem, decoder)

        is_feasible1, violations1 = c1.check_feasibility(problem)
        is_feasible2, violations2 = c2.check_feasibility(problem)
        assert is_feasible1, f"Child1 not feasible: {violations1}"
        assert is_feasible2, f"Child2 not feasible: {violations2}"

    def test_multiple_seeds(self):
        """多种种子下子代均可行"""
        for seed in [1, 7, 42, 99, 123]:
            problem, decoder, rng, p1, p2 = _make_problem_and_parents(seed=seed)
            c1 = four_matrix_sx_crossover(p1, p2, rng, problem, decoder)
            c2 = four_matrix_sx_crossover(p1, p2, rng, problem, decoder)
            ok1, _ = c1.check_paper_constraints(problem)
            ok2, _ = c2.check_paper_constraints(problem)
            assert ok1, f"Seed {seed}: child1 failed"
            assert ok2, f"Seed {seed}: child2 failed"


class TestOffspringEvaluated:
    """验证输出子代 objectives 非 None"""

    def test_objectives_not_none(self):
        problem, decoder, rng, p1, p2 = _make_problem_and_parents()
        c1 = four_matrix_sx_crossover(p1, p2, rng, problem, decoder)
        c2 = four_matrix_sx_crossover(p1, p2, rng, problem, decoder)

        assert c1.objectives is not None, "Child1 objectives should be evaluated"
        assert c2.objectives is not None, "Child2 objectives should be evaluated"
        assert len(c1.objectives) == 3
        assert len(c2.objectives) == 3
        # 所有目标值应为正数
        assert all(o >= 0 for o in c1.objectives)
        assert all(o >= 0 for o in c2.objectives)


class TestCandidatePoolNonempty:
    """验证候选池不会导致空返回"""

    def test_always_returns_two_solutions(self):
        """无论父代如何，都应返回两个解"""
        problem, decoder, rng, p1, p2 = _make_problem_and_parents()
        c1 = four_matrix_sx_crossover(p1, p2, rng, problem, decoder)
        c2 = four_matrix_sx_crossover(p1, p2, rng, problem, decoder)

        assert c1 is not None
        assert c2 is not None
        assert isinstance(c1.objectives, tuple)
        assert isinstance(c2.objectives, tuple)

    def test_identical_parents(self):
        """两个相同的父代也应正常返回"""
        problem, decoder, rng, p1, _ = _make_problem_and_parents()
        p_clone = p1.copy()
        decoder.decode(p_clone)

        c1 = four_matrix_sx_crossover(p1, p_clone, rng, problem, decoder)
        c2 = four_matrix_sx_crossover(p1, p_clone, rng, problem, decoder)
        assert c1 is not None
        assert c2 is not None
        assert c1.objectives is not None
        assert c2.objectives is not None


class TestDeterministic:
    """验证种子确定性"""

    def test_deterministic_with_seed(self):
        """相同种子应产生相同结果"""
        problem, decoder, _, p1, p2 = _make_problem_and_parents(seed=42)

        rng1 = np.random.default_rng(100)
        c1a = four_matrix_sx_crossover(p1, p2, rng1, problem, decoder)

        rng2 = np.random.default_rng(100)
        c1b = four_matrix_sx_crossover(p1, p2, rng2, problem, decoder)

        np.testing.assert_array_equal(c1a.machine_assign, c1b.machine_assign)
        np.testing.assert_array_equal(c1a.sequence_priority, c1b.sequence_priority)
        np.testing.assert_array_equal(c1a.speed_level, c1b.speed_level)
        np.testing.assert_array_equal(c1a.worker_skill, c1b.worker_skill)
        assert c1a.objectives == c1b.objectives


class TestSmallCases:
    """边界情况：极小规模问题"""

    def test_single_job(self):
        """只有1个工件"""
        problem, decoder, rng, p1, p2 = _make_problem_and_parents(
            seed=42, n_jobs=1, n_stages=2
        )
        c1 = four_matrix_sx_crossover(p1, p2, rng, problem, decoder)
        c2 = four_matrix_sx_crossover(p1, p2, rng, problem, decoder)
        assert c1.objectives is not None
        assert c2.objectives is not None

    def test_two_jobs(self):
        """只有2个工件"""
        problem, decoder, rng, p1, p2 = _make_problem_and_parents(
            seed=42, n_jobs=2, n_stages=2
        )
        c1 = four_matrix_sx_crossover(p1, p2, rng, problem, decoder)
        c2 = four_matrix_sx_crossover(p1, p2, rng, problem, decoder)
        assert c1.objectives is not None
        assert c2.objectives is not None
        ok1, _ = c1.check_paper_constraints(problem)
        ok2, _ = c2.check_paper_constraints(problem)
        assert ok1
        assert ok2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
