"""
论文约束一致性测试模块
Paper Constraints Consistency Tests

验证实现是否严格遵循论文中的约束定义和伪代码逻辑。
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.problem import SchedulingProblem
from models.solution import Solution
from models.decoder import Decoder
from models.data_loader import DataLoader


class TestPaperConstraints:
    """论文约束一致性测试"""
    
    @pytest.fixture
    def simple_problem(self):
        """创建简化的5工件3阶段测试问题"""
        return DataLoader.load_example1_5jobs()
    
    @pytest.fixture
    def full_problem(self):
        """创建完整的15工件3阶段测试问题"""
        return DataLoader.load_example2_15jobs(seed=42)
    
    def test_constraint_1_machine_binding(self, simple_problem):
        """
        测试约束1：人机绑定约束
        
        论文定义：同一阶段同一机器上的所有工序必须由同一技能等级的工人操作
        即 omega[j,f] 对该机器上所有工件一致
        """
        problem = simple_problem
        decoder = Decoder(problem)
        
        # 生成多个随机解并修复
        for _ in range(20):
            sol = Solution.generate_random(problem)
            repaired = sol.repair(problem)
            
            if repaired is None:
                continue
            
            # 检查每个阶段每台机器的工人技能一致性
            for stage in range(problem.n_stages):
                for machine in range(problem.machines_per_stage[stage]):
                    # 找出分配到该机器的所有工件
                    jobs_on_machine = np.where(repaired.machine_assign[:, stage] == machine)[0]
                    
                    if len(jobs_on_machine) > 0:
                        # 获取这些工件的工人技能等级
                        skills = repaired.worker_skill[jobs_on_machine, stage]
                        # 所有技能等级应该相同
                        assert len(np.unique(skills)) == 1, \
                            f"阶段{stage}机器{machine}上的工人技能不一致: {skills}"
    
    def test_constraint_2_skill_speed_compatibility(self, simple_problem):
        """
        测试约束2：技能-速度兼容约束
        
        论文定义：技能等级 alpha 的工人只能操作速度等级 v <= alpha 的机器
        即 can_operate(alpha, v) = True 当且仅当 alpha >= v
        """
        problem = simple_problem
        decoder = Decoder(problem)
        
        for _ in range(20):
            sol = Solution.generate_random(problem)
            repaired = sol.repair(problem)
            
            if repaired is None:
                continue
            
            # 检查每个工序的技能-速度兼容性
            for job in range(problem.n_jobs):
                for stage in range(problem.n_stages):
                    skill = int(repaired.worker_skill[job, stage])
                    speed = int(repaired.speed_level[job, stage])
                    
                    assert problem.can_operate(skill, speed), \
                        f"工件{job}阶段{stage}: 技能{skill}无法操作速度{speed}"
    
    def test_constraint_3_worker_availability(self, full_problem):
        """
        测试约束3：全局人力可用性约束
        
        论文定义：每种技能等级的工人数量不能超过可用数量
        计数方式：按"启用机器数"计数，而非按时间占用
        """
        problem = full_problem
        decoder = Decoder(problem)
        
        for _ in range(20):
            sol = Solution.generate_random(problem)
            repaired = sol.repair(problem)
            
            if repaired is None:
                continue
            
            # 统计每种技能等级的使用机器数
            skill_usage = np.zeros(problem.n_skill_levels, dtype=int)
            
            for stage in range(problem.n_stages):
                for machine in range(problem.machines_per_stage[stage]):
                    jobs_on_machine = np.where(repaired.machine_assign[:, stage] == machine)[0]
                    
                    if len(jobs_on_machine) > 0:
                        # 该机器的工人技能等级 (应该一致)
                        skill = int(repaired.worker_skill[jobs_on_machine[0], stage])
                        skill_usage[skill] += 1
            
            # 检查是否超过可用数量
            for skill in range(problem.n_skill_levels):
                assert skill_usage[skill] <= problem.workers_available[skill], \
                    f"技能{skill}使用{skill_usage[skill]}人，超过可用{problem.workers_available[skill]}人"
    
    def test_constraint_4_minimum_skill_level(self, full_problem):
        """
        测试约束4：最低可行技能约束
        
        论文定义：omega[j,f] 必须是能操作该机器最大速度的最低技能等级
        """
        problem = full_problem
        decoder = Decoder(problem)
        
        def min_skill_for_speed(speed: int) -> int:
            """返回能操作给定速度的最低技能等级"""
            for skill in range(problem.n_skill_levels):
                if problem.can_operate(skill, speed):
                    return skill
            return problem.n_skill_levels - 1
        
        for _ in range(20):
            sol = Solution.generate_random(problem)
            repaired = sol.repair(problem)
            
            if repaired is None:
                continue
            
            for stage in range(problem.n_stages):
                for machine in range(problem.machines_per_stage[stage]):
                    jobs_on_machine = np.where(repaired.machine_assign[:, stage] == machine)[0]
                    
                    if len(jobs_on_machine) == 0:
                        continue
                    
                    # 该机器上的最大速度
                    max_speed = int(np.max(repaired.speed_level[jobs_on_machine, stage]))
                    # 期望的最低可行技能
                    expected_skill = min_skill_for_speed(max_speed)
                    # 实际分配的技能
                    actual_skill = int(repaired.worker_skill[jobs_on_machine[0], stage])
                    
                    assert actual_skill == expected_skill, \
                        f"阶段{stage}机器{machine}: 最大速度{max_speed}, " \
                        f"期望技能{expected_skill}, 实际技能{actual_skill}"


class TestDecodeCorrectness:
    """解码正确性测试"""
    
    @pytest.fixture
    def simple_problem(self):
        """创建简化测试问题"""
        return DataLoader.load_example1_5jobs()
    
    def test_decode_produces_valid_schedule(self, simple_problem):
        """测试解码器产生有效的调度方案"""
        problem = simple_problem
        decoder = Decoder(problem)
        
        for _ in range(10):
            sol = Solution.generate_random(problem)
            repaired = sol.repair(problem)
            
            if repaired is None:
                continue
            
            # 使用decode_with_schedule获取详细调度信息
            objectives, schedule = decoder.decode_with_schedule(repaired)
            
            # 检查目标值是否有效
            assert repaired.objectives is not None
            assert len(repaired.objectives) == 3
            assert all(obj >= 0 for obj in repaired.objectives)
            
            # 检查调度结果
            assert schedule is not None
            assert 'operations' in schedule
            assert len(schedule['operations']) > 0
    
    def test_decode_respects_precedence(self, simple_problem):
        """测试解码器遵守工序前后约束"""
        problem = simple_problem
        decoder = Decoder(problem)
        
        for _ in range(10):
            sol = Solution.generate_random(problem)
            repaired = sol.repair(problem)
            
            if repaired is None:
                continue
            
            objectives, schedule = decoder.decode_with_schedule(repaired)
            
            # 按工件分组操作
            job_ops = {}
            for op in schedule['operations']:
                job = op['job']
                if job not in job_ops:
                    job_ops[job] = {}
                job_ops[job][op['stage']] = op
            
            # 检查每个工件的阶段顺序
            for job_id, stages in job_ops.items():
                prev_end = 0
                for stage in range(problem.n_stages):
                    if stage in stages:
                        op = stages[stage]
                        start_time = op['start']
                        assert start_time >= prev_end - 0.001, \
                            f"工件{job_id}阶段{stage}开始时间{start_time}早于前一阶段结束{prev_end}"
                        prev_end = op['end']
    
    def test_decode_respects_machine_capacity(self, simple_problem):
        """测试解码器遵守机器容量约束（同一时刻一台机器只能处理一个工件）"""
        problem = simple_problem
        decoder = Decoder(problem)
        
        for _ in range(10):
            sol = Solution.generate_random(problem)
            repaired = sol.repair(problem)
            
            if repaired is None:
                continue
            
            objectives, schedule = decoder.decode_with_schedule(repaired)
            
            # 收集每台机器的时间段
            for stage in range(problem.n_stages):
                for machine in range(problem.machines_per_stage[stage]):
                    intervals = []
                    
                    for op in schedule['operations']:
                        if op['stage'] == stage and op['machine'] == machine:
                            intervals.append((op['start'], op['end'], op['job']))
                    
                    # 检查时间段不重叠
                    intervals.sort(key=lambda x: x[0])
                    for i in range(len(intervals) - 1):
                        end_i = intervals[i][1]
                        start_next = intervals[i + 1][0]
                        assert end_i <= start_next + 0.001, \
                            f"阶段{stage}机器{machine}: 工件{intervals[i][2]}结束{end_i} > " \
                            f"工件{intervals[i+1][2]}开始{start_next}"


class TestRepairEffectiveness:
    """修复有效性测试"""
    
    @pytest.fixture
    def full_problem(self):
        """创建完整测试问题"""
        return DataLoader.load_example2_15jobs(seed=42)
    
    def test_repair_always_produces_feasible_solution(self, full_problem):
        """测试修复总是产生可行解或返回None"""
        problem = full_problem
        
        success_count = 0
        fail_count = 0
        
        for _ in range(100):
            sol = Solution.generate_random(problem)
            repaired = sol.repair(problem)
            
            if repaired is not None:
                success_count += 1
                ok, details = repaired.check_paper_constraints(problem)
                assert ok, f"修复后仍不满足约束: {details}"
            else:
                fail_count += 1
        
        # 修复成功率应该较高
        success_rate = success_count / (success_count + fail_count)
        assert success_rate > 0.5, f"修复成功率过低: {success_rate:.2%}"
    
    def test_repair_preserves_solution_structure(self, full_problem):
        """测试修复保持解的基本结构"""
        problem = full_problem
        
        for _ in range(20):
            sol = Solution.generate_random(problem)
            original_shape = sol.machine_assign.shape
            
            repaired = sol.repair(problem)
            
            if repaired is not None:
                assert repaired.machine_assign.shape == original_shape
                assert repaired.sequence_priority.shape == original_shape
                assert repaired.speed_level.shape == original_shape
                assert repaired.worker_skill.shape == original_shape


class TestObjectiveCalculation:
    """目标函数计算测试"""
    
    @pytest.fixture
    def full_problem(self):
        """创建完整测试问题"""
        return DataLoader.load_example2_15jobs(seed=42)
    
    def test_makespan_is_maximum_completion_time(self, full_problem):
        """测试最大完工时间计算正确"""
        problem = full_problem
        decoder = Decoder(problem)
        
        for _ in range(10):
            sol = Solution.generate_random(problem)
            repaired = sol.repair(problem)
            
            if repaired is None:
                continue
            
            objectives, schedule = decoder.decode_with_schedule(repaired)
            
            # 计算所有工件的完工时间
            completion_times = list(schedule['job_completion'].values())
            
            if completion_times:
                expected_makespan = max(completion_times)
                actual_makespan = repaired.objectives[0]
                
                assert abs(expected_makespan - actual_makespan) < 1e-6, \
                    f"Makespan计算错误: 期望{expected_makespan}, 实际{actual_makespan}"
    
    def test_objectives_are_non_negative(self, full_problem):
        """测试所有目标值非负"""
        problem = full_problem
        decoder = Decoder(problem)
        
        for _ in range(20):
            sol = Solution.generate_random(problem)
            repaired = sol.repair(problem)
            
            if repaired is None:
                continue
            
            decoder.decode(repaired)
            
            assert repaired.objectives[0] >= 0, "Makespan为负"
            assert repaired.objectives[1] >= 0, "人工成本为负"
            assert repaired.objectives[2] >= 0, "能耗为负"


class TestDominanceRelation:
    """支配关系测试"""
    
    def test_dominance_is_strict(self):
        """测试支配关系是严格的（至少一个目标严格更优）"""
        from models.problem import SchedulingProblem
        
        # 创建简单问题
        problem = SchedulingProblem(
            n_jobs=3, n_stages=2,
            machines_per_stage=[2, 2],
            n_speed_levels=2, n_skill_levels=2
        )
        
        sol1 = Solution(problem.n_jobs, problem.n_stages)
        sol2 = Solution(problem.n_jobs, problem.n_stages)
        
        # sol1 在所有目标上都更优
        sol1.objectives = (10, 100, 50)
        sol2.objectives = (20, 200, 100)
        
        assert sol1.dominates(sol2)
        assert not sol2.dominates(sol1)
        
        # 相等的解不支配
        sol3 = Solution(problem.n_jobs, problem.n_stages)
        sol3.objectives = (10, 100, 50)
        
        assert not sol1.dominates(sol3)
        assert not sol3.dominates(sol1)
    
    def test_non_dominated_solutions(self):
        """测试非支配解的识别"""
        from models.problem import SchedulingProblem
        
        problem = SchedulingProblem(
            n_jobs=3, n_stages=2,
            machines_per_stage=[2, 2],
            n_speed_levels=2, n_skill_levels=2
        )
        
        # 创建一组非支配解
        sol1 = Solution(problem.n_jobs, problem.n_stages)
        sol1.objectives = (10, 200, 100)  # 最优makespan
        
        sol2 = Solution(problem.n_jobs, problem.n_stages)
        sol2.objectives = (20, 100, 100)  # 最优人工成本
        
        sol3 = Solution(problem.n_jobs, problem.n_stages)
        sol3.objectives = (15, 150, 50)   # 最优能耗
        
        # 这三个解应该互不支配
        assert not sol1.dominates(sol2)
        assert not sol2.dominates(sol1)
        assert not sol1.dominates(sol3)
        assert not sol3.dominates(sol1)
        assert not sol2.dominates(sol3)
        assert not sol3.dominates(sol2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
