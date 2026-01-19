# -*- coding: utf-8 -*-
"""
诊断脚本：测试单个NSGA2-VNS-MOSA任务是否能正常运行
用于诊断进程池崩溃问题 (无外部依赖版本)
"""

import sys
import os
import time
import traceback

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from models.problem import SchedulingProblem
from algorithms.nsga2_vns_mosa import NSGA2_VNS_MOSA


def test_multiprocessing():
    """测试多进程是否正常工作"""
    print("\n" + "=" * 60)
    print("诊断：测试多进程子进程能否正常工作")
    print("=" * 60)
    
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    
    def simple_task(x):
        """简单的测试任务"""
        import time
        time.sleep(0.1)
        return x * x
    
    print(f"\n[1] 系统 CPU 核心数: {multiprocessing.cpu_count()}")
    print(f"    Windows 限制最大进程数: 61")
    
    # 测试少量进程
    n_workers = 4
    n_tasks = 8
    
    print(f"\n[2] 测试 {n_workers} 个工作进程执行 {n_tasks} 个任务...")
    
    try:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(simple_task, range(n_tasks)))
        print(f"    ✅ 多进程测试成功: {results}")
    except Exception as e:
        print(f"    ❌ 多进程测试失败: {e}")
        traceback.print_exc()
        return False
    
    return True


def test_subprocess_algorithm():
    """测试在子进程中运行算法 - 这是最关键的测试"""
    print("\n" + "=" * 60)
    print("诊断：测试子进程中运行算法 (模拟真实场景)")
    print("=" * 60)
    
    from concurrent.futures import ProcessPoolExecutor
    
    # 导入真实的任务函数
    from ui.comparison_worker import run_algorithm_task
    from ui.case_data import CaseConfig
    
    # 创建一个简单的测试用例 (使用正确的构造方式)
    case = CaseConfig(
        case_no=1, 
        n_jobs=10, 
        machines_per_stage=[3, 4, 3],
        workers_available=[4, 5, 3]  # 必须同时提供
    )
    
    # 准备任务参数
    task_args = {
        'case_no': 1,
        'run_idx': 0,
        'alg_name': 'NSGA2-VNS-MOSA',
        'case': case,
        'params': {
            'pop_size': 200,
            'n_generations': 100,
            'vns_max_iters': 5,
            'elite_ratio': 0.1,
        },
        'seed': 42,
        'base_seed': 42,
    }
    
    print("\n[1] 提交单个任务到进程池...")
    print("    (这将模拟真实的算法对比场景)")
    print("    问题规模: n_jobs=10, stages=3, machines=[3,4,3]")
    
    start_time = time.time()
    
    try:
        with ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_algorithm_task, task_args)
            print("    等待任务完成 (预计1-3分钟)...")
            result = future.result(timeout=300)  # 5分钟超时
        
        elapsed = time.time() - start_time
        
        if result.get('error'):
            print(f"\n    ⚠️ 任务返回错误: {result['error']}")
            return False
        else:
            n_solutions = len(result.get('objectives', []))
            print(f"\n    ✅ 子进程任务成功!")
            print(f"    耗时: {elapsed:.1f} 秒")
            print(f"    Pareto解数量: {n_solutions}")
            return True
            
    except Exception as e:
        print(f"\n    ❌ 子进程任务失败: {e}")
        traceback.print_exc()
        return False


def test_parallel_subprocess_algorithm(n_parallel=4):
    """测试多个子进程同时运行算法"""
    print("\n" + "=" * 60)
    print(f"诊断：测试 {n_parallel} 个子进程同时运行算法")
    print("=" * 60)
    
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    # 导入真实的任务函数
    from ui.comparison_worker import run_algorithm_task
    from ui.case_data import CaseConfig
    
    # 创建多个测试任务
    task_list = []
    for i in range(n_parallel):
        case = CaseConfig(
            case_no=i+1, 
            n_jobs=10, 
            machines_per_stage=[3, 4, 3],
            workers_available=[4, 5, 3]
        )
        task_args = {
            'case_no': i+1,
            'run_idx': 0,
            'alg_name': 'NSGA2-VNS-MOSA',
            'case': case,
            'params': {
                'pop_size': 200,
                'n_generations': 100,
                'vns_max_iters': 5,
                'elite_ratio': 0.1,
            },
            'seed': 42 + i,
            'base_seed': 42,
        }
        task_list.append(task_args)
    
    print(f"\n[1] 同时提交 {n_parallel} 个任务到进程池...")
    print("    每个任务: n_jobs=10, stages=3")
    
    start_time = time.time()
    success_count = 0
    error_count = 0
    
    try:
        with ProcessPoolExecutor(max_workers=n_parallel) as executor:
            futures = {executor.submit(run_algorithm_task, args): args['case_no'] for args in task_list}
            print(f"    等待 {n_parallel} 个任务完成...")
            
            for future in as_completed(futures, timeout=600):  # 10分钟超时
                case_no = futures[future]
                try:
                    result = future.result()
                    if result.get('error'):
                        print(f"    ⚠️ Case {case_no} 返回错误: {result['error']}")
                        error_count += 1
                    else:
                        n_solutions = len(result.get('objectives', []))
                        print(f"    ✅ Case {case_no} 成功, Pareto解: {n_solutions}个")
                        success_count += 1
                except Exception as e:
                    print(f"    ❌ Case {case_no} 异常: {e}")
                    error_count += 1
        
        elapsed = time.time() - start_time
        print(f"\n    总耗时: {elapsed:.1f} 秒")
        print(f"    成功: {success_count}/{n_parallel}, 失败: {error_count}/{n_parallel}")
        
        return error_count == 0
            
    except Exception as e:
        print(f"\n    ❌ 并行测试失败: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("NSGA2-VNS-MOSA 进程崩溃诊断工具")
    print("=" * 60)
    
    all_passed = True
    
    # 测试1: 多进程基本功能
    if not test_multiprocessing():
        all_passed = False
        print("\n❌ 多进程基本功能测试失败，可能是系统配置问题")
    
    # 测试2: 单个子进程运行算法
    print("\n" + "-" * 60)
    if not test_subprocess_algorithm():
        all_passed = False
        print("\n❌ 单个子进程算法测试失败!")
    else:
        # 测试3: 多个子进程同时运行算法 (测试资源竞争)
        print("\n" + "-" * 60)
        if not test_parallel_subprocess_algorithm(n_parallel=4):
            all_passed = False
            print("\n⚠️ 多进程并行测试有失败，可能存在资源竞争问题")
    
    # 总结
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ 所有诊断测试通过")
        print("问题可能出在并行数量过多 (61个进程) 导致的资源竞争")
        print("建议：将 max_workers 从 61 减少到 20-30")
    else:
        print("❌ 部分诊断测试失败")
        print("请查看上方的详细错误信息")
    print("=" * 60)
