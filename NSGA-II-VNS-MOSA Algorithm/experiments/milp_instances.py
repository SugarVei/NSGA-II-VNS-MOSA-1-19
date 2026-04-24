# -*- coding: utf-8 -*-
"""experiments/milp_instances.py

生成 MILP 验证实验所需的 V1-V5 小规模实例。
所有实例共享相同阶段结构：3阶段, 2机器/阶段, 3速度等级, 3技能等级。
使用固定种子确保可复现。
"""

import numpy as np
import json
import os
import sys

# 将项目根目录加入 sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from models.problem import SchedulingProblem


# ──────────────────────────────────────────────────────────────────
# 实例参数表
# ──────────────────────────────────────────────────────────────────
INSTANCE_CONFIGS = {
    "V1": {"n_jobs": 3,  "seed": 1001},
    "V2": {"n_jobs": 5,  "seed": 1002},
    "V3": {"n_jobs": 8,  "seed": 1003},
    "V4": {"n_jobs": 10, "seed": 1004},
    "V5": {"n_jobs": 12, "seed": 1005},
}

# 共享结构参数
N_STAGES = 3
MACHINES_PER_STAGE = [2, 2, 2]
N_SPEED_LEVELS = 3
N_SKILL_LEVELS = 3

# 能耗参数（与论文 15 工件案例一致）
PROCESSING_POWER = np.array([
    [[3.5, 5.5, 8.0], [4.0, 6.0, 8.5]],
    [[4.2, 6.5, 9.0], [3.8, 5.5, 7.8]],
    [[3.5, 5.2, 7.5], [4.0, 6.0, 8.8]],
])

SETUP_POWER = np.array([
    [2.5, 2.8],
    [3.0, 2.6],
    [2.4, 2.8],
])

IDLE_POWER = np.array([
    [0.5, 0.6],
    [0.6, 0.5],
    [0.5, 0.6],
])

TRANSPORT_POWER = 0.5
AUX_POWER = 1.0
TRANSPORT_TIME = np.array([3, 3])

# 工人参数
SKILL_WAGES = np.array([150.0, 225.0, 300.0])
SKILL_COMPATIBILITY = np.array([0, 1, 2])
WORKERS_AVAILABLE = np.array([7, 5, 3])


def generate_instance(n_jobs: int, seed: int) -> SchedulingProblem:
    """生成一个验证实例。"""
    rng = np.random.RandomState(seed)
    max_machines = max(MACHINES_PER_STAGE)

    # 加工时间：基础 15-45 分钟，速度折扣 100%/75%/50%
    processing_time = np.zeros((n_jobs, N_STAGES, max_machines, N_SPEED_LEVELS))
    for job in range(n_jobs):
        for stage in range(N_STAGES):
            for machine in range(MACHINES_PER_STAGE[stage]):
                base_time = rng.randint(15, 46)
                for speed in range(N_SPEED_LEVELS):
                    factor = 1.0 - 0.25 * speed
                    processing_time[job, stage, machine, speed] = max(5, int(base_time * factor))

    # 序列相关换模时间：2-8 分钟，同工件无需换模
    setup_time = np.zeros((N_STAGES, max_machines, n_jobs, n_jobs))
    for stage in range(N_STAGES):
        for machine in range(MACHINES_PER_STAGE[stage]):
            for i in range(n_jobs):
                for j in range(n_jobs):
                    if i == j:
                        setup_time[stage, machine, i, j] = 0
                    else:
                        setup_time[stage, machine, i, j] = rng.randint(2, 8)

    return SchedulingProblem(
        n_jobs=n_jobs,
        n_stages=N_STAGES,
        machines_per_stage=list(MACHINES_PER_STAGE),
        n_speed_levels=N_SPEED_LEVELS,
        n_skill_levels=N_SKILL_LEVELS,
        processing_time=processing_time,
        setup_time=setup_time,
        transport_time=TRANSPORT_TIME.copy(),
        processing_power=PROCESSING_POWER.copy(),
        setup_power=SETUP_POWER.copy(),
        idle_power=IDLE_POWER.copy(),
        transport_power=TRANSPORT_POWER,
        aux_power=AUX_POWER,
        skill_wages=SKILL_WAGES.copy(),
        skill_compatibility=SKILL_COMPATIBILITY.copy(),
        workers_available=WORKERS_AVAILABLE.copy(),
        shift_duration=480.0,
    )


def generate_all_instances() -> dict:
    """生成所有 V1-V5 实例，返回 {name: SchedulingProblem}。"""
    instances = {}
    for name, cfg in INSTANCE_CONFIGS.items():
        instances[name] = generate_instance(cfg["n_jobs"], cfg["seed"])
    return instances


def save_instance_to_json(problem: SchedulingProblem, name: str, out_dir: str):
    """将实例保存为 JSON（可选，用于存档）。"""
    data = {
        "name": f"MILP验证实例 {name}",
        "n_jobs": problem.n_jobs,
        "n_stages": problem.n_stages,
        "machines_per_stage": problem.machines_per_stage,
        "n_speed_levels": problem.n_speed_levels,
        "n_skill_levels": problem.n_skill_levels,
        "processing_time": {"data": problem.processing_time.tolist()},
        "setup_time": {},
        "transport_time": problem.transport_time.tolist(),
        "energy_params": {
            "processing_power": problem.processing_power.tolist(),
            "setup_power": problem.setup_power.tolist(),
            "idle_power": problem.idle_power.tolist(),
            "transport_power": problem.transport_power,
            "aux_power": problem.aux_power,
        },
        "worker_params": {
            "skill_wages": problem.skill_wages.tolist(),
            "workers_available": problem.workers_available.tolist(),
        },
        "shift_duration": problem.shift_duration,
    }
    # 序列相关换模时间
    for stage in range(problem.n_stages):
        for machine in range(problem.machines_per_stage[stage]):
            key = f"stage{stage+1}_machine{machine+1}"
            data["setup_time"][key] = problem.setup_time[stage, machine].tolist()

    os.makedirs(out_dir, exist_ok=True)
    filepath = os.path.join(out_dir, f"{name}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  已保存: {filepath}")


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "milp_data")
    instances = generate_all_instances()
    for name, problem in instances.items():
        print(f"\n{name}: {problem.n_jobs} 工件 × {problem.n_stages} 阶段 × {problem.machines_per_stage} 机器")
        print(f"  决策变量规模(约): {problem.n_jobs * problem.n_stages * max(problem.machines_per_stage) * problem.n_speed_levels}")
        save_instance_to_json(problem, name, out_dir)
    print("\n所有验证实例已生成。")
