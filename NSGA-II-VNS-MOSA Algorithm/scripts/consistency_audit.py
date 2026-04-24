
# -*- coding: utf-8 -*-
"""scripts/consistency_audit.py

用途：
- 运行 NSGA-II-VNS-MOSA（paper-aligned 实现）并输出“可贴到论文附录”的一致性验证表
- 同时输出两类 CSV 日志：
  1) audit_vns_moves_*.csv：每次 VNS 迭代的候选集规模、修复失败数、接受情况
  2) audit_constraints_*.csv：对 AP 抽样解做论文约束检查（人机绑定/兼容/可用性/最低可行技能）

说明：
- 不依赖 UI 数据文件；使用 SchedulingProblem.generate_random() 生成可复现实例
- 你应把该脚本的输出作为“实现严格遵循伪代码/约束”的证据（附录或补充材料）
"""

import argparse
import sys
from pathlib import Path

# 允许以 `python scripts/consistency_audit.py` 直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
from datetime import datetime

import numpy as np

from models.problem import SchedulingProblem
from algorithms.nsga2_vns_mosa import NSGA2_VNS_MOSA


def run_one(seed: int, out_dir: str,
            n_jobs: int, n_stages: int,
            pop_size: int, n_generations: int,
            mosa_layers: int, vns_max_iters: int) -> dict:
    problem = SchedulingProblem.generate_random(
        n_jobs=n_jobs,
        n_stages=n_stages,
        machines_per_stage=None,
        n_speed_levels=3,
        n_skill_levels=3,
        workers_available=None,
        seed=seed,
    )

    algo = NSGA2_VNS_MOSA(
        problem,
        pop_size=pop_size,
        n_generations=n_generations,
        mosa_layers=mosa_layers,
        vns_max_iters=vns_max_iters,
        audit_enabled=True,
        audit_dir=out_dir,
        audit_sample_k=20,
        seed=seed,
    )
    ap = algo.run()

    # 全量一致性检查（对 AP 内所有解）
    all_ok = 1
    worst = {"binding_violations": 0, "compatibility_violations": 0, "availability_violations": 0, "min_skill_violations": 0}
    for s in ap:
        ok, details = s.check_paper_constraints(problem)
        if not ok:
            all_ok = 0
        for k in worst.keys():
            worst[k] = max(worst[k], int(details.get(k, 0)))

    return {
        "seed": seed,
        "n_jobs": n_jobs,
        "n_stages": n_stages,
        "pop_size": pop_size,
        "n_generations": n_generations,
        "mosa_layers": mosa_layers,
        "vns_max_iters": vns_max_iters,
        "ap_size": len(ap),
        "ap_all_ok": all_ok,
        **{f"ap_worst_{k}": v for k, v in worst.items()},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=str, default="0,1,2,3,4")
    p.add_argument("--out", type=str, default="audit")
    p.add_argument("--n_jobs", type=int, default=12)
    p.add_argument("--n_stages", type=int, default=4)
    p.add_argument("--pop_size", type=int, default=60)
    p.add_argument("--n_generations", type=int, default=15)
    p.add_argument("--mosa_layers", type=int, default=8)
    p.add_argument("--vns_max_iters", type=int, default=4)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    rows = []
    for seed in seeds:
        rows.append(run_one(
            seed=seed,
            out_dir=args.out,
            n_jobs=args.n_jobs,
            n_stages=args.n_stages,
            pop_size=args.pop_size,
            n_generations=args.n_generations,
            mosa_layers=args.mosa_layers,
            vns_max_iters=args.vns_max_iters,
        ))

    # 输出附录汇总表（按 run 汇总）
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = os.path.join(args.out, f"appendix_overall_{ts}.md")
    header = "| seed | |AP| | AP全量通过(0/1) | worst绑定 | worst兼容 | worst可用性 | worst最低技能 |\n|---:|---:|---:|---:|---:|---:|---:|\n"
    lines = ["# 多次运行一致性汇总（自动生成）\n\n", header]
    for r in rows:
        lines.append(
            f"| {r['seed']} | {r['ap_size']} | {r['ap_all_ok']} | {r['ap_worst_binding_violations']} | {r['ap_worst_compatibility_violations']} | {r['ap_worst_availability_violations']} | {r['ap_worst_min_skill_violations']} |\n"
        )
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("".join(lines))

    print(f"Wrote: {md_path}")
    print("Note: per-run VNS/constraint CSV logs are also written into the same folder.")


if __name__ == "__main__":
    main()