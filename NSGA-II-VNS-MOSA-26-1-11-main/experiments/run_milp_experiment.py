import os
import sys
import time
import json
import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from experiments.milp_instances import generate_all_instances
from experiments.milp_solver_scipy import ScipyMILPFormulation
from pymoo.indicators.hv import HV

from algorithms.nsga2_vns_mosa import NSGA2_VNS_MOSA
from algorithms.nsga2 import NSGAII as NSGA2
from algorithms.spea2 import SPEA2
from algorithms.mopso import MOPSO
from algorithms.moead import MOEAD

def get_hv(pareto_front, ref_point):
    if len(pareto_front) == 0:
        return 0.0
    ind = HV(ref_point=ref_point)
    return ind(np.array(pareto_front))

def run_milp_epsilon_constraint(problem, grid_size=4, time_limit=30.0):
    model = ScipyMILPFormulation(problem)
    
    print("\n--- Step 1: Finding Payoff Extremes ---")
    res1 = model.solve("F1", time_limit=time_limit)
    res2 = model.solve("F2", time_limit=time_limit)
    res3 = model.solve("F3", time_limit=time_limit)
    
    def safe_get(res, default):
        return res['objectives'] if 'objectives' in res else default
        
    f1_extr = [ safe_get(res1, (10000, 10000, 10000))[0], safe_get(res2, (10000, 10000, 10000))[0], safe_get(res3, (10000, 10000, 10000))[0] ]
    f2_extr = [ safe_get(res1, (10000, 10000, 10000))[1], safe_get(res2, (10000, 10000, 10000))[1], safe_get(res3, (10000, 10000, 10000))[1] ]
    f3_extr = [ safe_get(res1, (10000, 10000, 10000))[2], safe_get(res2, (10000, 10000, 10000))[2], safe_get(res3, (10000, 10000, 10000))[2] ]
    
    F2_min = min(f2_extr)
    F2_max = max(f2_extr) if max(f2_extr) < 9000 else F2_min + 1500
    F3_min = min(f3_extr)
    F3_max = max(f3_extr) if max(f3_extr) < 9000 else F3_min + 50
    
    if F2_max <= F2_min: F2_max = F2_min + 100
    if F3_max <= F3_min: F3_max = F3_min + 10
    
    eps2_vals = np.linspace(F2_max, F2_min, grid_size)
    eps3_vals = np.linspace(F3_max, F3_min, grid_size)
    
    pf_list = []
    gaps = []
    total_time = 0.0
    feasible_count = 0
    
    for res in [res1, res2, res3]:
        if 'objectives' in res:
            pf_list.append(res['objectives'])
            gaps.append(res.get('mip_gap', 0.0))
            total_time += res['time']
    
    print(f"\n--- Step 2: Grid Search (F2: {F2_min:.1f}-{F2_max:.1f}, F3: {F3_min:.1f}-{F3_max:.1f}) ---")
    count = 0
    for e2 in eps2_vals:
        for e3 in eps3_vals:
            count += 1
            print(f"  Subproblem {count}/{grid_size*grid_size}: eps2={e2:.1f}, eps3={e3:.1f}")
            res = model.solve("F1", limit_F2=e2, limit_F3=e3, time_limit=time_limit)
            total_time += res['time']
            if 'objectives' in res:
                pf_list.append(res['objectives'])
                feasible_count += 1
                gaps.append(res.get('mip_gap', 0.0))
                
    # Filter Non-dominated
    pf_exact = []
    for sol in pf_list:
        dom = False
        for other in pf_list:
            if sol == other: continue
            if other[0] <= sol[0] and other[1] <= sol[1] and other[2] <= sol[2] and \
               (other[0] < sol[0] or other[1] < sol[1] or other[2] < sol[2]):
                dom = True
                break
        if not dom:
            # check duplicate
            if not any(np.allclose(sol, p, atol=1e-3) for p in pf_exact):
                pf_exact.append(sol)
                
    avg_gap = np.mean(gaps) * 100 if gaps else 0.0
    return pf_exact, avg_gap, total_time, feasible_count

def run_metaheuristics(problem, name, ref_point, n_runs=10):
    algos = {
        "NSGA-II-VNS-MOSA": NSGA2_VNS_MOSA(problem, pop_size=40, n_generations=40),
        "NSGA-II": NSGA2(problem, pop_size=40, n_generations=40),
        "MOEA/D": MOEAD(problem, pop_size=40, n_generations=40),
        "SPEA2": SPEA2(problem, pop_size=40, n_generations=40),
        "MOPSO": MOPSO(problem, swarm_size=40, max_iterations=40)
    }
    
    results = {}
    for aname, algo in algos.items():
        hvs = []
        for r in range(n_runs):
            print(f"    Running {aname} Run {r+1}/{n_runs}...")
            pf = algo.run()
            objs = [s.objectives for s in pf if s.objectives is not None]
            if not objs:
                hv = 0.0
            else:
                hv = get_hv(objs, ref_point)
            hvs.append(hv)
        results[aname] = float(np.mean(hvs))
    return results

def main():
    out_file = os.path.join(os.path.dirname(__file__), "table_Y_Z_results.json")
    instances = generate_all_instances()
    
    configs = {
        "V1": {"grid": 5, "time": 20.0},
        "V2": {"grid": 4, "time": 40.0},
        "V3": {"grid": 4, "time": 60.0},
        "V4": {"grid": 3, "time": 90.0},
        "V5": {"grid": 3, "time": 90.0}
    }
    
    final_results = {"TableY": {}, "TableZ": {}, "ReferencePoints": {}}
    
    for name in ["V1", "V2", "V3", "V4", "V5"]:
        problem = instances[name]
        cfg = configs[name]
        print(f"\n==================================================")
        print(f" Processing Instance {name} ({problem.n_jobs} jobs) ")
        print(f"==================================================")
        
        # 1. Exact MILP
        pf_exact, avg_gap, cpu_time, feas_count = run_milp_epsilon_constraint(
            problem, grid_size=cfg["grid"], time_limit=cfg["time"]
        )
        
        # Determine Reference Point for HV
        if pf_exact:
            pts = np.array(pf_exact)
            ref_point = np.max(pts, axis=0) * 1.1
        else:
            # Fallback Reference Point
            ref_point = np.array([500.0, 3000.0, 200.0])
        
        print(f"\n>>> Exact PF points: {len(pf_exact)}, Avg Gap: {avg_gap:.2f}%, CPU: {cpu_time:.1f}s")
        print(f">>> Reference Point for HV: {ref_point}")
        
        exact_hv = get_hv(pf_exact, ref_point)
        
        final_results["TableY"][name] = {
            "n": problem.n_jobs,
            "PF_exact_size": len(pf_exact),
            "Avg_MIP_Gap": f"{avg_gap:.2f}%",
            "CPU_Time": f"{cpu_time:.1f}s",
            "Feasible": f"{feas_count}/{cfg['grid']*cfg['grid']}",
            "Exact_HV": exact_hv
        }
        final_results["ReferencePoints"][name] = ref_point.tolist()
        
        # 2. Metaheuristics
        print(f"\n--- Step 3: Running Metaheuristics ({name}) ---")
        meta_results = run_metaheuristics(problem, name, ref_point, n_runs=10)
        
        # Calculate RPD_HV
        rpd_results = {}
        for aname, hv_mean in meta_results.items():
            if exact_hv > 0:
                rpd = ((exact_hv - hv_mean) / exact_hv) * 100.0
                # HV of algorithm can sometimes be slightly higher if Exact PF is just an approximation or missing extreme points under timeout.
                if rpd < 0: rpd = 0.0 
            else:
                rpd = 100.0
            rpd_results[aname] = f"{rpd:.2f}%"
            print(f"    {aname}: Mean HV = {hv_mean:.2f}, RPD = {rpd:.2f}%")
            
        final_results["TableZ"][name] = rpd_results
        
        # Save incrementally
        with open(out_file, "w") as f:
            json.dump(final_results, f, indent=4)
            
    print("\nAll experiments completed! Results saved to", out_file)

if __name__ == "__main__":
    main()
