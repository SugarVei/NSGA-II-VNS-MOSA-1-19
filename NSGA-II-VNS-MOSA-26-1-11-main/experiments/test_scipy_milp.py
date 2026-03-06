import os
import sys
import time
import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from experiments.milp_instances import generate_all_instances
from experiments.milp_solver_scipy import ScipyMILPFormulation

def test_v1():
    instances = generate_all_instances()
    problem = instances["V1"]
    
    print(f"Testing Scipy MILP on V1: {problem.n_jobs} jobs, {problem.n_stages} stages")
    t0 = time.time()
    model = ScipyMILPFormulation(problem)
    print(f"Model formulation done in {time.time() - t0:.2f}s. Total variables: {model.var_count}")
    
    print("\n--- Minimizing F1 (Makespan) ---")
    res1 = model.solve(obj_name="F1", time_limit=60.0)
    
    print("\n--- Minimizing F2 (Labor Cost) ---")
    res2 = model.solve(obj_name="F2", time_limit=60.0)
    
    print("\n--- Minimizing F3 (Energy) ---")
    res3 = model.solve(obj_name="F3", time_limit=60.0)
    
if __name__ == "__main__":
    test_v1()
