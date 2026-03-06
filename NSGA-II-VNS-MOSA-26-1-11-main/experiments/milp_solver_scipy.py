import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
import scipy.sparse as sp
import time

class ScipyMILPFormulation:
    def __init__(self, problem):
        self.problem = problem
        self.n = problem.n_jobs
        self.stages = problem.n_stages
        self.machines = problem.machines_per_stage
        self.speeds = problem.n_speed_levels
        self.skills = problem.n_skill_levels
        
        self.var_count = 0
        self.var_names = []
        self.var_types = []  # 1 for integer, 0 for continuous
        self.var_lbs = []
        self.var_ubs = []

        self.A_eq = []
        self.b_eq = []
        self.A_ub = []
        self.b_ub = []

        self._build_variables()
        self._build_constraints()

    def add_var(self, name, is_int=False, lb=0.0, ub=np.inf):
        idx = self.var_count
        self.var_count += 1
        self.var_names.append(name)
        self.var_types.append(1 if is_int else 0)
        self.var_lbs.append(lb)
        self.var_ubs.append(ub)
        return idx

    def add_eq(self, coefs, rhs):
        self.A_eq.append(coefs)
        self.b_eq.append(rhs)
        
    def add_le(self, coefs, rhs):
        self.A_ub.append(coefs)
        self.b_ub.append(rhs)
        
    def add_ge(self, coefs, rhs):
        self.A_ub.append({k: -v for k, v in coefs.items()})
        self.b_ub.append(-rhs)

    def _build_variables(self):
        self.X = {}
        for i in range(self.n):
            for s in range(self.stages):
                for m in range(self.machines[s]):
                    for v in range(self.speeds):
                        self.X[(i, s, m, v)] = self.add_var(f"X_{i}_{s}_{m}_{v}", is_int=True, lb=0, ub=1)
        
        self.Z = {}
        for i in range(self.n):
            for j in range(self.n):
                if i == j: continue
                for s in range(self.stages):
                    for m in range(self.machines[s]):
                        self.Z[(i, j, s, m)] = self.add_var(f"Z_{i}_{j}_{s}_{m}", is_int=True, lb=0, ub=1)
        
        self.U = {}
        for s in range(self.stages):
            for m in range(self.machines[s]):
                self.U[(s, m)] = self.add_var(f"U_{s}_{m}", is_int=True, lb=0, ub=1)
                
        self.W = {}
        for s in range(self.stages):
            for m in range(self.machines[s]):
                for sk in range(self.skills):
                    self.W[(s, m, sk)] = self.add_var(f"W_{s}_{m}_{sk}", is_int=True, lb=0, ub=1)
                    
        self.S = {}
        self.C = {}
        for i in range(self.n):
            for s in range(self.stages):
                self.S[(i, s)] = self.add_var(f"S_{i}_{s}", lb=0)
                self.C[(i, s)] = self.add_var(f"C_{i}_{s}", lb=0)
                
        self.UC = {}
        for s in range(self.stages):
            for m in range(self.machines[s]):
                self.UC[(s, m)] = self.add_var(f"UC_{s}_{m}", lb=0)
                
        self.Cmax = self.add_var("Cmax", lb=0)
        self.TotLabor = self.add_var("TotLabor", lb=0)
        self.TotEnergy = self.add_var("TotEnergy", lb=0)

    def _build_constraints(self):
        M = 10000.0  # BigM
        problem = self.problem

        # 1. Assignment
        for i in range(self.n):
            for s in range(self.stages):
                coef = {self.X[(i, s, m, v)]: 1 for m in range(self.machines[s]) for v in range(self.speeds)}
                self.add_eq(coef, 1.0)
                
        # 2. U definitions
        for s in range(self.stages):
            for m in range(self.machines[s]):
                # U <= sum X -> U - sum X <= 0
                coef = {self.U[(s, m)]: 1}
                for i in range(self.n):
                    for v in range(self.speeds):
                        coef[self.X[(i, s, m, v)]] = -1
                self.add_le(coef, 0.0)
                
                # U >= X -> X - U <= 0
                for i in range(self.n):
                    coef = {self.U[(s, m)]: -1}
                    for v in range(self.speeds):
                        coef[self.X[(i, s, m, v)]] = 1
                    self.add_le(coef, 0.0)
                    
        # 3. Worker Assignment
        for s in range(self.stages):
            for m in range(self.machines[s]):
                coef = {self.W[(s, m, sk)]: 1 for sk in range(self.skills)}
                coef[self.U[(s, m)]] = -1
                self.add_eq(coef, 0.0)
                
        # 4. Speed Capability
        for i in range(self.n):
            for s in range(self.stages):
                for m in range(self.machines[s]):
                    for v in range(self.speeds):
                        coef = {self.X[(i, s, m, v)]: 1}
                        for sk in range(self.skills):
                            if problem.can_operate(sk, v):
                                coef[self.W[(s, m, sk)]] = -1
                        self.add_le(coef, 0.0)
                        
        # 5. Worker Limit
        for sk in range(self.skills):
            coef = {self.W[(s, m, sk)]: 1 for s in range(self.stages) for m in range(self.machines[s])}
            self.add_le(coef, problem.workers_available[sk])
            
        # 6. Z constraints
        for s in range(self.stages):
            for m in range(self.machines[s]):
                for i in range(self.n):
                    # sum_j Z_ij <= sum_v X_imv -> sum Z_ij - sum X <= 0
                    coef = {self.Z[(i, j, s, m)]: 1 for j in range(self.n) if i != j}
                    for v in range(self.speeds):
                        coef[self.X[(i, s, m, v)]] = -1
                    self.add_le(coef, 0.0)
                    
                    # sum_j Z_ji <= sum_v X_imv -> sum Z_ji - sum X <= 0
                    coef = {self.Z[(j, i, s, m)]: 1 for j in range(self.n) if i != j}
                    for v in range(self.speeds):
                        coef[self.X[(i, s, m, v)]] = -1
                    self.add_le(coef, 0.0)
                    
                # sum Z_ij = sum X - U -> sum Z_ij - sum X + U = 0
                coef = {}
                for i in range(self.n):
                    for j in range(self.n):
                        if i != j:
                            coef[self.Z[(i, j, s, m)]] = 1
                for i in range(self.n):
                    for v in range(self.speeds):
                        coef[self.X[(i, s, m, v)]] = -1
                coef[self.U[(s, m)]] = 1
                self.add_eq(coef, 0.0)
                
                # Time precedence: Sj >= Ci + Setup - M(1 - Z_ij)
                # Ci - Sj + M Z_ij <= M - Setup
                for i in range(self.n):
                    for j in range(self.n):
                        if i != j:
                            setup = float(problem.get_setup_time(s, m, i, j))
                            coef = {
                                self.S[(j, s)]: -1,
                                self.C[(i, s)]: 1,
                                self.Z[(i, j, s, m)]: M
                            }
                            self.add_le(coef, M - setup)

        # 7. Time evaluation
        for i in range(self.n):
            for s in range(self.stages):
                # C = S + sum X*P -> C - S - sum X*P = 0
                coef = {self.C[(i, s)]: 1, self.S[(i, s)]: -1}
                for m in range(self.machines[s]):
                    for v in range(self.speeds):
                        p = float(problem.get_processing_time(i, s, m, v))
                        coef[self.X[(i, s, m, v)]] = -p
                self.add_eq(coef, 0.0)
                
            for s in range(1, self.stages):
                trans = float(problem.get_transport_time(s - 1))
                # S_s >= C_{s-1} + trans -> S_s - C_{s-1} >= trans
                coef = {self.S[(i, s)]: 1, self.C[(i, s - 1)]: -1}
                self.add_ge(coef, trans)
                
        # 8. Cmax
        for i in range(self.n):
            # Cmax >= C_{n-1} -> Cmax - C_{n-1} >= 0
            coef = {self.Cmax: 1, self.C[(i, self.stages - 1)]: -1}
            self.add_ge(coef, 0.0)
            
        # 9. Linearization UC = U * Cmax
        for s in range(self.stages):
            for m in range(self.machines[s]):
                self.add_le({self.UC[(s, m)]: 1, self.Cmax: -1}, 0.0)
                self.add_le({self.UC[(s, m)]: 1, self.U[(s, m)]: -M}, 0.0)
                self.add_ge({self.UC[(s, m)]: 1, self.Cmax: -1, self.U[(s, m)]: -M}, -M)
                
        # 10. TotLabor
        # -TotLabor + sum W * wage = 0
        coef = {self.TotLabor: -1}
        for s in range(self.stages):
            for m in range(self.machines[s]):
                for sk in range(self.skills):
                    wage = float(problem.get_wage(sk))
                    coef[self.W[(s, m, sk)]] = wage
        self.add_eq(coef, 0.0)
        
        # 11. TotEnergy
        coef = {self.TotEnergy: -1}
        for i in range(self.n):
            for s in range(self.stages):
                for m in range(self.machines[s]):
                    for v in range(self.speeds):
                        p = float(problem.get_processing_time(i, s, m, v))
                        power = float(problem.get_processing_power(s, m, v))
                        coef[self.X[(i, s, m, v)]] = (p * power) / 60.0
                        
        for s in range(self.stages):
            for m in range(self.machines[s]):
                for i in range(self.n):
                    for j in range(self.n):
                        if i != j:
                            setup = float(problem.get_setup_time(s, m, i, j))
                            power = float(problem.get_setup_power(s, m))
                            coef[self.Z[(i, j, s, m)]] = coef.get(self.Z[(i, j, s, m)], 0) + (setup * power) / 60.0
                            
        for s in range(self.stages):
            for m in range(self.machines[s]):
                idle_power = float(problem.get_idle_power(s, m)) / 60.0
                coef[self.UC[(s, m)]] = coef.get(self.UC[(s, m)], 0) + idle_power
                
                for i in range(self.n):
                    for v in range(self.speeds):
                        p = float(problem.get_processing_time(i, s, m, v))
                        coef[self.X[(i, s, m, v)]] -= idle_power * p
                        
                for i in range(self.n):
                    for j in range(self.n):
                        if i != j:
                            setup = float(problem.get_setup_time(s, m, i, j))
                            coef[self.Z[(i, j, s, m)]] -= idle_power * setup
                            
        trans_eng_const = 0.0
        for s in range(self.stages - 1):
            trans_eng_const += problem.n_jobs * float(problem.get_transport_time(s)) * float(problem.transport_power) / 60.0
            
        aux_power = float(problem.aux_power) / 60.0
        coef[self.Cmax] = coef.get(self.Cmax, 0) + aux_power
        
        self.add_eq(coef, -trans_eng_const)

    def _build_matrix(self, dict_list, n_cols):
        rows = []
        cols = []
        vals = []
        for r, d in enumerate(dict_list):
            for c, v in d.items():
                if abs(v) > 1e-9:
                    rows.append(r)
                    cols.append(c)
                    vals.append(v)
        return sp.coo_matrix((vals, (rows, cols)), shape=(len(dict_list), n_cols))

    def solve(self, obj_name='F1', limit_F2=None, limit_F3=None, time_limit=600.0, mip_gap=0.01):
        c = np.zeros(self.var_count)
        eps_weight = 1e-4
        if obj_name == 'F1':
            c[self.Cmax] = 1.0
            c[self.TotLabor] = eps_weight
            c[self.TotEnergy] = eps_weight
        elif obj_name == 'F2':
            c[self.TotLabor] = 1.0
            c[self.Cmax] = eps_weight
            c[self.TotEnergy] = eps_weight
        elif obj_name == 'F3':
            c[self.TotEnergy] = 1.0
            c[self.Cmax] = eps_weight
            c[self.TotLabor] = eps_weight
            
        A_ub_temp = self.A_ub.copy()
        b_ub_temp = self.b_ub.copy()
        
        if limit_F2 is not None:
            A_ub_temp.append({self.TotLabor: 1})
            b_ub_temp.append(limit_F2)
            
        if limit_F3 is not None:
            A_ub_temp.append({self.TotEnergy: 1})
            b_ub_temp.append(limit_F3)
            
        constraints = []
        if len(self.A_eq) > 0:
            A_eq_mat = self._build_matrix(self.A_eq, self.var_count)
            b_eq_vec = np.array(self.b_eq)
            constraints.append(LinearConstraint(A_eq_mat, b_eq_vec, b_eq_vec))

        if len(A_ub_temp) > 0:
            A_ub_mat = self._build_matrix(A_ub_temp, self.var_count)
            b_ub_vec = np.array(b_ub_temp)
            constraints.append(LinearConstraint(A_ub_mat, -np.inf, b_ub_vec))

        lb = np.array(self.var_lbs)
        ub = np.array(self.var_ubs)
        integrality = np.array(self.var_types)

        print(f"Solving MILP: obj={obj_name}, F2<={limit_F2}, F3<={limit_F3}, variables={self.var_count}")
        start_t = time.time()
        res = milp(
            c=c, 
            bounds=Bounds(lb, ub), 
            integrality=integrality, 
            constraints=constraints, 
            options={'time_limit': time_limit, 'mip_rel_gap': mip_gap}
        )
        end_t = time.time()
        
        status_map = {0: 'OPTIMAL', 1: 'ITER_LIMIT', 2: 'INFEASIBLE', 3: 'UNBOUNDED', 4: 'NODE_LIMIT', 5: 'TIME_LIMIT'}
        status_str = status_map.get(res.status, f"UNKNOWN({res.status})")
        print(f"Solver status: {status_str}, Time: {end_t - start_t:.2f}s")
        
        if res.success or res.status == 5:
            # Re-read objective values
            obj_vals = (res.x[self.Cmax], res.x[self.TotLabor], res.x[self.TotEnergy])
            print(f"Found Solution: F1={obj_vals[0]:.2f}, F2={obj_vals[1]:.2f}, F3={obj_vals[2]:.2f}")
            return {'status': res.status, 'x': res.x, 'objectives': obj_vals, 'time': end_t - start_t, 'mip_gap': getattr(res, 'mip_gap', 0)}
        else:
            return {'status': res.status, 'time': end_t - start_t}
