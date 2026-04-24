# NSGA-II-VNS-MOSA：逐条对照论文伪代码的实现映射

本文件的目的：把论文中的“伪代码/流程条目”与代码实现逐条对应，便于你在论文答辩或复现实验时做到**可追溯**与**可复核**。

> 说明：本仓库实现采用“两阶段”结构：Phase 1 为 NSGA-II；Phase 2 为 MOSA + VNS。
> 其中，VNS 内部生成候选集合 C 并进行“可行性检查 + 修复”，修复失败直接丢弃候选；若 C 为空，则令 S_new = S_cur（等价于不发生状态转移）。

---

## 1. 顶层流程（NSGA-II → MOSA）

### 论文：总体流程
1. 运行 NSGA-II 至最大代数，得到 Pareto Frontier  
2. 用 Pareto Frontier 初始化 AP 与 RP  
3. 在降温循环中：对 RP 中每个代表解执行 VNS 生成候选 C，再按 MOSA Metropolis 准则接受/拒绝，维护 AP  
4. 温度降至阈值或达到迭代上限终止，输出 AP

### 代码对应
- 顶层类：`algorithms/nsga2_vns_mosa.py::class NSGA2_VNS_MOSA`
- Phase 1：`NSGA2_VNS_MOSA.run()` 中调用 `self.nsga2.run()`
- AP 初始化：`run()` 中 `fronts = self.nsga2.non_dominated_sort(pop)` 取第一前沿作为初始 AP，并做拥挤度裁剪
- RP 构建：`_build_rp_from_ap(ap)`
- 降温循环：`run()` 中 `for layer in range(self.mosa_layers): ... T *= self.cooling_rate`
- VNS 生成候选集：`_generate_candidate_set(s_cur)`
- 候选选择（ε-greedy + 标量化 Φ）：`_epsilon_greedy_select(C, ref, weights)` 与 `_phi(...)`
- Metropolis 接受：`_mosa_accept(s_cur, s_new, T, ref, weights)`
- AP 维护：`_update_archive(ap, s_new)`（删除被支配解 + 拥挤度裁剪）

---

## 2. VNS：候选集合 C 的生成与“空集合处理”

### 论文：VNS 生成候选集合
- 按邻域序列依次调用 8 个算子生成候选（剔除不可行/修复失败者），得到候选集合 C
- 若最终无有效候选，则令 S_new = S_cur

### 代码对应
- 邻域算子（8个）：
  - N1：`_n1_machine`, `_n1_speed`, `_n1_worker`, `_n1_queue_swap`
  - N2：`_n2_path`, `_n2_mode`
  - N3：`_n3_swap_rows`, `_n3_block_insert`
- 候选生成与过滤：
  - `algorithms/nsga2_vns_mosa.py::_generate_candidate_set`
  - 每个算子内部调用 `Solution.repair(problem)`；repair 返回 `None` 视为修复失败 → 该候选直接丢弃
- 空集合处理：
  - `run()` 中：若 `C` 为空则 `break`（等价于本轮不发生状态转移，即 S_new = S_cur）

---

## 3. 可行性检查与 Repair（对照论文 5.4.5）

### 论文：修复顺序（先合法性，再绑定，再兼容，再可用性；任一阶段失败即丢弃）
A. 编码取值合法  
B. 人机绑定（同一阶段同一机器的 W 必须一致）  
C. 技能–速度兼容（V 必须可由 W 操作）  
D. 全局人力可用性（按“启用机器数”计数）

### 代码对应（核心：`models/solution.py::Solution.repair`）
- A 编码取值合法：
  - 修复 `machine_assign[i,j]` ∈ 可选机器集合
  - 修复 `speed_level[i,j]` ∈ {0,...,n_speed_levels-1}
  - 修复 `worker_skill[i,j]` ∈ {0,...,n_skill_levels-1}
  - 修复 `sequence_priority[i,j]` 为非负可排序整数
- B 人机绑定（ω 的生成规则）：
  - 对任意 (j,f)，取该机任务速度最大值 `smax = max V`
  - 取**最低可行技能等级** `ω = min { α | can_operate(α, smax) }`
  - 强制该机所有工序 `W[i,j] = ω`
- C 技能–速度兼容：
  - 对任意 (i,j) 若 `not can_operate(W[i,j], V[i,j])`：
    - 优先“提升技能”（并保持同机一致）
    - 若提升仍不可能，则“降速”到该技能可操作的最大速度
  - 修复后重新按“最大速度→最低可行技能”计算 ω 并同步 W
- D 全局人力可用性：
  - 对每个技能等级 α，统计启用机器数 `count[α]`
  - 若 `count[α] > available[α]`，优先降级部分机器技能（并同步降速以保持兼容）
  - 若仍溢出，尝试通过“迁移使机器空载”来减少启用机器数
  - 若仍无法满足 → repair 失败返回 `None`（上层直接丢弃候选）

---

## 4. 你需要重点核对的三个“论文一致性点”

1. **Repair 失败返回值与候选丢弃策略**  
   - 论文：任一阶段修复失败 → 该候选解直接丢弃，不进入候选集合 C  
   - 代码：`Solution.repair()` 返回 `None` 表示修复失败；`_generate_candidate_set` 过滤 `None` 候选

2. **VNS 候选集合为空的处理**  
   - 论文：若无有效候选，则令 `S_new = S_cur`（不发生状态转移）  
   - 代码：`run()` 中若 `C` 为空则 `break`

3. **ω 的生成规则（机器绑定值）**  
   - 论文：ω 由该机任务最大速度决定，并取最低可行技能  
   - 代码：`Solution.repair()` 的 B 步严格按 `smax = max V` 计算 `ω`

---

## 5. 实验复现建议（你必须做，否则你是在“靠运气”写论文）

- 在日志中打印每一轮 VNS 的：
  - 候选数 | 修复失败数 | 进入 C 的有效候选数
- 随机抽取若干个体，逐机检查：
  - 同一 (j,f) 上所有工序 `W` 是否一致
  - `W` 是否等于 `min feasible skill for max V`
  - `count[α]` 是否满足 `available[α]`

如果以上三项你没有做，所谓“一致性”只是你主观感觉，并不具有可证性。
