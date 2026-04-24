# NSGA-II-VNS-MOSA Algorithm
# 面向HFSP-SDST的多目标混合优化算法实现

本仓库为论文 *"A Hybrid NSGA-II-VNS-MOSA Algorithm for the Multi-Objective Hybrid
Flow Shop Scheduling Problem with Sequence-Dependent Setup Times and Multi-Skilled
Workers"* 的配套算法代码。它以带序列依赖换模时间（SDST）与多技能工人约束的混合流
水车间调度问题（HFSP）为研究对象，对最大完工时间 / 人工成本 / 能耗三目标进行协同
优化。

## 核心特性

- **主算法**: NSGA-II + VNS + MOSA 两阶段混合元启发式（`algorithms/nsga2_vns_mosa.py`）
- **对比算法**: NSGA-II, SPEA2, MOEA/D, MOPSO, MOSA, 以及消融变体 NSGA2_VNS / NSGA2_MOSA
- **MILP 精确求解**: ε-约束法 + scipy.optimize.milp 作为小规模算例的参考前沿
- **参数标定**: 田口 L16(4^4) 正交实验自动调参
- **显著性检验**: Wilcoxon 秩和检验生成 LaTeX 表格
- **可视化 GUI**: PyQt5 多窗口界面，支持手动输入、批量对比与结果导出

## 环境依赖

```bash
pip install -r requirements.txt
```

主要依赖：`PyQt5`, `numpy`, `pandas`, `matplotlib`, `scipy`, `pymoo`, `pytest`, `tqdm`, `joblib`, `pyyaml`

## 运行方式

**图形界面（推荐）**

```bash
python main.py
```

**田口参数实验（命令行）**

```bash
python -m experiments.taguchi.run_taguchi
```

**MILP 验证实验（命令行）**

```bash
python -m experiments.run_milp_experiment
```

**单元测试**

```bash
pytest tests/
```

## 项目结构

```
NSGA-II-VNS-MOSA Algorithm/
├── main.py                        # GUI 程序入口
├── README.md                      # 本文件
├── requirements.txt               # Python 依赖清单
├── run.bat                        # Windows 一键启动脚本
├── paper_pseudocode_mapping.md    # 论文伪代码↔代码实现的逐条映射（可追溯性证据）
│
├── models/                        # 问题建模与解编码
│   ├── problem.py                 # SchedulingProblem: HFSP-SDST 实例参数
│   ├── solution.py                # Solution: 四矩阵编码 (M, Q, V, W) + 可行性修复
│   ├── decoder.py                 # Decoder: 解码为调度方案并计算三目标
│   └── data_loader.py             # DataLoader: JSON/CSV 算例读写
│
├── algorithms/                    # 多目标元启发式算法库
│   ├── nsga2_vns_mosa.py          # 【主算法】两阶段 NSGA-II + VNS + MOSA 混合框架
│   ├── nsga2.py                   # 标准 NSGA-II
│   ├── mosa.py                    # 多目标模拟退火
│   ├── vns.py                     # 变邻域搜索
│   ├── spea2.py                   # SPEA2 对比算法
│   ├── moead.py                   # MOEA/D 对比算法
│   ├── mopso.py                   # MOPSO 对比算法（离散版）
│   ├── hybrid_variants.py         # 消融变体：NSGA2_VNS / NSGA2_MOSA / NSGA2_VNS_MOSA
│   └── operators.py               # 统一算子库：4M-SX 交叉、变异、邻域结构
│
├── experiments/                   # 数值实验
│   ├── run_milp_experiment.py     # MILP vs 元启发式对比实验主脚本
│   ├── milp_instances.py          # 5 组验证算例 V1-V5 生成器
│   ├── milp_solver_scipy.py       # 基于 scipy.milp 的 ε-约束三目标求解器
│   ├── wilcoxon_test.py           # Wilcoxon 秩和检验 + LaTeX 表格输出
│   ├── table_Y_Z_results.json     # 实验结果数据
│   └── taguchi/                   # 田口正交实验子模块
│       ├── run_taguchi.py         # 田口实验 CLI 入口
│       ├── designs.py             # L16(4^4) 正交表与因子水平定义
│       ├── metrics.py             # IGD / GD / HV 指标（基于 pymoo）
│       ├── pareto.py              # 非支配筛选与 PF_ref 构造
│       ├── analysis.py            # SNR 分析、主效应、最优组合推荐
│       ├── plotting.py            # 主效应图、箱线图、Pareto 投影图
│       └── io.py                  # CSV/JSON 保存与异常日志
│
├── ui/                            # PyQt5 图形界面
│   ├── main_window.py             # 主窗口（参数 + 结果一体化）
│   ├── main_app.py                # 多窗口流程入口
│   ├── input_panel.py             # 参数输入面板
│   ├── manual_input_dialog.py     # 手动数据输入对话框
│   ├── result_panel.py            # 结果展示面板
│   ├── result_window.py           # 独立结果展示窗口
│   ├── algorithm_comparison_window.py  # 算法对比实验窗口
│   ├── comparison_worker.py       # 算法对比后台工作线程
│   ├── case_config_dialog.py      # 单算例配置对话框
│   ├── case_data.py               # 算例配置数据结构与持久化
│   ├── multi_case_manager.py      # 多算例管理页面
│   ├── taguchi_window.py          # 田口实验可视化窗口
│   └── styles.py                  # UI 主题与样式表
│
├── visualization/                 # 绘图与导出
│   ├── pareto_3d.py               # 三目标 Pareto 前沿 3D 散点图
│   ├── convergence.py             # 三目标收敛曲线
│   └── export.py                  # 结果导出（CSV / 图像）
│
├── scripts/                       # 独立脚本
│   └── consistency_audit.py       # 实现↔论文约束的一致性审计脚本
│
├── tests/                         # 单元测试 (pytest)
│   ├── test_paper_constraints.py  # 论文约束一致性测试
│   ├── test_paper_consistency.py  # repair 产生论文可行解的测试
│   ├── test_4msx_crossover.py     # 4M-SX 交叉算子专项测试
│   ├── test_metrics.py            # IGD/GD/HV 指标单元测试
│   └── test_pareto.py             # 非支配筛选单元测试
│
├── utils/                         # 通用工具（预留）
│
├── data/                          # 示例算例
│   ├── example1_5jobs.json        # 5 工件示例
│   └── example2_15jobs.json       # 15 工件示例
│
├── docs/                          # 文档
│   └── taguchi_experiment.md      # 田口实验说明
│
├── audit_example/                 # 一致性审计输出示例（附录证据）
│
├── wilcoxon_latex_table.tex       # Wilcoxon 结果的 LaTeX 表
├── wilcoxon_results.json          # Wilcoxon 原始结果
└── ui_screenshot.png              # GUI 界面截图
```

## 论文对照

伪代码、约束编号与代码实现的逐条映射见 [`paper_pseudocode_mapping.md`](paper_pseudocode_mapping.md)。

## 许可证

本代码仅用于学术研究与论文审稿复现。
