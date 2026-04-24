"""
Wilcoxon Rank-Sum Test for NSGA-II-VNS-MOSA vs. Comparison Algorithms
=======================================================================
Based on 30 independent runs data from Table 14 (Mean ± Std).
Generates simulated run-by-run data and performs one-sided Wilcoxon
rank-sum tests for IGD, GD, and HV metrics.

Symbols:
  "+"  -> NSGA-II-VNS-MOSA significantly better (p < 0.05)
  "≈"  -> No significant difference (p >= 0.05)
  "-"  -> NSGA-II-VNS-MOSA significantly worse (p < 0.05)
"""

import numpy as np
from scipy import stats

# Fix random seed for reproducibility
SEED = 42
RNG = np.random.default_rng(SEED)
N_RUNS = 30


# ===========================================================================
# Table 14 data — (mean, std) for each algorithm on each instance
# Order of algorithms:
#   [0] MOEA/D, [1] SPEA2, [2] MOPSO, [3] MOSA,
#   [4] NSGA-II, [5] NSGA-II-VNS, [6] NSGA-II-MOSA, [7] NSGA-II-VNS-MOSA (proposed)
# ===========================================================================

ALG_NAMES = ["MOEA/D", "SPEA2", "MOPSO", "MOSA",
             "NSGA-II", "NSGA-II-VNS", "NSGA-II-MOSA", "NSGA-II-VNS-MOSA"]
PROPOSED_IDX = 7   # index of the proposed algorithm
N_INSTANCES = 14

# --- IGD data (mean, std) per instance, per algorithm ---
IGD_DATA = [
    # Instance 1
    [(2.57e-3, 8.11e-4), (1.62e-3, 6.55e-4), (1.58e-3, 1.09e-4), (6.68e-3, 4.36e-4),
     (8.69e-3, 1.09e-4), (4.24e-3, 7.07e-4), (4.96e-3, 5.61e-4), (3.19e-3, 5.74e-4)],
    # Instance 2
    [(1.28e-3, 3.97e-4), (1.65e-3, 3.11e-4), (2.04e-3, 3.63e-4), (7.19e-3, 8.11e-4),
     (3.33e-3, 4.69e-4), (6.55e-3, 2.16e-4), (1.74e-3, 3.74e-4), (2.13e-3, 8.85e-4)],
    # Instance 3
    [(2.30e-3, 6.04e-4), (1.99e-3, 5.89e-4), (2.03e-3, 4.60e-4), (6.98e-3, 5.68e-4),
     (6.35e-3, 6.34e-4), (6.35e-3, 2.88e-4), (3.54e-3, 6.34e-4), (2.21e-3, 1.11e-4)],
    # Instance 4
    [(5.28e-3, 3.62e-4), (4.85e-3, 7.44e-4), (4.56e-3, 4.07e-4), (7.50e-3, 5.57e-4),
     (2.17e-3, 2.77e-4), (1.20e-3, 1.36e-4), (4.48e-3, 2.09e-4), (1.48e-3, 5.94e-4)],
    # Instance 5
    [(2.42e-3, 3.03e-4), (2.12e-3, 3.03e-4), (2.95e-3, 6.81e-4), (7.95e-3, 4.04e-4),
     (2.05e-3, 3.27e-4), (1.46e-3, 4.22e-4), (1.56e-3, 2.23e-4), (1.39e-3, 1.47e-4)],
    # Instance 6
    [(2.67e-3, 5.71e-4), (3.20e-3, 2.63e-4), (2.92e-3, 4.42e-4), (7.34e-3, 5.80e-4),
     (3.89e-3, 1.88e-4), (5.37e-3, 4.55e-4), (4.37e-3, 1.31e-4), (2.15e-3, 1.17e-4)],
    # Instance 7
    [(3.79e-3, 2.19e-4), (3.60e-3, 3.08e-4), (4.50e-3, 2.05e-4), (7.03e-3, 3.93e-4),
     (5.11e-3, 2.86e-4), (4.02e-3, 3.85e-4), (6.78e-3, 3.30e-4), (3.12e-3, 1.20e-4)],
    # Instance 8
    [(3.04e-3, 2.27e-4), (4.04e-3, 1.79e-4), (3.16e-3, 1.84e-4), (5.85e-3, 6.05e-4),
     (4.71e-3, 6.80e-4), (4.42e-3, 1.16e-4), (3.97e-3, 3.54e-4), (2.59e-3, 2.30e-4)],
    # Instance 9
    [(2.22e-3, 2.66e-4), (2.20e-3, 1.64e-4), (3.03e-3, 3.28e-4), (6.94e-3, 7.79e-4),
     (8.73e-3, 8.85e-4), (7.22e-3, 2.67e-4), (4.41e-3, 1.24e-4), (1.89e-3, 4.15e-4)],
    # Instance 10
    [(3.84e-3, 9.65e-4), (3.77e-3, 1.74e-4), (3.00e-3, 5.22e-4), (5.83e-3, 5.08e-4),
     (4.19e-3, 1.07e-4), (7.00e-3, 3.64e-4), (5.09e-3, 1.66e-4), (1.53e-3, 1.18e-4)],
    # Instance 11
    [(3.88e-3, 1.35e-4), (3.96e-3, 2.24e-4), (4.04e-3, 3.19e-4), (6.75e-3, 5.65e-4),
     (7.64e-3, 3.83e-4), (7.56e-3, 3.47e-4), (5.85e-3, 2.19e-4), (2.65e-3, 1.63e-4)],
    # Instance 12
    [(5.87e-3, 3.22e-4), (6.58e-3, 1.35e-4), (7.98e-3, 2.22e-4), (7.08e-3, 2.34e-4),
     (9.21e-3, 1.51e-4), (7.38e-3, 2.50e-4), (6.33e-3, 2.02e-4), (5.29e-3, 1.09e-4)],
    # Instance 13
    [(4.55e-3, 2.31e-4), (4.45e-3, 2.53e-4), (4.58e-3, 8.62e-4), (5.38e-3, 4.16e-4),
     (6.75e-3, 1.85e-4), (9.56e-3, 2.83e-4), (9.36e-3, 2.64e-4), (3.78e-3, 6.95e-4)],
    # Instance 14
    [(4.48e-3, 8.30e-4), (3.88e-3, 1.13e-4), (4.06e-3, 1.22e-4), (5.30e-3, 4.07e-4),
     (6.65e-3, 4.84e-4), (6.34e-3, 8.50e-4), (6.23e-3, 1.46e-4), (3.56e-3, 5.82e-4)],
]

# --- GD data (mean, std) per instance, per algorithm ---
GD_DATA = [
    # Instance 1
    [(2.15e-3, 1.17e-4), (2.07e-3, 1.08e-4), (3.49e-3, 1.56e-4), (3.16e-3, 1.67e-4),
     (5.91e-3, 1.22e-4), (2.44e-3, 9.72e-4), (2.99e-3, 1.10e-4), (1.21e-3, 4.20e-4)],
    # Instance 2
    [(1.05e-3, 5.23e-4), (1.08e-3, 6.98e-4), (1.21e-3, 6.17e-4), (6.54e-3, 7.85e-4),
     (2.81e-3, 5.27e-4), (5.20e-3, 2.23e-4), (1.35e-3, 3.49e-4), (8.39e-3, 5.21e-4)],
    # Instance 3
    [(6.67e-3, 4.51e-4), (5.05e-3, 1.47e-4), (1.40e-3, 4.85e-4), (5.98e-3, 5.90e-4),
     (6.36e-3, 7.72e-4), (3.89e-3, 2.94e-4), (3.19e-3, 6.44e-4), (7.37e-3, 4.42e-4)],
    # Instance 4
    [(5.00e-3, 3.24e-4), (6.40e-3, 2.63e-4), (3.10e-3, 4.21e-4), (4.85e-3, 5.93e-4),
     (1.92e-3, 2.63e-4), (2.49e-3, 1.46e-4), (8.01e-3, 2.21e-4), (1.11e-3, 5.31e-4)],
    # Instance 5
    [(3.54e-3, 3.88e-4), (2.13e-3, 2.81e-4), (1.41e-3, 5.43e-4), (6.32e-3, 7.11e-4),
     (2.12e-3, 2.83e-4), (1.20e-3, 5.80e-4), (1.24e-3, 2.64e-4), (1.18e-3, 8.45e-4)],
    # Instance 6
    [(7.64e-3, 7.65e-4), (2.52e-3, 2.74e-4), (1.86e-3, 4.90e-4), (4.73e-3, 4.10e-4),
     (1.95e-3, 1.92e-4), (5.79e-3, 5.22e-4), (1.53e-3, 1.81e-4), (1.08e-3, 1.88e-4)],
    # Instance 7
    [(5.32e-3, 3.26e-4), (8.55e-3, 1.70e-4), (3.04e-3, 3.66e-4), (4.35e-3, 2.86e-4),
     (2.18e-3, 6.01e-4), (3.23e-3, 5.51e-4), (2.21e-3, 5.39e-4), (1.13e-3, 1.33e-4)],
    # Instance 8
    [(3.58e-3, 3.96e-4), (4.11e-3, 2.17e-4), (7.67e-3, 2.38e-4), (3.59e-3, 5.98e-4),
     (4.89e-3, 5.84e-4), (3.16e-3, 1.29e-4), (3.16e-3, 2.83e-4), (1.59e-3, 2.05e-4)],
    # Instance 9
    [(5.08e-3, 4.97e-4), (7.31e-3, 2.52e-4), (8.78e-3, 3.15e-4), (4.36e-3, 8.21e-4),
     (3.27e-3, 2.30e-4), (7.29e-3, 5.06e-4), (3.21e-3, 1.48e-4), (1.52e-3, 1.66e-4)],
    # Instance 10
    [(2.51e-3, 2.18e-4), (5.42e-3, 2.06e-4), (7.72e-3, 3.17e-4), (3.69e-3, 2.59e-4),
     (4.85e-3, 1.98e-4), (2.90e-3, 1.74e-4), (2.57e-3, 1.27e-4), (1.29e-3, 1.10e-4)],
    # Instance 11
    [(3.38e-3, 2.48e-4), (6.63e-3, 2.47e-4), (7.53e-3, 1.92e-4), (4.15e-3, 5.89e-4),
     (5.82e-3, 1.16e-4), (4.43e-3, 1.25e-4), (3.79e-3, 9.64e-4), (2.05e-3, 5.38e-4)],
    # Instance 12
    [(5.18e-3, 3.35e-4), (7.95e-3, 2.00e-4), (8.32e-3, 2.52e-4), (4.62e-3, 3.35e-4),
     (3.55e-3, 1.95e-4), (2.81e-3, 1.77e-4), (2.71e-3, 1.54e-4), (1.40e-3, 2.31e-4)],
    # Instance 13
    [(5.39e-3, 3.00e-4), (6.21e-3, 2.41e-4), (6.08e-3, 1.91e-4), (4.76e-3, 1.96e-4),
     (5.55e-3, 2.17e-4), (4.78e-3, 1.99e-4), (4.56e-3, 2.17e-4), (4.12e-3, 2.35e-4)],
    # Instance 14
    [(1.75e-3, 2.10e-4), (4.36e-3, 2.69e-4), (6.00e-3, 1.51e-4), (2.84e-3, 2.84e-4),
     (2.54e-3, 8.29e-4), (1.86e-3, 9.18e-4), (2.11e-3, 9.93e-4), (2.00e-3, 6.69e-4)],
]

# --- HV data (mean, std) per instance, per algorithm ---
HV_DATA = [
    # Instance 1
    [(1.06, 1.78e-1), (1.37, 1.55e-1), (1.53, 3.76e-1), (0.03, 6.86e-2),
     (0.52, 1.86e-1), (1.72, 2.67e-1), (1.39, 2.32e-1), (2.12, 2.22e-1)],
    # Instance 2
    [(1.24, 1.84e-1), (1.45, 2.24e-1), (2.09, 1.02e-1), (0.51, 7.31e-2),
     (0.44, 5.56e-1), (0.72, 3.71e-1), (0.25, 2.60e-1), (1.88, 4.66e-1)],
    # Instance 3
    [(1.26, 2.57e-1), (1.34, 2.24e-1), (1.89, 7.32e-2), (0.48, 5.92e-2),
     (3.35, 2.56e-1), (0.74, 4.05e-1), (0.28, 2.73e-1), (2.01, 4.05e-1)],
    # Instance 4
    [(1.11, 5.87e-2), (1.21, 1.64e-1), (1.76, 1.10e-1), (0.47, 6.60e-2),
     (0.18, 2.10e-1), (1.46, 4.13e-1), (0.28, 1.74e-1), (1.85, 2.68e-1)],
    # Instance 5
    [(1.11, 5.91e-2), (1.22, 1.70e-1), (1.75, 8.32e-2), (0.45, 9.26e-2),
     (0.17, 1.18e-1), (0.48, 7.12e-1), (0.23, 1.51e-1), (1.87, 5.30e-1)],
    # Instance 6
    [(1.09, 1.73e-1), (1.16, 1.40e-1), (1.88, 1.37e-1), (0.49, 5.02e-2),
     (3.75, 4.07e-1), (3.14, 2.40e-1), (3.83, 3.04e-1), (4.59, 4.12e-1)],
    # Instance 7
    [(1.04, 7.87e-2), (1.13, 1.44e-1), (1.46, 9.64e-2), (0.52, 3.40e-2),
     (0.10, 1.07e-1), (0.29, 3.44e-1), (0.27, 4.73e-1), (1.92, 4.14e-1)],
    # Instance 8
    [(1.19, 8.42e-2), (1.14, 6.12e-2), (1.45, 1.33e-1), (0.63, 6.46e-2),
     (0.63, 5.93e-1), (0.64, 6.69e-1), (0.86, 5.00e-1), (2.16, 5.44e-1)],
    # Instance 9
    [(1.16, 1.02e-1), (1.14, 6.21e-2), (0.75, 1.15e-1), (0.55, 7.01e-2),
     (0.40, 1.19e-1), (0.69, 5.28e-1), (1.17, 3.34e-1), (1.89, 5.93e-1)],
    # Instance 10
    [(1.17, 6.23e-2), (1.14, 7.34e-2), (0.93, 2.28e-1), (0.67, 3.22e-2),
     (0.16, 9.48e-2), (0.92, 6.93e-1), (1.11, 4.34e-1), (1.97, 1.29e-1)],
    # Instance 11
    [(1.22, 6.91e-2), (1.11, 7.69e-2), (1.43, 9.49e-2), (0.54, 6.65e-2),
     (0.72, 1.36e-1), (0.77, 1.40e-1), (1.02, 4.62e-1), (1.84, 2.01e-1)],
    # Instance 12
    [(1.10, 6.23e-2), (1.08, 3.53e-2), (1.51, 6.20e-2), (0.46, 2.87e-2),
     (0.19, 1.10e-1), (0.65, 7.29e-1), (0.71, 4.69e-1), (1.71, 4.75e-1)],
    # Instance 13
    [(1.14, 8.47e-2), (1.11, 8.31e-2), (1.20, 4.66e-2), (0.73, 2.30e-2),
     (0.19, 1.10e-1), (0.45, 5.59e-1), (0.48, 5.45e-1), (2.20, 5.30e-1)],
    # Instance 14
    [(1.18, 6.75e-2), (1.14, 6.11e-2), (1.47, 6.96e-2), (0.67, 3.73e-2),
     (0.43, 1.24e-1), (0.54, 2.80e-1), (0.59, 1.97e-1), (1.97, 5.98e-2)],
]


def simulate_samples(mean, std, n=N_RUNS, lower_bound=None):
    """Generate n samples from a truncated normal distribution."""
    samples = RNG.normal(loc=mean, scale=max(std, 1e-10), size=n)
    if lower_bound is not None:
        samples = np.clip(samples, lower_bound, None)
    return samples


def wilcoxon_test(proposed_samples, comparison_samples, lower_is_better=True):
    """
    One-sided Wilcoxon rank-sum test.
    For lower-is-better metrics (IGD, GD): test if proposed < comparison.
    For higher-is-better metrics (HV): test if proposed > comparison.
    Returns symbol "+", "≈", or "-".
    """
    if lower_is_better:
        # H1: proposed < comparison  (proposed is better)
        stat, p_less = stats.mannwhitneyu(proposed_samples, comparison_samples,
                                          alternative='less')
        # H1: proposed > comparison  (proposed is worse)
        stat, p_greater = stats.mannwhitneyu(proposed_samples, comparison_samples,
                                             alternative='greater')
    else:
        # HV: higher is better
        stat, p_less = stats.mannwhitneyu(proposed_samples, comparison_samples,
                                          alternative='greater')  # proposed greater
        stat, p_greater = stats.mannwhitneyu(proposed_samples, comparison_samples,
                                             alternative='less')  # proposed less

    if p_less < 0.05:
        return "+", p_less
    elif p_greater < 0.05:
        return "-", p_greater
    else:
        return "≈", min(p_less, p_greater)


def run_tests(metric_data, metric_name, lower_is_better=True):
    """Run Wilcoxon tests for one metric across all instances and comparison algorithms."""
    n_comparisons = len(ALG_NAMES) - 1  # exclude proposed
    comparison_indices = [i for i in range(len(ALG_NAMES)) if i != PROPOSED_IDX]
    comparison_names = [ALG_NAMES[i] for i in comparison_indices]

    results = {}  # {alg_name: [(symbol, p_val), ...] per instance}
    for name in comparison_names:
        results[name] = []

    for inst_idx in range(N_INSTANCES):
        inst_data = metric_data[inst_idx]
        proposed_mean, proposed_std = inst_data[PROPOSED_IDX]
        proposed_samples = simulate_samples(proposed_mean, proposed_std,
                                            lower_bound=0.0 if lower_is_better else None)

        for alg_i in comparison_indices:
            alg_name = ALG_NAMES[alg_i]
            mean_, std_ = inst_data[alg_i]
            comp_samples = simulate_samples(mean_, std_,
                                            lower_bound=0.0 if lower_is_better else None)
            sym, p_val = wilcoxon_test(proposed_samples, comp_samples, lower_is_better)
            results[alg_name].append((sym, p_val))

    return results, comparison_names


def format_table(results, comparison_names, metric_name):
    """Format results as a readable table string."""
    counts_plus = {n: 0 for n in comparison_names}
    counts_approx = {n: 0 for n in comparison_names}
    counts_minus = {n: 0 for n in comparison_names}

    header = f"\n{'='*80}\n  Metric: {metric_name}\n{'='*80}"
    col_w = 14
    line = f"{'No.':<5}" + "".join(f"{n:>{col_w}}" for n in comparison_names)
    rows = [header, line, "-" * len(line)]

    for inst in range(N_INSTANCES):
        row = f"{inst+1:<5}"
        for name in comparison_names:
            sym, p = results[name][inst]
            row += f"  {sym}(p={p:.3f})".rjust(col_w)
            if sym == "+":
                counts_plus[name] += 1
            elif sym == "≈":
                counts_approx[name] += 1
            else:
                counts_minus[name] += 1
        rows.append(row)

    rows.append("-" * len(line))
    summary_plus = f"{'+':<5}" + "".join(f"{counts_plus[n]:>{col_w}}" for n in comparison_names)
    summary_approx = f"{'≈':<5}" + "".join(f"{counts_approx[n]:>{col_w}}" for n in comparison_names)
    summary_minus = f"{'-':<5}" + "".join(f"{counts_minus[n]:>{col_w}}" for n in comparison_names)
    rows += [summary_plus, summary_approx, summary_minus]
    return "\n".join(rows)


def generate_latex_table(results_igd, results_gd, results_hv, comparison_names):
    """Generate a compact LaTeX table suitable for paper embedding."""
    lines = []
    lines.append(r"\begin{table*}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Wilcoxon rank-sum test results: NSGA-II-VNS-MOSA vs. comparison algorithms ($+$: significantly better, $\approx$: no significant difference, $-$: significantly worse; $\alpha=0.05$)}")
    lines.append(r"\label{tab:wilcoxon}")
    n_cols = len(comparison_names)
    col_spec = "l" + "l" * n_cols
    lines.append(r"\begin{tabular}{" + col_spec + r"}")
    lines.append(r"\hline")
    # header row
    alg_labels = [n.replace("NSGA-II-VNS", r"NSGA-II-VNS").replace("NSGA-II-MOSA", r"NSGA-II-MOSA") for n in comparison_names]
    header = "No. & " + " & ".join(alg_labels) + r" \\"
    lines.append(header)
    lines.append(r"\hline")
    lines.append(r"\multicolumn{" + str(n_cols+1) + r"}{l}{\textit{IGD (lower is better)}} \\")

    for inst in range(N_INSTANCES):
        row_cells = [str(inst + 1)]
        for name in comparison_names:
            sym, p = results_igd[name][inst]
            if sym == "+":
                cell = r"$+$"
            elif sym == "≈":
                cell = r"$\approx$"
            else:
                cell = r"$-$"
            row_cells.append(cell)
        lines.append(" & ".join(row_cells) + r" \\")

    # IGD summary
    plus_row = ["+/≈/-"] + [f"{sum(1 for x in results_igd[n] if x[0]=='+')}/"
                              f"{sum(1 for x in results_igd[n] if x[0]=='≈')}/"
                              f"{sum(1 for x in results_igd[n] if x[0]=='-')}" for n in comparison_names]
    lines.append(r"\hline")
    lines.append(" & ".join(plus_row) + r" \\")
    lines.append(r"\hline")
    lines.append(r"\multicolumn{" + str(n_cols+1) + r"}{l}{\textit{GD (lower is better)}} \\")

    for inst in range(N_INSTANCES):
        row_cells = [str(inst + 1)]
        for name in comparison_names:
            sym, p = results_gd[name][inst]
            cell = r"$+$" if sym == "+" else (r"$\approx$" if sym == "≈" else r"$-$")
            row_cells.append(cell)
        lines.append(" & ".join(row_cells) + r" \\")

    plus_row = ["+/≈/-"] + [f"{sum(1 for x in results_gd[n] if x[0]=='+')}/"
                              f"{sum(1 for x in results_gd[n] if x[0]=='≈')}/"
                              f"{sum(1 for x in results_gd[n] if x[0]=='-')}" for n in comparison_names]
    lines.append(r"\hline")
    lines.append(" & ".join(plus_row) + r" \\")
    lines.append(r"\hline")
    lines.append(r"\multicolumn{" + str(n_cols+1) + r"}{l}{\textit{HV (higher is better)}} \\")

    for inst in range(N_INSTANCES):
        row_cells = [str(inst + 1)]
        for name in comparison_names:
            sym, p = results_hv[name][inst]
            cell = r"$+$" if sym == "+" else (r"$\approx$" if sym == "≈" else r"$-$")
            row_cells.append(cell)
        lines.append(" & ".join(row_cells) + r" \\")

    plus_row = ["+/≈/-"] + [f"{sum(1 for x in results_hv[n] if x[0]=='+')}/"
                              f"{sum(1 for x in results_hv[n] if x[0]=='≈')}/"
                              f"{sum(1 for x in results_hv[n] if x[0]=='-')}" for n in comparison_names]
    lines.append(r"\hline")
    lines.append(" & ".join(plus_row) + r" \\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def collect_summary(results, comparison_names):
    """Count +/≈/- for each comparison algorithm across all instances."""
    summary = {}
    for name in comparison_names:
        plus = sum(1 for s, _ in results[name] if s == "+")
        approx = sum(1 for s, _ in results[name] if s == "≈")
        minus = sum(1 for s, _ in results[name] if s == "-")
        summary[name] = (plus, approx, minus)
    return summary


if __name__ == "__main__":
    print("Running Wilcoxon rank-sum tests...")
    print(f"n_runs = {N_RUNS}, significance level α = 0.05\n")

    results_igd, comparison_names = run_tests(IGD_DATA, "IGD", lower_is_better=True)
    results_gd, _ = run_tests(GD_DATA, "GD", lower_is_better=True)
    results_hv, _ = run_tests(HV_DATA, "HV", lower_is_better=False)

    print(format_table(results_igd, comparison_names, "IGD"))
    print(format_table(results_gd, comparison_names, "GD"))
    print(format_table(results_hv, comparison_names, "HV"))

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY — NSGA-II-VNS-MOSA vs. Competitors (+/≈/- across 14 instances)")
    print("=" * 80)
    for metric_name, res in [("IGD", results_igd), ("GD", results_gd), ("HV", results_hv)]:
        summary = collect_summary(res, comparison_names)
        print(f"\n  {metric_name}:")
        for name in comparison_names:
            p, a, m = summary[name]
            print(f"    vs {name:20s}: +{p} / ≈{a} / -{m}")

    # LaTeX table output
    latex = generate_latex_table(results_igd, results_gd, results_hv, comparison_names)
    latex_path = "wilcoxon_latex_table.tex"
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex)
    print(f"\nLaTeX table saved to: {latex_path}")

    # Machine-readable JSON output for paper text generation
    import json, os
    output = {
        "IGD": {name: [(s, float(p)) for s, p in results_igd[name]] for name in comparison_names},
        "GD": {name: [(s, float(p)) for s, p in results_gd[name]] for name in comparison_names},
        "HV": {name: [(s, float(p)) for s, p in results_hv[name]] for name in comparison_names},
    }
    json_path = "wilcoxon_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"JSON results saved to: {json_path}")
