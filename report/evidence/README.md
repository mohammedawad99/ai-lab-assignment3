# Report evidence snapshots

Small copies of the final result files that the report cites (final rerun
with the tuned settings plus the Stage 11 advanced local-search moves):

- `cvrp_all_summary.csv` — per instance/algorithm CVRP summary (144 rows raw,
  including both ALNS variants)
- `cvrp_all_policy_effective_summary.csv` — the same summary after applying
  the pre-declared ALNS selection rule (enhanced everywhere except
  M-n200-k17)
- `cvrp_algorithm_mean_gaps.csv` — best-of-seeds gap per algorithm/instance
- `ackley_d10_summary.csv` — per algorithm Ackley summary (unchanged setup)
- `gp_gep_comparison_runs.csv` — the original small GP/GEP comparison
  (historical; the hard benchmark below is the main Part B result)
- `rushhour_hard_manual_summary.csv` — manual heuristics on the 14 hard puzzles
- `rushhour_hard_gp_gep_summary.csv` — GP/GEP vs manual on the hard set
- `rushhour_hard_config.json` / `rushhour_hard_manifest.json` — benchmark setup
- `final_execution_manifest.json` — what ran, budgets, tuned settings, policy
- `final_v3_summary.txt` — the current final comparison summary
  (Stage 11-C rerun vs the previous committed evidence), printed by
  `scripts/extract_final_results_v3.py`
- `cvrp_seed_robustness_summary.csv` — the 8-seed robustness study
  (measured with the Stage 10 configuration, before the advanced pass)

The full generated outputs (raw per-run CSVs, solution files, extra assets)
stay under `results/final_experiments/` locally and are not committed. The
official CVRP `.vrp` files are user-provided data and are also not committed.
