$ErrorActionPreference = "Stop"

python paper_tools/run_specified_baseline_protocol.py `
  --scenarios calce calce2 nasa tju `
  --count 5 `
  --max-epochs 200 `
  --precision 16-mixed

python paper_tools/run_ablation_protocol.py `
  --count 5 `
  --max-epochs 200 `
  --precision 16-mixed

python paper_tools/plot_distribution_and_noise_robustness.py
