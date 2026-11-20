# Ancillary files

- `code/m2/`: model, state space, Kalman filter, priors, sampler, diagnostics, informal share,
  capital-share estimator and earnings measurement (Python >= 3.10; see code requirements below).
- `code/scripts/`: command-line entry points (estimation, capital share, value-added earnings ratio,
  comparison with published estimates, long annual series, figures and tables, GEIH earnings script).
- `code/tests/`: regression tests (`python -m pytest code/tests -q` from this folder after
  copying `data/` to `datos/`, or run from the project layout).
- `data/datos_dsge_estimacion.csv`: quarterly dataset 1993Q1-2026Q2 (DANE national accounts base
  2015; ICML 2.6.1 employment and informality).
- `data/geih/`: quarterly relative labor earnings of informal and formal workers from GEIH microdata.
- `data/referencias/`: published estimates for Colombia used in the comparison.
- `final_results/`: all quarterly and annual series delivered by the baseline model, parameters, steady state, variance decomposition, sensitivity and identification checks (dictionary in final_results/LEEME.md, in Spanish).
- `results/`: estimation outputs used in the paper (posterior summaries, fit, informal output share,
  run summaries; thinned posterior draws for the main runs).

Requirements: numpy, scipy, pandas, matplotlib (pyarrow only for the GEIH script).
