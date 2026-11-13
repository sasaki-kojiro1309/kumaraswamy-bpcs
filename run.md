python3 -m pytest -q

head -n 10 results/simulation_submission/summaries/parameter_performance.csv

Completed study (all 360 fits):

```sh
MPLCONFIGDIR=tmp/matplotlib OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m experiments.completion_study analyze
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m experiments.completion_study fit --workers 3
```

See `docs/SIMULATION_COMPLETION.md` and `results/simulation_completion_v1/latest/SUMMARY.md`.
