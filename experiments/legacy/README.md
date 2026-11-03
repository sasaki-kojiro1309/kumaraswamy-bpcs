# Historical execution scripts

These modules preserve the original scheduling and finalization code from the
November 2026 execution attempts. They include date-specific deadlines and
assumptions about saved supervision logs. They are retained for traceability,
not as fresh experiment launchers. The statistical algorithms are elsewhere.

| Module | Historical purpose |
|---|---|
| `priority_supervisor.py` | Prioritize nearly completed paired datasets |
| `priority_assist.py` | Reallocate a worker after violation-scenario failures |
| `priority_watchdog.py` | External cap for that execution's parent process |
| `urgent30_finalize.py` | Verify/publish that time-limited attempt |

For new work use `experiments.simulation_runner` with an explicit configuration
and budget, or inspect `experiments.execution.submission_supervisor --help`.
The old flat module paths remain import/command compatibility aliases. Moving
these modules did not repair or alter their historical timing behavior.
