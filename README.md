# Basic grids

Agents collecting items on a grid: random, greedy and Q-learning.

```bash
.venv/bin/python world.py                                # watch one episode (settings at the top of world.py)
.venv/bin/python run.py configs/q_vs_greedy.toml         # play every run in the config, record it to data/raw/
.venv/bin/python analyse.py configs/q_vs_greedy.toml     # learning curves + summary table from those runs
.venv/bin/python -m pytest                               # check the pipeline (a few seconds)
.venv/bin/jupyter lab basic_grids.ipynb                  # the notebook: run the experiment, curves, summary, what Q learned
```

## Notebook and Colab

`basic_grids.ipynb` runs the whole pipeline from the top (about 5 seconds; finished runs are skipped).
Before sharing it, run every cell and save, so the figures and tables are stored in the file:

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace basic_grids.ipynb
```

To share it, send these links. Push first, so they show the latest version:

- **Read it** (figures and tables included, nothing to install): https://github.com/Cyrl-Alt-Del/basic-grids/blob/main/basic_grids.ipynb
- **Run it**: https://colab.research.google.com/github/Cyrl-Alt-Del/basic-grids/blob/main/basic_grids.ipynb, then **Runtime → Run all**.
  The first cell fetches the code from GitHub and installs Polars; on your own computer that cell does nothing.

Colab fetches the code from the repo, so it only works while the repo is public.
`.gitignore` leaves out `.venv/`, `data/` and `figures/`; the notebook remakes them.

Works on Python 3.11 and newer (3.11 is the first with `tomllib`, which reads the configs).

## Files

| File | What it does |
|---|---|
| `world.py` | The world: grid, items, moves, one episode (`run_episode`) |
| `agents.py` | The agents. Each gets its own dice (`rng`); learners return numbers about themselves from `learn()` |
| `configs/*.toml` | One experiment each: world, agents, episodes, seeds. Every setting must be there |
| `run.py` | Plays a config: one run per seed, skips finished runs, `--jobs 4` for four at once |
| `data.py` | Table layouts, the recorder, and `scan()` / `runs()` for reading data back |
| `analyse.py` | Makes `data/processed/` and `figures/` from the runs a config describes; its functions also work in a notebook |
| `basic_grids.ipynb` | The experiment as a notebook, to share or run on Colab |

## Data

```
data/raw/<experiment>/<run_id>/     one folder per run, never edited after it's written
    steps.parquet      one row per agent per step: where it went, what it did, reward
    events.parquet     one row per thing the world did: collect (who got it), spawn (where an item appeared)
    metrics.parquet    one row per number a learning agent reported (td_error, explored)
    episodes.parquet   one row per agent per episode: items, score
    q_tables.json      what each Q agent knew at the end
    run.json           settings, seed, code fingerprint, times; written last, so no run.json = didn't finish
data/processed/<experiment>/       made by analyse.py; safe to delete
figures/                           made by analyse.py; safe to delete
```

The run id is the experiment, the seed and a fingerprint of the settings, e.g. `q_vs_greedy__s3__1a2b3c4d`.
Same settings give the same id, so finished runs are skipped. Change a setting and you get new ids, so nothing is overwritten.

Reading data yourself:

```python
import polars as pl, data
steps = data.scan("steps", "q_vs_greedy")                   # nothing read yet
steps.filter(pl.col("agent") == "q1").group_by("episode").agg(pl.col("reward").sum()).collect()
data.runs("q_vs_greedy")                                    # one row per run with every setting
```

Changing the experiment: copy a config, give it a new `experiment` name, edit the settings.
