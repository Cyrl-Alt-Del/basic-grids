
# DATA - what gets recorded, where it is kept, and how to read it back
# ------------------------------------------------------------

# Four tables. Each has one grain: every row in it means the same kind of thing.
##   steps     one agent's move in one step        where everyone was and what they did
##   events    one thing the world did             "collect" (who got the item: with agents chasing the same item, turn order decides)
##                                                 and "spawn" (where items appeared; steps only has agents, so this is the only record of items)
##   metrics   one number an agent reported        whatever learn() returns, e.g. the Q agent's td_error; a new agent can report
##                                                 new numbers without any change here, and agents that report nothing have no rows
##   episodes  one agent's whole episode           items and score; small, and most figures only need this
# plus run.json: the settings, seed, code fingerprint and times. It is written last, so a folder without one is a run that didn't finish.

# data/raw/<experiment>/<run_id>/  steps.parquet  events.parquet  metrics.parquet  episodes.parquet  q_tables.json  run.json
# Each run writes only its own folder, so runs can't overwrite each other, even when several run at once.
# Parquet rather than CSV: columns keep their types (an episode number stays a whole number), files are much smaller,
# and Polars reads only the columns and runs a question asks for.

import json                                                         # run.json
import os                                                           # swap a finished file into place in one go
from pathlib import Path
import polars as pl

Schema_version = 1                                                  # bump when any table's columns change, so old files say which layout they use
Data = Path(__file__).parent / "data"

Schemas = {                                                         # every column and its type; UInt8 = whole number 0-255, plenty for row/col on grids up to 256 wide
    "steps": {
        "run_id": pl.String, "episode": pl.UInt32, "step": pl.UInt16, "turn": pl.UInt8,   # turn = order within the step, 0 moves first
        "agent": pl.String, "row": pl.UInt8, "col": pl.UInt8,                             # where the agent ended up after its move
        "action": pl.String, "reward": pl.Float32,
    },
    "events": {
        "run_id": pl.String, "episode": pl.UInt32, "step": pl.UInt16, "turn": pl.UInt8,   # step 0, turn empty = items placed before anyone moved
        "event": pl.String, "agent": pl.String,                                           # agent is empty for spawns: the world did it
        "row": pl.UInt8, "col": pl.UInt8, "value": pl.Float32,
    },
    "metrics": {
        "run_id": pl.String, "episode": pl.UInt32, "step": pl.UInt16, "agent": pl.String,
        "key": pl.String, "value": pl.Float64,                                            # one row per number, e.g. key "td_error"
    },
    "episodes": {
        "run_id": pl.String, "episode": pl.UInt32, "agent": pl.String, "type": pl.String,
        "turn": pl.UInt8, "items": pl.UInt16, "score": pl.Float32,
    },
}


# RECORDING
# ------------------------------------------------------------

# Rows are kept in plain lists while the run plays (adding to a list is cheap; adding to a table one row at a time copies the whole table)
# and turned into tables once, at the end.

class Recorder:
    def __init__(self, run_id, episodes, steps_every=1):
        self.run_id = run_id
        self.episodes = episodes                                    # how many episodes this run will play
        self.steps_every = steps_every                              # 1 = keep steps and metrics for every episode; 10 = episodes 1, 11, 21... and the last
        self.rows = {table: [] for table in Schemas}
        self.detailed = []                                          # episodes that have steps and metrics, so the data says what it is missing

    def start_episode(self, episode):
        self.number = episode                                       # which episode is playing
        self.detail = (episode - 1) % self.steps_every == 0 or episode == self.episodes
        if self.detail:
            self.detailed.append(episode)

    def step(self, step, turn, agent, row, col, action, reward):
        if self.detail:
            self.rows["steps"].append({"run_id": self.run_id, "episode": self.number, "step": step, "turn": turn,
                                       "agent": agent, "row": row, "col": col, "action": action, "reward": reward})

    def event(self, step, turn, event, agent, row, col, value):     # events are few, so they are always kept
        self.rows["events"].append({"run_id": self.run_id, "episode": self.number, "step": step, "turn": turn,
                                    "event": event, "agent": agent, "row": row, "col": col, "value": value})

    def metrics(self, step, agent, values):                         # values = whatever the agent reported, e.g. {"td_error": 0.3, "explored": 0.0}
        if self.detail:
            for key, value in values.items():
                self.rows["metrics"].append({"run_id": self.run_id, "episode": self.number, "step": step,
                                             "agent": agent, "key": key, "value": value})

    def episode(self, agent, agent_type, turn, items, score):
        self.rows["episodes"].append({"run_id": self.run_id, "episode": self.number, "agent": agent, "type": agent_type,
                                      "turn": turn, "items": items, "score": score})

    def write(self, folder, info):                                  # info = everything about the run, for run.json
        folder.mkdir(parents=True, exist_ok=True)
        for table, rows in self.rows.items():
            pl.DataFrame(rows, schema=Schemas[table]).write_parquet(folder / f"{table}.parquet")
        info = {**info, "schema_version": Schema_version, "steps_kept_for_episodes": self.detailed}
        temporary = folder / "run.json.tmp"
        temporary.write_text(json.dumps(info, indent=2))
        os.replace(temporary, folder / "run.json")                  # last, and all at once: run.json only exists once everything else is written


# READING
# ------------------------------------------------------------

# run_ids = only these runs (analyse.py passes the runs its config describes); None = every finished run of the experiment

def finished_runs(experiment, run_ids=None, root=Data):
    folders = sorted(p.parent for p in (Path(root) / "raw" / experiment).glob("*/run.json"))
    if run_ids is not None:
        folders = [f for f in folders if f.name in set(run_ids)]
    return folders

def runs(experiment, run_ids=None, root=Data):
    """One row per finished run: its id, seed, every setting (as "world.width" etc.), and where it came from."""
    rows = []
    for folder in finished_runs(experiment, run_ids, root):
        info = json.loads((folder / "run.json").read_text())
        row = {k: v for k, v in info.items() if k not in ("config", "steps_kept_for_episodes")}
        for section, values in info["config"].items():              # flatten {"world": {"width": 10}} to {"world.width": 10}
            if isinstance(values, dict):
                row.update({f"{section}.{k}": v for k, v in values.items()})
            else:
                row[section] = values
        rows.append(row)
    return pl.DataFrame(rows)

def scan(table, experiment, run_ids=None, root=Data):
    """A table across runs, not loaded yet: add .filter() / .select() / .group_by(), then .collect() reads only what's needed."""
    folders = finished_runs(experiment, run_ids, root)
    if not folders:
        raise FileNotFoundError(f"No finished runs for '{experiment}' in {Path(root) / 'raw'}. Run: python run.py configs/{experiment}.toml")
    return pl.scan_parquet([f / f"{table}.parquet" for f in folders])
