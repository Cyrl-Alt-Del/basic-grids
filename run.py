
# RUN AN EXPERIMENT - python run.py configs/q_vs_greedy.toml
# ------------------------------------------------------------

# A config file describes the question: the world, which agents, how many episodes, which seeds.
# Each seed is one run: fresh agents (the Q agent starts knowing nothing) playing every episode in a row, learning as they go.
# A run's id is made from its settings, so the same settings always get the same id:
##   finished runs are skipped, so an experiment that was stopped carries on where it left off
##   change any setting and the ids change, so new results never overwrite old ones
# --force runs everything again; --jobs 4 runs four at a time.

import argparse                                                     # read the command line
import ast                                                          # read code the way Python does, to fingerprint it without its comments
import hashlib                                                     # fingerprints of settings and code
import json
import platform                                                     # which Python and computer made the data
import time
import tomllib                                                      # reads .toml config files; comes with Python
from concurrent.futures import ProcessPoolExecutor                  # several runs at once
from datetime import datetime, timezone
from pathlib import Path
import polars as pl
import world
import data

Here = Path(__file__).parent
Code_files = ["world.py", "agents.py", "data.py", "run.py"]         # the code that makes the data; its fingerprint goes in every run.json


# CONFIG
# ------------------------------------------------------------

Sections = {                                                        # every setting a config must have: a missing or misspelt one stops here, rather than quietly using a default
    "world":  {"width", "height", "num_items", "item_value", "num_steps"},
    "q":      {"alpha", "gamma", "epsilon"},
    "record": {"steps_every"},
}
Top_level = {"experiment", "seeds", "episodes", "agents", *Sections}

def load_config(path):
    config = tomllib.loads(Path(path).read_text())
    problems = []
    if set(config) != Top_level:
        problems.append(f"top level should have {sorted(Top_level)}, has {sorted(config)}")
    for section, keys in Sections.items():
        if set(config.get(section, {})) != keys:
            problems.append(f"[{section}] should have {sorted(keys)}, has {sorted(config.get(section, {}))}")
    unknown = set(config.get("agents", {})) - set(world.Agent_types)
    if unknown:
        problems.append(f"[agents] has unknown types {sorted(unknown)}; known: {sorted(world.Agent_types)}")
    if problems:
        raise SystemExit(f"{path}:\n  " + "\n  ".join(problems))
    if isinstance(config["seeds"], int):
        config["seeds"] = list(range(config["seeds"]))              # seeds = 20 means seeds 0 to 19
    return config

def run_settings(config, seed):                                     # everything one run needs: the config, with one seed instead of the list
    settings = {k: v for k, v in config.items() if k != "seeds"}
    settings["seed"] = seed
    return settings

def fingerprint(thing):                                             # same settings -> same fingerprint, any change -> a different one
    return hashlib.sha256(json.dumps(thing, sort_keys=True).encode()).hexdigest()

def run_id(config, seed):                                           # e.g. "q_vs_greedy__s3__1a2b3c4d": readable, plus the settings fingerprint
    return f"{config['experiment']}__s{seed}__{fingerprint(run_settings(config, seed))[:8]}"

def run_ids(config):
    return [run_id(config, seed) for seed in config["seeds"]]

def code_fingerprint():                                             # changes whenever world, agents or recording code changes; comments and spacing don't count
    h = hashlib.sha256()
    for name in Code_files:
        h.update(ast.dump(ast.parse((Here / name).read_text())).encode())   # the code as Python understands it, which has no comments
    return h.hexdigest()[:12]


# RUN
# ------------------------------------------------------------

def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def run_one(config, seed, root=data.Data):
    settings = run_settings(config, seed)
    rid = run_id(config, seed)
    folder = Path(root) / "raw" / config["experiment"] / rid
    (folder / "run.json").unlink(missing_ok=True)                   # running again: mark it unfinished before touching anything
    started, clock = now(), time.perf_counter()

    agent_list = world.make_agents(settings["agents"], settings["q"])   # fresh agents: nothing learned yet
    recorder = data.Recorder(rid, settings["episodes"], settings["record"]["steps_every"])
    for episode in range(1, settings["episodes"] + 1):
        recorder.start_episode(episode)
        world.run_episode(settings["world"], agent_list, seed, episode, recorder)

    folder.mkdir(parents=True, exist_ok=True)
    q_tables = {a["name"]: a["policy"].to_dict() for a in agent_list if hasattr(a["policy"], "to_dict")}   # what each learner knew at the end
    (folder / "q_tables.json").write_text(json.dumps(q_tables))
    recorder.write(folder, {
        "run_id": rid, "experiment": config["experiment"], "seed": seed,
        "config": settings, "config_hash": fingerprint(settings), "code": code_fingerprint(),
        "started": started, "finished": now(), "seconds": round(time.perf_counter() - clock, 2),
        "python": platform.python_version(), "polars": pl.__version__, "machine": platform.node(),
    })
    return rid

def run_experiment(config, root=data.Data, force=False, jobs=1):
    experiment = config["experiment"]
    finished = {f.name: f for f in data.finished_runs(experiment, root=root)}
    wanted = run_ids(config)
    todo = [seed for seed, rid in zip(config["seeds"], wanted) if force or rid not in finished]
    print(f"{experiment}: {len(wanted)} runs, {len(wanted) - len(todo)} already done, {len(todo)} to run")

    code = code_fingerprint()
    stale = [rid for rid in wanted if rid in finished and json.loads((finished[rid] / "run.json").read_text())["code"] != code]
    if stale and not force:
        print(f"  note: {len(stale)} finished runs were made before the code last changed; --force to run them again")
    others = set(finished) - set(wanted)
    if others:
        print(f"  note: {Path(root) / 'raw' / experiment} also has {len(others)} runs with other settings; analyse.py leaves them out")

    if jobs > 1:                                                    # each run writes only its own folder, so they can't get in each other's way
        with ProcessPoolExecutor(jobs) as pool:
            for rid in pool.map(run_one, [config] * len(todo), todo, [root] * len(todo)):
                print(f"  {rid}")
    else:
        for seed in todo:
            print(f"  {run_one(config, seed, root)}")
    return todo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play every seed of an experiment and record it in data/raw/.")
    parser.add_argument("config", help="e.g. configs/q_vs_greedy.toml")
    parser.add_argument("--force", action="store_true", help="run again even if already finished")
    parser.add_argument("--jobs", type=int, default=1, help="how many runs at the same time")
    args = parser.parse_args()
    run_experiment(load_config(args.config), force=args.force, jobs=args.jobs)
