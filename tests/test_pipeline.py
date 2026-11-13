
# TESTS - .venv/bin/python -m pytest
# ------------------------------------------------------------

# Quick checks (a few seconds) that the pipeline still records what it says it records.
# Each plays a tiny experiment into a temporary folder, so real data in data/ is never touched.

import json
import polars as pl
import pytest
import agents
import analyse
import data
import run

def tiny_config(**changes):
    config = {
        "experiment": "tiny", "seeds": [0, 1], "episodes": 3,
        "world": {"width": 6, "height": 5, "num_items": 3, "item_value": 5, "num_steps": 20},
        "agents": {"random": 1, "greedy": 1, "q": 1},
        "q": {"alpha": 0.1, "gamma": 0.9, "epsilon": 0.1},
        "record": {"steps_every": 1},
    }
    config.update(changes)
    return config

def tables(config, root):                                           # every table of a finished experiment, read into memory
    ids = run.run_ids(config)
    return {table: data.scan(table, config["experiment"], ids, root).collect() for table in data.Schemas}

@pytest.fixture(scope="module")
def played(tmp_path_factory):                                       # play the tiny experiment once and share it between tests
    root = tmp_path_factory.mktemp("data")
    config = tiny_config()
    run.run_experiment(config, root)
    return config, root, tables(config, root)


# SAME SEED, SAME DATA
# ------------------------------------------------------------

def test_same_seed_gives_identical_tables(played, tmp_path):
    config, _, first = played
    run.run_experiment(config, tmp_path)
    again = tables(config, tmp_path)
    for table in data.Schemas:
        assert first[table].equals(again[table]), table

def test_several_at_once_gives_identical_tables(played, tmp_path):
    config, _, first = played
    run.run_experiment(config, tmp_path, jobs=2)
    again = tables(config, tmp_path)
    for table in data.Schemas:
        assert first[table].equals(again[table]), table

def test_different_seeds_give_different_games(played):
    _, _, t = played
    by_run = t["steps"].partition_by("run_id")
    assert not by_run[0].select("row", "col").equals(by_run[1].select("row", "col"))


# ONE GRAIN PER TABLE, AND THE TABLES AGREE
# ------------------------------------------------------------

def test_every_table_has_its_columns_and_types(played):
    _, _, t = played
    for table, schema in data.Schemas.items():
        assert dict(t[table].schema) == schema, table

def test_one_row_per_agent_per_step(played):
    config, _, t = played
    w = config["world"]
    expected = len(config["seeds"]) * config["episodes"] * w["num_steps"] * sum(config["agents"].values())
    assert t["steps"].height == expected
    assert t["steps"].select("run_id", "episode", "step", "agent").is_unique().all()
    assert t["episodes"].select("run_id", "episode", "agent").is_unique().all()
    assert t["metrics"].select("run_id", "episode", "step", "agent", "key").is_unique().all()

def test_episode_totals_match_the_moves_and_events(played):
    _, _, t = played
    key = ["run_id", "episode", "agent"]
    from_steps = t["steps"].group_by(key).agg(pl.col("reward").sum().alias("reward"))
    from_events = t["events"].filter(pl.col("event") == "collect").group_by(key).agg(pl.len().alias("collected"))
    joined = (t["episodes"].join(from_steps, on=key, how="left")
              .join(from_events, on=key, how="left").fill_null(0))
    assert (joined["score"] == joined["reward"]).all()
    assert (joined["items"] == joined["collected"]).all()

def test_every_collected_item_is_replaced(played):
    config, _, t = played
    counts = t["events"].group_by("run_id", "episode").agg(
        spawns=(pl.col("event") == "spawn").sum(), collects=(pl.col("event") == "collect").sum())
    assert (counts["spawns"] == counts["collects"] + config["world"]["num_items"]).all()

def test_collections_happen_where_the_agent_is(played):
    _, _, t = played
    collects = t["events"].filter(pl.col("event") == "collect")
    joined = collects.join(t["steps"], on=["run_id", "episode", "step", "turn", "agent"], suffix="_step")
    assert joined.height == collects.height
    assert (joined["row"] == joined["row_step"]).all() and (joined["col"] == joined["col_step"]).all()

def test_only_learning_agents_report_metrics(played):
    _, _, t = played
    assert set(t["metrics"]["agent"]) == {"q1"}
    assert set(t["metrics"]["key"]) == {"td_error", "explored"}


# RUNS: SKIPPED WHEN DONE, REDONE WHEN NOT
# ------------------------------------------------------------

def test_finished_runs_are_skipped(played):
    config, root, _ = played
    assert run.run_experiment(config, root) == []

def test_a_run_without_run_json_counts_as_unfinished(tmp_path):
    config = tiny_config()
    run.run_experiment(config, tmp_path)
    folder = data.finished_runs("tiny", root=tmp_path)[0]
    (folder / "run.json").unlink()                                  # as if it crashed before the end
    assert folder not in data.finished_runs("tiny", root=tmp_path)
    assert run.run_experiment(config, tmp_path) == [0]              # only that run is played again

def test_changing_a_setting_changes_the_run_ids():
    assert run.run_ids(tiny_config()) != run.run_ids(tiny_config(episodes=4))

def test_steps_every_keeps_some_episodes_and_says_which(tmp_path):
    config = tiny_config(seeds=[0], episodes=5, record={"steps_every": 2})
    run.run_experiment(config, tmp_path)
    t = tables(config, tmp_path)
    assert sorted(t["steps"]["episode"].unique()) == [1, 3, 5]
    assert t["episodes"]["episode"].n_unique() == 5                 # episode totals are always kept
    info = json.loads((data.finished_runs("tiny", root=tmp_path)[0] / "run.json").read_text())
    assert info["steps_kept_for_episodes"] == [1, 3, 5]


# CONFIGS, AGENTS, ANALYSIS
# ------------------------------------------------------------

def test_shipped_configs_load():
    for path in (run.Here / "configs").glob("*.toml"):
        run.load_config(path)

def test_misspelt_setting_is_rejected(tmp_path):
    text = (run.Here / "configs" / "q_vs_greedy.toml").read_text().replace("epsilon =", "epsilom =")
    (tmp_path / "bad.toml").write_text(text)
    with pytest.raises(SystemExit):
        run.load_config(tmp_path / "bad.toml")

def test_q_agent_reports_numbers():
    q = agents.QLearningAgent(epsilon=0.5)
    grid = [[".", "£"], [".", "."]]
    options = {"right": (0, 1), "down": (1, 0)}
    q(grid, (0, 0), options)
    report = q.learn(0, grid, (1, 0), {"up": (0, 0), "right": (1, 1)})
    assert report and all(isinstance(k, str) and isinstance(v, float) for k, v in report.items())

def test_analyse_writes_curves_and_figure(played, tmp_path):
    config, root, _ = played
    analyse.analyse(config, root, figures=tmp_path)
    curves = pl.read_parquet(root / "processed" / "tiny" / "learning_curves.parquet")
    assert curves.height == config["episodes"] * len(config["agents"])
    assert (tmp_path / "tiny_learning_curves.png").stat().st_size > 0
