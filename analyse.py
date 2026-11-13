
# ANALYSE AN EXPERIMENT - python analyse.py configs/q_vs_greedy.toml
# ------------------------------------------------------------

# Reads exactly the runs this config describes: runs in the same folder made with other settings are left out,
# so changing a setting can never quietly mix old and new results.
# Everything it makes can be deleted and made again from data/raw/:
##   data/processed/<experiment>/learning_curves.parquet   one row per agent type per episode: mean, standard error, runs
##   figures/<experiment>_learning_curves.png
# and it prints a summary table.

import argparse
from pathlib import Path
from matplotlib.figure import Figure                                # a figure that needs no window: saves to a file from the command line, shows under the cell in a notebook
import polars as pl
import data
import run

Here = Path(__file__).parent
Smooth = 10                                                         # each point = mean of the last 10 episodes, so the curve shows the trend rather than every bump

# Colours follow the agent type, never its position in a list, so a type keeps its colour in every figure.
# These three pass a colour-blindness check against each other; types added later take the next colours in order.
Type_colours = {"greedy": "#2a78d6", "random": "#eb6834", "q": "#1baf7a"}   # blue, orange, aqua
More_colours = ["#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]     # yellow, pink, green, violet, red
Type_labels = {"greedy": "Greedy", "random": "Random", "q": "Q-learning"}
Ink, Ink_light, Muted, Grid_line, Axis_line, Surface = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


# NUMBERS
# ------------------------------------------------------------

def per_run_items(experiment, run_ids, root):                       # one row per run, agent type, episode
    return (data.scan("episodes", experiment, run_ids, root)
            .group_by("run_id", "type", "episode")
            .agg(pl.col("items").mean())                            # two agents of the same type in a run: their average
            .sort("run_id", "type", "episode"))

def learning_curves(experiment, run_ids, root=data.Data):
    return (per_run_items(experiment, run_ids, root)
            .with_columns(pl.col("items").rolling_mean(Smooth, min_samples=1).over("run_id", "type").alias("smoothed"))
            .group_by("type", "episode")
            .agg(mean=pl.col("smoothed").mean(),
                 se=pl.col("smoothed").std() / pl.len().sqrt(),     # standard error: how far the mean might be off, from how much runs disagree
                 runs=pl.len())
            .sort("type", "episode")
            .collect())

def summary(experiment, run_ids, root=data.Data):                   # items per episode, first and last `Smooth` episodes, mean ± standard error over runs
    items = per_run_items(experiment, run_ids, root)
    last = items.select(pl.col("episode").max()).collect().item()
    windows = {"first": pl.col("episode") <= Smooth, "last": pl.col("episode") > last - Smooth}
    parts = []
    for name, keep in windows.items():
        parts.append(items.filter(keep)
                     .group_by("run_id", "type").agg(pl.col("items").mean())    # each run's average over the window
                     .group_by("type").agg(pl.col("items").mean().alias(f"{name}_{Smooth}"),
                                           (pl.col("items").std() / pl.len().sqrt()).alias(f"{name}_se"),
                                           pl.len().alias("runs")))
    return (parts[0].join(parts[1].drop("runs"), on="type")
            .select("type", "runs", f"first_{Smooth}", "first_se", f"last_{Smooth}", "last_se")
            .with_columns(pl.selectors.float().round(2))
            .sort(f"last_{Smooth}", descending=True).collect())


# FIGURE
# ------------------------------------------------------------

def colours_for(types):
    extra = iter(More_colours)
    return {t: Type_colours[t] if t in Type_colours else next(extra) for t in types}

def plot_learning_curves(curves, runs):                             # returns the figure; in a notebook it shows up under the cell
    types =[t for t in Type_colours if t in curves["type"]] + sorted(set(curves["type"]) - set(Type_colours))
    colours = colours_for(types)
    final = dict(curves.filter(pl.col("episode") == pl.col("episode").max()).select("type", "mean").iter_rows())
    types.sort(key=lambda t: -final[t])                             # legend top to bottom in the same order as the lines

    fig = Figure(figsize=(8, 4.5), dpi=100, facecolor=Surface)
    ax = fig.subplots()
    ax.set_facecolor(Surface)
    for t in types:
        c = curves.filter(pl.col("type") == t)
        label = Type_labels.get(t, t)
        ax.fill_between(c["episode"], c["mean"] - c["se"], c["mean"] + c["se"], color=colours[t], alpha=0.15, linewidth=0)
        ax.plot(c["episode"], c["mean"], color=colours[t], linewidth=1.75, label=label)
        ax.annotate(label, (c["episode"][-1], c["mean"][-1]), xytext=(6, 0), textcoords="offset points",   # name at the end of each line
                    va="center", fontsize=9, color=Ink_light, annotation_clip=False)

    ax.set_xlim(1, curves["episode"].max())
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Episode", color=Ink_light)
    ax.set_ylabel(f"Items per episode (mean of last {Smooth})", color=Ink_light)
    ax.set_title(f"Items collected per episode, mean of {runs} runs (band = ±1 standard error)", color=Ink, fontsize=11, loc="left")
    ax.grid(axis="y", color=Grid_line, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(Axis_line)
    ax.tick_params(colors=Muted, labelsize=9, length=0)
    legend = ax.legend(loc="center right", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(Ink_light)
    fig.subplots_adjust(left=0.08, right=0.86, top=0.9, bottom=0.12)  # room on the right for the line names
    return fig


# RUN
# ------------------------------------------------------------

def analyse(config, root=data.Data, figures=Here / "figures"):
    experiment = config["experiment"]
    wanted = run.run_ids(config)
    found = data.finished_runs(experiment, wanted, root)
    if len(found) < len(wanted):
        print(f"note: {len(wanted) - len(found)} of {len(wanted)} runs haven't finished; using the {len(found)} that have")

    curves = learning_curves(experiment, wanted, root)
    processed = Path(root) / "processed" / experiment
    processed.mkdir(parents=True, exist_ok=True)
    curves.write_parquet(processed / "learning_curves.parquet")
    figure = Path(figures) / f"{experiment}_learning_curves.png"
    figure.parent.mkdir(parents=True, exist_ok=True)
    plot_learning_curves(curves, len(found)).savefig(figure, dpi=200)

    print(f"wrote {processed / 'learning_curves.parquet'}")
    print(f"wrote {figure}")
    return summary(experiment, wanted, root)                        # in a notebook this shows as a table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Learning curves and a summary table for the runs a config describes.")
    parser.add_argument("config", help="e.g. configs/q_vs_greedy.toml")
    args = parser.parse_args()
    table = analyse(run.load_config(args.config))
    with pl.Config(tbl_hide_dataframe_shape=True, tbl_hide_column_data_types=True, float_precision=1):
        print(f"Items per episode, first and last {Smooth} episodes ({table['runs'].max()} runs; se = standard error over runs):")
        print(table)
