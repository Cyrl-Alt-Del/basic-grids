
# IMPORTS
# ------------------------------------------------------------

import random                                    # dice for start positions, item spawns, moves, etc. (see SEEDS)
import json                                      # save the Q agent memory between runs
from functools import partial                    # fix an agent's own dice into its policy function
from pathlib import Path                         # find the memory file next to this script
import matplotlib.pyplot as plt                  # plotting
from matplotlib.colors import ListedColormap     # graphics
import agents                                    # import agent profiles

# World
## is grid
## has agents to call
## has objects of value

# Two ways to use this file:
##   python world.py                         watch one episode; the Q agent keeps learning between runs (SETTINGS below)
##   python run.py configs/<experiment>.toml  many runs of many episodes, no window, everything recorded (settings in the config)


# SETTINGS - for watching; experiments take theirs from configs/
# ------------------------------------------------------------

# Grid settings
Width = 10
Height = 10

Item = "£"          # item appearance
Item_value = 5
Num_items = 5
Num_steps = 100
Num_random_agent = 1
Num_greedy_agent = 1
Num_q_agent = 1
Seed = None         # None = new seeds each run

# Q agent settings
Q_epsilon = 0.1                                  # chance of a random move, to keep exploring
Q_memory_file = Path(__file__).parent / "q_memory.json"   # Q-tables are saved here after each run and loaded at the start of the next
Reset_q_memory = False                           # True = forget everything and start learning from scratch

Colours = ListedColormap(["white", "orange", "blue", "red", "green"])   # empty, item, agents
Frame_delay = 0.1


# AGENTS
# ------------------------------------------------------------

Agent_types = {                                  # look up agent by type name
    "random": {"policy": agents.random_agent,   "letter": "R", "colour": 2},                  # colour 2 = blue
    "greedy": {"policy": agents.greedy_agent,   "letter": "G", "colour": 3},                  # colour 3 = red
    "q":      {"policy": agents.QLearningAgent, "letter": "Q", "colour": 4, "learns": True},  # colour 4 = green; a class, so each Q agent is made with its own memory
}

def make_agents(counts, q_settings, q_memory=None):                 # counts e.g. {"random": 1, "q": 2}; q_settings e.g. {"epsilon": 0.1}; q_memory = saved Q-tables by agent name
    agent_list = []                                                 # make an empty list to add agent dictionaries to
    for type_name, count in counts.items():                         # .items() gives (key, value) pairs, e.g. "random", 1
        info = Agent_types[type_name]
        for n in range(count):                                      # make "count" agents of this type; n counts 0, 1, 2...
            name = f"{type_name}{n + 1}"                            # type + number, e.g. "random1"
            rng = random.Random()                                   # this agent's own dice, reseeded at the start of every episode (see SEEDS)
            if info.get("learns"):
                policy = info["policy"](**q_settings, epsilon_min=q_settings["epsilon"], rng=rng)   # min = start, so exploration stays the same every episode
                if q_memory and name in q_memory:
                    policy.from_dict(q_memory[name])                # carry on from where it left off
            else:
                policy = partial(info["policy"], rng=rng)           # the same function, but it always rolls this agent's dice
            agent_list.append({"name": name, "type": type_name, "policy": policy, "rng": rng})

    if not agent_list:
        raise SystemExit("No agents: set at least one agent count above 0.")   # do not run without agents
    return agent_list


# SEEDS - same seed, same episode
# ------------------------------------------------------------

# The world and every agent roll their own dice, each seeded from (seed, episode, owner).
# With one shared set of dice, an agent that rolls more would change every later roll for everyone else,
# so adding an agent would quietly move the items. With separate dice an agent's rolls are its own,
# and any episode can be played again from its seed and episode number (plus the Q agent's memory at the time).

def seed_dice(rng, seed, episode, owner):                           # owner = "world" or an agent's name
    rng.seed(f"{seed}:{episode}:{owner}")                           # text seeds turn into the same number every time, on any computer


# WORLD
# ------------------------------------------------------------

def new_grid(settings):
    return [["." for col in range(settings["width"])] for row in range(settings["height"])]   # build grid

def random_empty_cell(grid, rng, avoid=()):                         # avoid = cells to skip: search grid, find empty cell, add to list, skip. If nothing listed, don't skip
    empty = []                                                      # find empty cell
    for row in range(len(grid)):                                    # size comes from the grid itself, so any size works
        for col in range(len(grid[row])):
            if grid[row][col] == "." and (row, col) not in avoid:   # check rows, cols, ignore items
                empty.append((row, col))                            # note empty cells
    return rng.choice(empty)                                        # make random choice from empty cells


# AGENTS - With help of Claude once I wanted to move agents into a separate file, but pretty sure I get it
# ------------------------------------------------------------

def agent_positions(agent_list):
    """Every agent's current (row, col)."""
    return [a["position"] for a in agent_list]                      # go through each agent dictionary, take its position, collect positions as a list, e.g. [(4, 7), (0, 2)]


# ITEMS
# ------------------------------------------------------------

def spawn_item(grid, agent_list, rng):                                      # look at grid, look at agents, don't spawn item under agent
    row, col = random_empty_cell(grid, rng, avoid=agent_positions(agent_list))   # find current position of agents, check grid, keep list of empty cells, randomly select cell
    grid[row][col] = Item
    return row, col                                                         # where it went, for the record

def collect_item(grid, agent, item_value):                          # agent = the one agent collecting
    row, col = agent["position"]                                    # unpack this agent's (row, col)
    if grid[row][col] == Item:
        grid[row][col] = "."                                        # remove the collected item (the episode puts a new one somewhere else)
        return item_value                                           # add to score
    return 0                                                        # no points


# TEXT
# ------------------------------------------------------------

def show(grid, agent_list):
    letters = {a["position"]: Agent_types[a["type"]]["letter"] for a in agent_list}   # dict: position -> letter; tuples can be dict keys, lists can't
    for row in range(len(grid)):
        cells = []                                                  # characters for this row
        for col in range(len(grid[row])):
            if (row, col) in letters:                               # is an agent standing here?
                cells.append(letters[(row, col)])                   # draw its letter (if two share a cell, the later one shows)
            else:
                cells.append(grid[row][col])                        # otherwise whatever is in the cell
        print(" ".join(cells))                                      # one row, spaces between cells
    print()                                                         # blank line after the grid


# GRAPHICS -- Courtesy of Claude
# ------------------------------------------------------------

def render(ax, grid, agent_list, step):                             # ax = the window's drawing area, made in WATCH below
    agent_colours = {a["position"]: Agent_types[a["type"]]["colour"] for a in agent_list}   # dict: position -> colour number
    picture = []                                                    # Look at grid, assign colours accordingly
    for row in range(len(grid)):
        numbers = []
        for col in range(len(grid[row])):
            if (row, col) in agent_colours:                         # look for agent first, assign colour, colour will appear on top of white
                numbers.append(agent_colours[(row, col)])
            elif grid[row][col] == Item:
                numbers.append(1)                                   # 1 = item
            else:
                numbers.append(0)                                   # 0 = empty
        picture.append(numbers)

    ax.clear()                                   # wipe last frame
    ax.pcolormesh(picture, cmap=Colours, vmin=0, vmax=len(Colours.colors) - 1,   # vmax from the colour list, so adding a colour needs no change here
                  edgecolors="lightgrey", linewidth=1)
    ax.invert_yaxis()                            # row 0 at top
    ax.set_aspect("equal")                       # cells square
    ax.axis("off")                               # hide axis
    totals = [f"{t}: {sum(a['score'] for a in agent_list if a['type'] == t)}"   # total score per type
              for t in dict.fromkeys(a["type"] for a in agent_list)]           # each type once, in the order agents were made
    ax.set_title(f"Step {step}   " + "   ".join(totals))
    plt.pause(Frame_delay)                       # draw, wait


# MOVEMENT
# ------------------------------------------------------------

Moves = {                                                           # look up a change in (row, col) by direction name
    "up":    (-1, 0),                                               # up is -1 as row 0 is the top because this is a spreadsheet not a graph
    "down":  (1, 0),
    "left":  (0, -1),
    "right": (0, 1),
}

# Is agent on grid?
def on_grid(grid, row, col):
    return 0 <= row < len(grid) and 0 <= col < len(grid[0])

# Stay on grid
def legal_moves(grid, position):
    row, col = position
    legal = []
    for direction, (d_row, d_col) in Moves.items():
        if on_grid(grid, row + d_row, col + d_col):
            legal.append(direction)
    return legal

def move_options(grid, position):                                   # every legal move  e.g. {"up": (2, 5)} for agents to choose from
    return {direction: move(grid, position, direction) for direction in legal_moves(grid, position)}

def move(grid, position, direction):
    d_row, d_col = Moves[direction]     # Distance
    row, col = position                 # Position
    new_row = row + d_row               # New position
    new_col = col + d_col

    if not on_grid(grid, new_row, new_col):   # Stay on the grid
        return position
    return (new_row, new_col)


# EPISODE - one game, start to finish
# ------------------------------------------------------------

# settings:  {"width", "height", "num_items", "item_value", "num_steps"}
# recorder:  something with step(), event(), metrics() and episode() (see data.py), or None to record nothing
# ax:        a window to draw in, or None to run without one (much faster)

def run_episode(settings, agent_list, seed, episode, recorder=None, ax=None):
    world_rng = random.Random()                                     # the world's own dice
    seed_dice(world_rng, seed, episode, "world")
    for agent in agent_list:                                        # every agent starts the episode fresh, apart from what it has learned
        seed_dice(agent["rng"], seed, episode, agent["name"])
        agent["score"] = 0
        agent["items"] = 0

    grid = new_grid(settings)
    placed = []                                                     # cells taken by agents placed so far
    for agent in agent_list:
        agent["position"] = random_empty_cell(grid, world_rng, avoid=placed)   # random empty square, avoiding every agent already placed
        placed.append(agent["position"])

    for i in range(settings["num_items"]):                          # starting item count at num_items
        row, col = spawn_item(grid, agent_list, world_rng)
        if recorder:
            recorder.event(0, None, "spawn", None, row, col, 0)     # step 0 = before anyone moves

    if ax is not None:
        show(grid, agent_list)
        render(ax, grid, agent_list, 0)                             # step 0 = start

    for step in range(1, settings["num_steps"] + 1):
        for turn, agent in enumerate(agent_list):                   # each agent moves once per step, in order; turn = 0 for the first to move
            direction = agent["policy"](grid, agent["position"], move_options(grid, agent["position"]))   # look up agent, then call it
            agent["position"] = move(grid, agent["position"], direction)   # store the new position in agent's dictionary
            points = collect_item(grid, agent, settings["item_value"])

            if points > 0:
                agent["score"] += points                            # += adds to the existing value
                agent["items"] += 1
                row, col = spawn_item(grid, agent_list, world_rng)  # put a new item somewhere else
                if recorder:                                        # the world says who got it: with several agents chasing one item, turn order decides
                    recorder.event(step, turn, "collect", agent["name"], *agent["position"], points)
                    recorder.event(step, turn, "spawn", None, row, col, 0)
                if ax is not None:
                    print(f"Step {step}: {agent['name']} moved {direction} to {agent['position']}, "
                          f"collected {Item} (+{points}), score {agent['score']}")

            if hasattr(agent["policy"], "learn"):                   # learning agents: tell it the reward, and where it ended up
                diagnostics = agent["policy"].learn(points, grid, agent["position"], move_options(grid, agent["position"]))
                if recorder:
                    recorder.metrics(step, agent["name"], diagnostics)   # whatever the agent reports about itself, e.g. {"td_error": 0.3}

            if recorder:
                recorder.step(step, turn, agent["name"], *agent["position"], direction, points)

        if ax is not None:
            render(ax, grid, agent_list, step)                      # one frame per step, after moves

    for turn, agent in enumerate(agent_list):
        if hasattr(agent["policy"], "end_episode"):
            agent["policy"].end_episode()                           # e.g. explore a little less next time
        if recorder:
            recorder.episode(agent["name"], agent["type"], turn, agent["items"], agent["score"])
    return grid


# WATCH - python world.py
# ------------------------------------------------------------

if __name__ == "__main__":
    q_memory = {}                                                   # Q memory from earlier runs, by agent name e.g. {"q1": {...}}
    if Q_memory_file.exists() and not Reset_q_memory:
        q_memory = json.loads(Q_memory_file.read_text())

    settings = {"width": Width, "height": Height, "num_items": Num_items, "item_value": Item_value, "num_steps": Num_steps}
    counts = {"random": Num_random_agent, "greedy": Num_greedy_agent, "q": Num_q_agent}
    agent_list = make_agents(counts, {"epsilon": Q_epsilon}, q_memory)
    seed = Seed if Seed is not None else random.randrange(1_000_000)   # no seed set: pick one, and say which

    print ("Initalising world")
    print("Agents:", ", ".join(a["name"] for a in agent_list))      # list agent in the world
    print(f"Seed: {seed}")
    fig, ax = plt.subplots()                                        # make window
    grid = run_episode(settings, agent_list, seed, episode=1, ax=ax)

    print()
    print(f"After {Num_steps} steps:")
    show(grid, agent_list)
    for agent in agent_list:
        print(f"{agent['name']:>10}: {agent['items']} items, score {agent['score']}")

    # SAVE Q MEMORY - so the next run carries on learning
    for agent in agent_list:
        if hasattr(agent["policy"], "learn"):
            runs = q_memory.get(agent["name"], {}).get("runs", 0) + 1   # how many runs this agent has learned from
            q_memory[agent["name"]] = {"runs": runs, **agent["policy"].to_dict()}
            print(f"{agent['name']:>10}: has now learned from {runs} runs")
    if any(a["type"] == "q" for a in agent_list):
        Q_memory_file.write_text(json.dumps(q_memory))

    plt.show()                                                      # keep window open
