# AGENTS

# choose(grid, position, options) -> direction
## grid:      world (list of rows)
## position:  agent's (row, col)
## options:   dictionary of legal moves, direction e.g. {"up": (2, 5), "left": (3, 4)}
## rng:       agent's dice (random.Random)
# ------------------------------------------------------------------

import random



# RANDOM AGENT
# ------------------------------------------------------------

def random_agent(grid, position, options, rng=random):              # rng=random: with no dice given, use Python's shared ones
    return rng.choice(list(options))                                # list of movement options, chosen randomly




# GREEDY AGENT
# ------------------------------------------------------------

def greedy_agent(grid, position, options, item="£", rng=random):
    items = []                                                      # see every item's (row, col)
    for row in range(len(grid)):                                    # determine size of grid
        for col in range(len(grid[row])):                           
            if grid[row][col] == item:                              # see if there is an item in a cell
                items.append((row, col))                            # if there is an item, note it

    if not items:                                                   # can't find anything? default to random movement
        return random_agent(grid, position, options, rng)           # use random agent as default behaviour

    def distance(a, b):                                             # distance is travelled horizontal and vertical, not diagonal
        return abs(a[0] - b[0]) + abs(a[1] - b[1])                  # go from a to b by calculating number of rows and columns apart, remove - to get direction

    def nearest_item_distance(cell):                                # sees items
        return min(distance(cell, it) for it in items)              # consider distance to each item, pursue smallest


    best = min(nearest_item_distance(new_pos) for new_pos in options.values())   # consider available positions, look at each one, 'how far would nearest item be if here?', keep nearest
    best_moves = [d for d, new_pos in options.items()               # consider which moves get to the best distance
                  if nearest_item_distance(new_pos) == best]        # keep every direction that reaches that best distance (could be more than one)
    return rng.choice(best_moves)                                   # pick randomly from equally good ones




# Q-LEARNING AGENT
# ------------------------------------------------------------

# state:   which direction is the nearest item? e.g. (1, -1) = down and left
#          9 possible states (all composed of -1, 0 or 1)
# action:  a direction, e.g. "up"
# Q-table: dictionary (state, action) -> expected points for actions
# reward:  points from the world = Item_value when an item is collected, otherwise 0 points

# after each move, world calls learn(), and the agent adjusts its guess for the move it made:
# where new guess = old guess + alpha * (reward + gamma * best guess from where I landed - old guess)

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.998, item="£", rng=random):
        self.alpha = alpha                                          # learning rate: how much new experiences moves guesses
        self.gamma = gamma                                          # discount rate: how much future points count compared with points now
        self.epsilon = epsilon                                      # exploration: % of the time to choose a random move
        self.epsilon_min = epsilon_min                              # minimum exploration rate for training
        self.epsilon_decay = epsilon_decay                          # epsilon multiplied by this after episodes
        self.item = item
        self.rng = rng                                              # dice, for exploring with some degree of randomness and for tie breaking
        self.q = {}                                                 # Q-table, empty at start; unseen (state, action) pairs count as 0
        self.training = True                                        # False = no exploring or learning, just use the Q-table
        self.last_state = None
        self.last_action = None
        self.explored = False                                       # was the last move random?

    def state(self, grid, position):                                # direction of the nearest item, as (up/down, left/right)
        items = [(row, col) for row in range(len(grid)) for col in range(len(grid[row])) if grid[row][col] == self.item]
        nearest = min(items, key=lambda it: (abs(it[0] - position[0]) + abs(it[1] - position[1]), it))   # ties: lowest (row, col) wins, so the state is always the same for the same grid
        dr, dc = nearest[0] - position[0], nearest[1] - position[1] # rows away, cols away
        return ((dr > 0) - (dr < 0), (dc > 0) - (dc < 0))           # keep only the sign: -1 (up/left), 0 (same row/col) or 1 (down/right)

    def best_moves(self, state, options):                           # every legal direction with the highest Q value
        best = max(self.q.get((state, d), 0.0) for d in options)
        return [d for d in options if self.q.get((state, d), 0.0) == best]

    def __call__(self, grid, position, options):                   
        state = self.state(grid, position)
        self.explored = self.training and self.rng.random() < self.epsilon
        if self.explored:
            action = self.rng.choice(list(options))                 # explore
        else:
            action = self.rng.choice(self.best_moves(state, options))   # exploit; random choice among equally good options
        self.last_state, self.last_action = state, action
        return action

    def learn(self, reward, grid, position, options):               # called by the world after the move: "what happened and where am I now?"
        if not self.training:
            return {}
        next_state = self.state(grid, position)
        best_next = max(self.q.get((next_state, d), 0.0) for d in options)   # value of the best move from the new position
        old = self.q.get((self.last_state, self.last_action), 0.0)
        target = reward + self.gamma * best_next                    # what this move was worth
        self.q[(self.last_state, self.last_action)] = old + self.alpha * (target - old)
        return {"td_error": target - old,                           # how wrong the old guess was; shrinks as the agent learns
                "explored": float(self.explored)}                   # 1 = that was a random move

    def end_episode(self):                                          # explore a little less each episode (to force more reliance on learning)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def to_dict(self):                                              # save memory between runs
        return {"epsilon": self.epsilon,
                "q": [[state[0], state[1], action, value] for (state, action), value in self.q.items()]}

    def from_dict(self, data):                                      # put saved memory back
        self.epsilon = data["epsilon"]
        self.q = {((row, col), action): value for row, col, action, value in data["q"]}
