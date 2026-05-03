"""
environment.py
--------------
Original reward structure:
  - Step penalty      : -0.01  every step
  - Collision penalty : -0.5   hitting wall or obstacle
  - Shaping reward    : +0.1 * (old_dist - new_dist)
  - Goal reward       : +10.0

Goal logic:
  - dist == 0  → agent stepped EXACTLY onto goal cell
  - Triggers goal reward +10.0 and end the episode
"""

import numpy as np
import random


class GridEnvironment:

    SENSOR_DIRS = [
        (-1,  0), (-1,  1), ( 0,  1), ( 1,  1),
        ( 1,  0), ( 1, -1), ( 0, -1), (-1, -1),
    ]

    ACTION_DELTAS = {
        0: (-1,  0),  # N
        1: ( 0,  1),  # E
        2: ( 1,  0),  # S
        3: ( 0, -1),  # W
        4: ( 0,  0),  # Stay
    }

    def __init__(
        self,
        grid_size=(20, 20),
        n_sensors=8,
        n_obstacles=15,
        dynamic=False,
        dynamic_interval=5,
        max_steps=200,
        seed=None,
    ):
        self.rows, self.cols  = grid_size
        self.n_sensors        = n_sensors
        self.n_obstacles      = n_obstacles
        self.dynamic          = dynamic
        self.dynamic_interval = dynamic_interval
        self.max_steps        = max_steps
        self.rng              = random.Random(seed)
        self.np_rng           = np.random.default_rng(seed)

        self.grid               = None
        self.agent_pos          = None
        self.goal_pos           = None
        self.obstacle_positions = []
        self.step_count         = 0
        self.done               = False

        self.reset()

    def reset(self):
        self.grid       = np.zeros((self.rows, self.cols), dtype=int)
        self.step_count = 0
        self.done       = False

        self.agent_pos = self._random_pos(
            row_range=(self.rows // 2, self.rows - 1),
            col_range=(0, self.cols // 2)
        )
        self.goal_pos = self._random_pos(
            row_range=(0, self.rows // 2),
            col_range=(self.cols // 2, self.cols - 1)
        )

        self.obstacle_positions = []
        attempts = 0
        while len(self.obstacle_positions) < self.n_obstacles and attempts < 1000:
            pos = self._random_pos()
            if (pos != self.agent_pos
                    and pos != self.goal_pos
                    and pos not in self.obstacle_positions):
                self.obstacle_positions.append(pos)

                # Uncomment for robustness, but leaving it out for time
                # if not self.contains_path(self.agent_pos, self.goal_pos):
                #     self.obstacle_positions.pop()
            attempts += 1

        self._sync_grid()
        return self._get_sensor_readings()

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode finished. Call reset().")

        self.step_count += 1
        dr, dc  = self.ACTION_DELTAS[action]
        new_pos = (self.agent_pos[0] + dr, self.agent_pos[1] + dc)

        # Default step penalty
        reward = -0.01
        collision = False

        if self._out_of_bounds(new_pos) or new_pos in self.obstacle_positions:
            # Collision — agent stays, gets penalty
            reward = -0.5
            collision = True
        else:
            # Valid move — apply shaping reward
            old_dist = self._manhattan(self.agent_pos, self.goal_pos)
            new_dist = self._manhattan(new_pos,        self.goal_pos)
            reward  += 0.1 * (old_dist - new_dist)
            self.agent_pos = new_pos

        # --------------------------------------------------
        # Goal check:
        #   dist == 0 → exactly on goal cell
        #   Counts as goal reached → reward +10.0
        # --------------------------------------------------
        dist_to_goal = self._manhattan(self.agent_pos, self.goal_pos)
        goal_reached = False

        if dist_to_goal == 0:
            reward       = 10.0
            self.done    = True
            goal_reached = True

        # Max steps reached
        if self.step_count >= self.max_steps:
            self.done = True

        # Move obstacles if dynamic mode
        if self.dynamic and self.step_count % self.dynamic_interval == 0:
            self._move_obstacles()
            
            # Uncomment the following for robustness, but otherwise leaving it out for time
            # while not self.contains_path(self.agent_pos, self.goal_pos):
            #     self._move_obstacles()

            self.move_goal()

        self._sync_grid()
        sensors = self._get_sensor_readings()

        info = {
            "step"         : self.step_count,
            "agent_pos"    : self.agent_pos,
            "goal_pos"     : self.goal_pos,
            "dist_to_goal" : dist_to_goal,
            "goal_reached" : goal_reached,
            "collision"    : collision,
        }
        return sensors, reward, self.done, info

    def get_sensor_readings(self):
        return self._get_sensor_readings()

    def render(self):
        symbols = {0: ".", 1: "X", 2: "G", 3: "R"}
        print(f"Step {self.step_count} | "
              f"Agent: {self.agent_pos} | Goal: {self.goal_pos}")
        for r in range(self.rows):
            print(" ".join(symbols.get(self.grid[r, c], "?")
                           for c in range(self.cols)))
        print()

    def _sync_grid(self):
        self.grid[:] = 0
        self.grid[self.goal_pos] = 2
        for obs in self.obstacle_positions:
            if 0 <= obs[0] < self.rows and 0 <= obs[1] < self.cols:
                self.grid[obs] = 1
        self.grid[self.agent_pos] = 3

    def _get_sensor_readings(self):
        readings = []
        max_dist = max(self.rows, self.cols)
        dirs     = self.SENSOR_DIRS[:self.n_sensors]

        for dr, dc in dirs:
            dist = 0
            r, c = self.agent_pos
            while True:
                dist += 1
                r    += dr
                c    += dc
                if (self._out_of_bounds((r, c))
                        or (r, c) in self.obstacle_positions):
                    break
                if dist >= max_dist:
                    break
            readings.append(dist / max_dist)

        # Goal direction (normalised)
        goal_dr = (self.goal_pos[0] - self.agent_pos[0]) / self.rows
        goal_dc = (self.goal_pos[1] - self.agent_pos[1]) / self.cols
        readings.append(goal_dr)
        readings.append(goal_dc)

        return np.array(readings, dtype=float)

    def _move_obstacles(self):
        new_positions = []
        for pos in self.obstacle_positions:
            direction = self.rng.choice(list(self.ACTION_DELTAS.values())[:4])
            new_pos   = (pos[0] + direction[0], pos[1] + direction[1])
            if (not self._out_of_bounds(new_pos)
                    and new_pos != self.agent_pos
                    and new_pos != self.goal_pos
                    and new_pos not in new_positions):
                new_positions.append(new_pos)
            else:
                new_positions.append(pos)
        self.obstacle_positions = new_positions

    def _random_pos(self, row_range=None, col_range=None):
        if row_range is None:
            row_range = (0, self.rows - 1)
        if col_range is None:
            col_range = (0, self.cols - 1)
        return (self.rng.randint(*row_range),
                self.rng.randint(*col_range))

    def _out_of_bounds(self, pos):
        r, c = pos
        return r < 0 or r >= self.rows or c < 0 or c >= self.cols

    @staticmethod
    def _manhattan(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])



    def contains_path(self, start, goal) -> bool:
        queue = [start]
        visited = {start}

        while queue:
            cur = queue.pop(0)
            if cur == goal:
                return True
            
            for row, col in list(self.ACTION_DELTAS.values())[:4]:
                next_pos = (cur[0] + row, cur[1] + col)
                if (not self._out_of_bounds(next_pos)
                    and next_pos not in self.obstacle_positions
                    and next_pos not in visited):
                    visited.add(next_pos)
                    queue.append(next_pos)

        return False
    

    def move_goal(self):
        goal = self.goal_pos
        moves = list(self.ACTION_DELTAS.values())[:4]
        random.shuffle(moves)
        moves.append((0, 0))
        
        for i in range(len(moves)): 
            direction = moves[i]
            new_pos   = (goal[0] + direction[0], goal[1] + direction[1])
            if (not self._out_of_bounds(new_pos)
                    and new_pos not in self.obstacle_positions
                    and new_pos != self.agent_pos):
                self.goal_pos = new_pos
                break

