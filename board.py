import numpy as np
from collections import deque

EMPTY = 0
RED   = 1
BLUE  = 2

class Board:
    def __init__(self, size: int):
        self.size = size
        self.grid = np.zeros((size, size), dtype=np.int8)

    def in_bounds(self, r, c):
        return 0 <= r < self.size and 0 <= c < self.size

    def neighbors(self, r, c):
        """
        this is a generator function for getting the neighbours coordinates of a cell
        :param r: row
        :param c: collum
        :return:
        """
        directions = [
            (-1, 0), (-1, 1),
            (0, -1), (0, 1),
            (1, -1), (1, 0)
        ]
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if self.in_bounds(nr, nc):
                yield nr, nc

    def place(self, r, c, player):
        """
        this is a function that places a cell on a board
        :param r: row
        :param c: collum
        :param player: turn of the game
        :return:
        """
        if self.grid[r, c] != EMPTY:
            raise ValueError("Cell already occupied")
        self.grid[r, c] = player

    def is_full(self):
        return not np.any(self.grid == EMPTY)

    def empty_cells(self):
        return list(zip(*np.where(self.grid == EMPTY)))

    def __str__(self):
        symbols = {EMPTY: ".", RED: "R", BLUE: "B"}
        lines = []
        for r in range(self.size):
            indent = " " * r
            row = " ".join(symbols[self.grid[r, c]] for c in range(self.size))
            lines.append(indent + row)
        return "\n".join(lines)

    def red_wins(self):
        n = self.size
        visited = set()
        q = deque()

        # Start from top row
        for c in range(n):
            if self.grid[0, c] == RED:
                q.append((0, c))
                visited.add((0, c))

        while q:
            r, c = q.popleft()
            if r == n - 1:
                return True

            for nr, nc in self.neighbors(r, c):
                if (nr, nc) not in visited and self.grid[nr, nc] == RED:
                    visited.add((nr, nc))
                    q.append((nr, nc))

        return False

    def blue_wins(self):
        n = self.size
        visited = set()
        q = deque()

        # Start from left column
        for r in range(n):
            if self.grid[r, 0] == BLUE:
                q.append((r, 0))
                visited.add((r, 0))

        while q:
            r, c = q.popleft()
            if c == n - 1:
                return True

            for nr, nc in self.neighbors(r, c):
                if (nr, nc) not in visited and self.grid[nr, nc] == BLUE:
                    visited.add((nr, nc))
                    q.append((nr, nc))

        return False

    def blue_distances(self):
        """
        Returns:
            (min_total, left_map, right_map)

        min_total: minimal number of EMPTY tiles needed to connect LEFT->RIGHT
        left_map: distance from LEFT edge to each cell
        right_map: distance from RIGHT edge to each cell
        """
        left_map = self._bfs_edge(0, BLUE)
        right_map = self._bfs_edge(self.size - 1, BLUE)

        min_total = float('inf')
        for pos, ld in left_map.items():
            if pos in right_map:
                rd = right_map[pos]
                min_total = min(min_total, ld + rd)

        if min_total == float('inf'):
            min_total = None

        return min_total, left_map, right_map

    def red_distances(self):
        """
        Returns:
            (min_total, top_map, bottom_map)

        min_total: minimal number of EMPTY tiles needed to connect TOP->BOTTOM
        top_map: distance from TOP edge to each cell
        bottom_map: distance from BOTTOM edge to each cell
        """
        top_map = self._bfs_edge(0, RED, vertical=True)
        bottom_map = self._bfs_edge(self.size - 1, RED, vertical=True)

        min_total = float('inf')
        for pos, td in top_map.items():
            if pos in bottom_map:
                bd = bottom_map[pos]
                min_total = min(min_total, td + bd)

        if min_total == float('inf'):
            min_total = None

        return min_total, top_map, bottom_map

    def _bfs_edge(self, edge_index, color, vertical=False):
        """
        BFS from one edge to all reachable cells.
        If vertical is False -> left/right edges (Blue)
        If vertical is True  -> top/bottom edges (Red)

        Returns the minimum number of empty tiles that need to be filled
        to connect from a given edge to every reachable cell.

        Returns:
            dict: {(r, c): distance}
        """
        visited = set()
        q = deque()

        # Initialize from the given edge
        for i in range(self.size):
            r, c = (i, edge_index) if not vertical else (edge_index, i)

            if self.grid[r, c] in (color, EMPTY):
                cost = 0 if self.grid[r, c] == color else 1
                q.append((r, c, cost))
                visited.add((r, c))

        dist_map = {}

        while q:
            r, c, cost = q.popleft()

            # Store minimal distance for this cell
            if (r, c) not in dist_map or cost < dist_map[(r, c)]:
                dist_map[(r, c)] = cost

            for nr, nc in self.neighbors(r, c):
                if (nr, nc) not in visited:
                    cell = self.grid[nr, nc]

                    if cell == color:
                        q.appendleft((nr, nc, cost))
                        visited.add((nr, nc))

                    elif cell == EMPTY:
                        q.append((nr, nc, cost + 1))
                        visited.add((nr, nc))

        return dist_map