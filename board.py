import numpy as np

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
