from board import Board, RED, BLUE, EMPTY
from collections import deque
import player

class Game:
    def __init__(self, size, red_player, blue_player):
        self.board = Board(size)
        self.players = {
            RED: red_player,
            BLUE: blue_player
        }
        self.size = size
        self.current = RED
        self.winner = None

        self.move_history = []
        self.board_states = []  # Track all board states

    def switch_player(self):
        self.current = BLUE if self.current == RED else RED

    def play(self, verbose=False):
        """
        Play a complete game and return the result

        Returns:
            dict: Game result with winner, moves, and board states
        """
        while True:
            # Save current board state before making a move
            self.board_states.append(self.board.grid.copy())

            if verbose:
                print(f"\nMove {len(self.move_history) + 1}")
                print(self.board)

            # Get current player's move
            player = self.players[self.current]
            r, c = player.get_move(self.board)

            # Record move
            self.move_history.append({
                'player': 'RED' if self.current == RED else 'BLUE',
                'row': int(r),
                'col': int(c),
                'move_number': len(self.move_history) + 1
            })

            # Make move
            self.board.place(r, c, self.current)

            # Check for win
            if self.current == RED and self.red_wins():
                if verbose:
                    print(self.board)
                    print("RED wins!")
                    self.winner = self.current
                return self._create_result()

            if self.current == BLUE and self.blue_wins():
                if verbose:
                    print(self.board)
                    print("BLUE wins!")
                    self.winner = self.current
                return self._create_result()

            # Check for tie (board full)
            if self.board.is_full():
                if verbose:
                    print(self.board)
                    print("Tie!")

                return self._create_result()

            # Switch player
            self.switch_player()

    def reset_game(self):
        self.board = Board(self.size)
        self.current = RED
        self.winner = None

        self.move_history = []
        self.board_states = []  # Track all board states


    def _create_result(self):
        """Create a game result dictionary"""
        winner = 'RED' if self.current == RED else \
            ('BLUE' if self.current == BLUE else 'Tie')

        return {
            'winner': winner,
            'total_moves': len(self.move_history),
            'moves': self.move_history,
            'board_states': self.board_states,
            'final_board': self.board.grid.tolist()
        }

    def red_wins(self):
        n = self.board.size
        visited = set()
        q = deque()

        # Start from top row
        for c in range(n):
            if self.board.grid[0, c] == RED:
                q.append((0, c))
                visited.add((0, c))

        while q:
            r, c = q.popleft()
            if r == n - 1:
                return True

            for nr, nc in self.board.neighbors(r, c):
                if (nr, nc) not in visited and self.board.grid[nr, nc] == RED:
                    visited.add((nr, nc))
                    q.append((nr, nc))

        return False

    def blue_wins(self):
        n = self.board.size
        visited = set()
        q = deque()

        # Start from left column
        for r in range(n):
            if self.board.grid[r, 0] == BLUE:
                q.append((r, 0))
                visited.add((r, 0))

        while q:
            r, c = q.popleft()
            if c == n - 1:
                return True

            for nr, nc in self.board.neighbors(r, c):
                if (nr, nc) not in visited and self.board.grid[nr, nc] == BLUE:
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

    # =========================
    # RED DISTANCE FUNCTIONS
    # =========================
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

    # =========================
    # INTERNAL BFS HELPER
    # =========================
    def _bfs_edge(self, edge_index, color, vertical=False):
        """
        BFS from one edge to all reachable cells.
        If vertical is False -> left/right edges (Blue)
        If vertical is True  -> top/bottom edges (Red)

        Returns:
            dict: {(r, c): distance}
        """
        visited = set()
        q = deque()

        # Initialize from the given edge
        for i in range(self.size):
            r, c = (i, edge_index) if not vertical else (edge_index, i)

            if self.board.grid[r, c] in (color, EMPTY):
                cost = 0 if self.board.grid[r, c] == color else 1
                q.append((r, c, cost))
                visited.add((r, c))

        dist_map = {}

        while q:
            r, c, cost = q.popleft()

            # Store minimal distance for this cell
            if (r, c) not in dist_map or cost < dist_map[(r, c)]:
                dist_map[(r, c)] = cost

            for nr, nc in self.board.neighbors(r, c):
                if (nr, nc) not in visited:
                    cell = self.board.grid[nr, nc]

                    if cell == color:
                        q.appendleft((nr, nc, cost))
                        visited.add((nr, nc))

                    elif cell == EMPTY:
                        q.append((nr, nc, cost + 1))
                        visited.add((nr, nc))

        return dist_map