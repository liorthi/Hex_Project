import random
from board import Board, EMPTY, BLUE, RED
from DatabaseHandler import DatabaseHandler


class Player:
    def get_move(self, board):
        raise NotImplementedError

class HumanPlayer(Player):
    def get_move(self, board):
        while True:
            try:
                r, c = map(int, input("Enter move (row col): ").split())
                if board.grid[r, c] == EMPTY:
                    return r, c
                print("Cell occupied.")
            except Exception:
                print("Invalid input.")


class RandomAI(Player):
    def get_move(self, board):
        return random.choice(board.empty_cells())


class GreedyAI(Player):
    def __init__(self, database_path, color, gama=0.9):
        """
        :param database: <flattened_board> : (score, num_of_occurrences)
        """
        self.database = DatabaseHandler.load_board_database(database_path)
        self.color = color
        self.gama = gama

    def get_move(self, board):
        best_score = None
        best_move = None

        for r, c in board.empty_cells():
            # Create board copy
            temp = Board(board.size)
            temp.grid = board.grid.copy()

            # Apply move
            temp.place(r, c, self.color)

            # Lookup board score
            key = str(temp.grid.flatten().tolist())

            if key in self.database:
                score, _ = self.database[key]

            #if unknown
            else:
                score = 0.5

            # Greedy choice
            if best_move is None:
                best_score = score
                best_move = (r, c)
            else:
                if score > best_score:
                    best_score = score
                    best_move = (r, c)

        # return the best move 90% of the time
        if random.random() < self.gama:
            return best_move
        return random.choice(board.empty_cells())


class HeuristicAI(Player):
    def __init__(self, database_path, color):
        self._greedy = GreedyAI(database_path, color)
        self.color = color
        self.opponent = RED if color == BLUE else BLUE

    def get_move(self, board):
        # RULES OVERRIDE
        override_move = self._apply_rules(board)
        if override_move is not None:
            return override_move

        # NO RULES -> GREEDY MOVE
        move = self._greedy.get_move(board)

        # IF GREEDY MOVE IS BAD -> CORRECT IT
        if self._is_bad_move(board, move):
            correct_move = self._correct_move(board)

            if correct_move:
                return correct_move

        return move

    def _apply_rules(self, board):
        """
        The function applies simple rules on the player:
        1. check distance from completing a chain, if distance 1, win
        2. check distance of opponent from completing a chain, if distance 1, block
        3. if opponent is at distance 2 from the end of the board, block it's shortest rout to
        let us block the path  completely later.
        """
        if self.color == BLUE:
            my_min_distance, _, _ = board.blue_distances()
            opponent_min_distance, _, _ = board.red_distances()
        else:
            my_min_distance, _, _ = board.red_distances()
            opponent_min_distance, _, _ = board.blue_distances()

        # Rule 1: win immediately
        if my_min_distance == 1:
            return self._find_winning_move(board, self.color)

        # Rule 2: block opponent win
        if opponent_min_distance == 1:
            return self._find_winning_move(board, self.opponent)

        # Rule 3: pre-block if opponent distance is 2
        if opponent_min_distance == 2:
            return self._preblock(board, opponent_min_distance)

        return None

    def _find_winning_move(self, board, color):
        """
        Return a move that immediately wins the game, or None
        """
        for r, c in board.empty_cells():
            # simulate move
            temp = Board(board.size)
            temp.grid = board.grid.copy()
            temp.place(r, c, color)

            # check win
            if color == BLUE and temp.blue_wins():
                return r, c

            if color == RED and temp.red_wins():
                return r, c

        return None

    def _preblock(self, board, opponent_min_distance):
        """
        The function adds a "preblock", the option to block the opponent's move before he wins.
        """
        if self.opponent == BLUE:
            _, left_map, right_map = board.blue_distances()
            edge_col = board.size - 1
        else:
            _, left_map, right_map = board.red_distances()
            edge_row = board.size - 1

        empty = set(board.empty_cells())

        # collect shortest-path cells that are on the edge
        for pos in left_map:
            if pos in right_map:
                if left_map[pos] + right_map[pos] == opponent_min_distance:
                    r, c = pos

                    if self.opponent == BLUE:
                        if c == edge_col and (r, c) in empty:
                            return r, c
                    else:
                        if r == edge_row and (r, c) in empty:
                            return r, c

        return None

    def _is_bad_move(self, board, move):
        """
        The function checks if the move is bad or not.
        """
        r, c = move

        if board.grid[r, c] != EMPTY:
            return True

        # distance before move
        if self.color == BLUE:
            dist_before, _, _ = board.blue_distances()
        else:
            dist_before, _, _ = board.red_distances()

        # distance after move
        temp = Board(board.size)
        temp.grid = board.grid.copy()
        temp.place(r, c, self.color)

        if self.color == BLUE:
            dist_after, _, _ = temp.blue_distances()
        else:
            dist_after, _, _ = temp.red_distances()

        # pocket (meaningless move)
        if dist_after is None:
            return True

        # if move does not improve or maintain distance
        if dist_after >= dist_before:
            return True

        return False



    def _correct_move(self, board):
        best_move = None
        best_dist = float('inf')

        for r, c in board.empty_cells():
            temp = Board(board.size)
            temp.grid = board.grid.copy()
            temp.place(r, c, self.color)

            if self.color == BLUE:
                dist, _, _ = temp.blue_distances()
            else:
                dist, _, _ = temp.red_distances()

            if dist is not None and dist < best_dist:
                best_dist = dist
                best_move = (r, c)

        return best_move