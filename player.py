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
    def __init__(self, database_path, color):
        """
        :param database: <flattened_board> : (score, num_of_occurrences)
        """
        self.database = DatabaseHandler.load_board_database(database_path)
        self.color = color

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

        return best_move