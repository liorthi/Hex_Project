from board import Board, RED, BLUE
from player import RandomAI
from game import Game

class Tournament:
    """Run a Hex game without GUI for fast simulation"""

    def __init__(self, num_games, board_size=7, red_player_class=RandomAI,
                       blue_player_class=RandomAI, gamma=0.9):
        """
        The Tournament constructor
        :param num_games:
        :param board_size:
        :param red_player_class:
        :param blue_player_class:
        :param gamma: used for score calculation
        """
        self.num_games = num_games
        self.board_size = board_size
        self.gamma = gamma

        self.players = {
            RED: red_player_class,
            BLUE: blue_player_class
        }

        self.board_database = {}

    @staticmethod
    def board_to_key(board_array):
        """Convert a board numpy array to a JSON string key like [0,0,0,...]"""
        # Flatten the 2D board to 1D and convert to list
        return str(board_array.flatten().tolist())

    @staticmethod
    def calculate_board_scores(board_states, winner, gamma=0.9):
        """
        Calculate scores for all board states in a game

        Args:
            board_states: List of board state numpy arrays
            winner: 'RED', 'BLUE', or 'TIE'
            gamma: Discount factor (default 0.9)

        Returns:
            dict: {board_key: score} for each board state
        """
        # Determine outcome
        if winner == 'TIE':
            outcome = 0.1
        elif winner == 'RED':
            outcome = 0.0
        else:  # BLUE wins
            outcome = 1.0

        N = len(board_states)
        board_scores = {}

        for i, board_state in enumerate(board_states):
            # Calculate score: outcome * gamma^(N-i-1)
            score = outcome * (gamma ** (N - i - 1))
            board_key = Tournament.board_to_key(board_state)
            board_scores[board_key] = score

        return board_scores


    def run_multiple_games(self, verbose=False):
        """
        Run multiple games and collect results

        Args:
            verbose: Print game progress

        Returns:
            tuple: (results list, board_database dict, winners dict)
        """
        results = []
        winners = {
            'RED': 0,
            'BLUE': 0,
            'Tie': 0
        }

        game = Game(self.board_size, self.players[RED], self.players[BLUE])

        for i in range(self.num_games):
            if verbose or (i + 1) % 1000 == 0:
                print(f"Playing game {i + 1}/{self.num_games}...")

            # Play game
            game.reset_game()
            result = game.play(verbose=verbose and i == 1)  # print only the first game
            result['game_number'] = i + 1
            winners[result['winner']] += 1

            # Calculate scores for all board states in this game
            board_scores = Tournament.calculate_board_scores(
                result['board_states'],
                result['winner'],
                self.gamma
            )

            # Update board database with dynamic averaging
            self.update_board_database(board_scores)

            # Remove board_states from result to save memory (they're in the database now)
            del result['board_states']
            results.append(result)

        return results, self.board_database, winners

    def update_board_database(self, board_scores):
        for board_key, score in board_scores.items():
            if board_key in self.board_database:

                # Dynamic average: new_avg = (old_avg * count + new_score) / (count + 1)
                old_avg, count = self.board_database[board_key]
                new_count = count + 1
                new_avg = (old_avg * count + score) / new_count
                self.board_database[board_key] = [new_avg, new_count]

            # First time seeing this board
            else:
                self.board_database[board_key] = [score, 1]