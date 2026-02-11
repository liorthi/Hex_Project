from board import Board, RED, BLUE, EMPTY
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
            if self.current == RED and self.board.red_wins():
                if verbose:
                    print(self.board)
                    print("RED wins!")
                    self.winner = self.current
                return self._create_result()

            if self.current == BLUE and self.board.blue_wins():
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