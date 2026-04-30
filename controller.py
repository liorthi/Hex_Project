from board import Board, RED, BLUE, EMPTY
from player import HumanPlayer
from PySide6.QtCore import QObject, Signal


class GameController(QObject):
    """
    Controller that manages the game logic and coordinates between Model (Board, Players) and View (UI)
    """
    
    # Signals to communicate with the View
    board_updated = Signal()
    game_over = Signal(str)  # Emits winner: "RED" or "BLUE"
    
    def __init__(self, board_size, red_player, blue_player, ui=None):
        super().__init__()
        
        # Model components
        self.board_size = board_size
        self.board = Board(board_size)
        self.red_player = red_player
        self.blue_player = blue_player
        
        # View component
        self.ui = ui
        
        # Game state
        self.current_turn = RED
        self.game_active = True
        self.move_history = []
        
    def set_ui(self, ui):
        """Set the UI component and connect signals"""
        self.ui = ui
        self.board_updated.connect(ui.update_display)
        self.game_over.connect(ui.handle_game_over)
        
    def get_current_player(self):
        """Get the current player object"""
        return self.red_player if self.current_turn == RED else self.blue_player
    
    def is_human_turn(self):
        """Check if current player is human"""
        current_player = self.get_current_player()
        return isinstance(current_player, HumanPlayer)
    
    def place_tile(self, row, col):
        """
        Place a tile at the given position for the current player
        
        Args:
            row: Row index
            col: Column index
            
        Returns:
            bool: True if move was successful, False otherwise
        """
        if not self.game_active:
            return False
            
        # Check if cell is empty
        if self.board.grid[row, col] != EMPTY:
            return False
        
        # Place the tile
        self.board.place(row, col, self.current_turn)
        
        # Record move
        self.move_history.append({
            'player': 'RED' if self.current_turn == RED else 'BLUE',
            'row': row,
            'col': col,
            'move_number': len(self.move_history) + 1
        })
        
        # Update UI
        self.board_updated.emit()
        
        # Check for win
        if self._check_win():
            return True
        
        # Switch turn
        self._switch_turn()
        
        # If next player is AI, make their move
        if not self.is_human_turn():
            self._make_ai_move()
        
        return True
    
    def _check_win(self):
        """
        Check if current player has won
        
        Returns:
            bool: True if current player won, False otherwise
        """
        if self.current_turn == RED and self.board.red_wins():
            self.game_active = False
            self.game_over.emit("RED")
            return True
        
        if self.current_turn == BLUE and self.board.blue_wins():
            self.game_active = False
            self.game_over.emit("BLUE")
            return True
        
        return False
    
    def _switch_turn(self):
        """Switch to the other player's turn"""
        self.current_turn = BLUE if self.current_turn == RED else RED
    
    def _make_ai_move(self):
        """Make a move for the AI player"""
        if not self.game_active:
            return
        
        current_player = self.get_current_player()
        row, col = current_player.get_move(self.board)
        self.place_tile(row, col)
    
    def start_game(self):
        """Start or restart the game"""
        self.board = Board(self.board_size)
        self.current_turn = RED
        self.game_active = True
        self.move_history = []
        
        # Update UI
        if self.ui:
            self.board_updated.emit()
        
        # If RED is AI, make first move
        if not self.is_human_turn():
            self._make_ai_move()
    
    def reset_game(self):
        """Reset the game (same as start_game)"""
        self.start_game()
    
    def get_board_state(self):
        """Get the current board state for UI rendering"""
        return self.board.grid.copy()
    
    def get_current_turn_color(self):
        """Get the current turn color"""
        return self.current_turn
    
    def is_game_active(self):
        """Check if game is still active"""
        return self.game_active
