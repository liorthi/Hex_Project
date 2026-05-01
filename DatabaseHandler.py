import json
from pathlib import Path
from board import RED, BLUE
import torch
import numpy as np

class DatabaseHandler:
    @staticmethod
    def save_games_to_json(results, filename="Hex_database_result.json"):
        """
        Save game results to a JSON file (optional, for analysis)

        Args:
            results: List of game results
            filename: Output filename

        Returns:
            str: Path to saved file
        """

        # Create output directory if it doesn't exist
        output_dir = Path("game_database")
        output_dir.mkdir(exist_ok=True)

        filepath = output_dir / filename

        # Add metadata
        data = {
            'metadata': {
                'total_games': len(results),
                'board_size': results[0]['board_size'] if results else None,
                'red_wins': sum(1 for r in results if r['winner'] == 'RED'),
                'blue_wins': sum(1 for r in results if r['winner'] == 'BLUE'),
            },
            'games': results
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Saved {len(results)} games to {filepath}")
        print(f"RED wins: {data['metadata']['red_wins']}")
        print(f"BLUE wins: {data['metadata']['blue_wins']}")

        return str(filepath)

    @staticmethod
    def save_board_database(board_database, filename="Hex_database_games.json"):
        """
        Save board database to JSON file in the exact format: {"[0,0,0,...]": [score, count]}

        Args:
            board_database: Dictionary of board states with scores
            filename: Output filename (default: auto-generated with timestamp)

        Returns:
            str: Path to saved file
        """

        # Create output directory if it doesn't exist
        output_dir = Path("game_database")
        output_dir.mkdir(exist_ok=True)

        filepath = output_dir / filename

        # Save directly as the board database without metadata wrapper
        with open(filepath, 'w') as f:
            json.dump(board_database, f)

        print(f"\nSaved {len(board_database)} unique board states to {filepath}")

        return str(filepath)

    @staticmethod
    def load_board_database(filename):
        """
        Load a board database from game_database/<filename>

        Args:
            filename: Name of the JSON file (e.g. "Hex_database_games.json")

        Returns:
            dict: {board_key: [avg_score, count]}
        """
        base_dir = Path("game_database")
        filepath = base_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Database file not found: {filepath}")

        with open(filepath, "r") as f:
            board_database = json.load(f)

        print(f"Loaded {len(board_database)} board states from {filepath}")
        return board_database

    @staticmethod
    def load_board_database_encoded(filename):
        """
        Load board database and convert to normalized tensor format.

        Returns:
            X: (N, board_size^2) tensor with values in [-1, 0, 1]
            Y: (N, 1) tensor with values in [-1, 1]
        """

        raw_data = DatabaseHandler.load_board_database(filename)

        X_list = []
        Y_list = []

        for board_str, (score, count) in raw_data.items():
            # --- Convert string to numpy array ---
            # "[0, 1, 2, ...]" → list[int]
            board_list = list(map(int, board_str.strip('[]').split(',')))

            size = int(len(board_list) ** 0.5)
            grid = np.array(board_list, dtype=np.int8).reshape(size, size)

            # --- Normalize to [-1, 1] ---
            norm_grid = np.zeros_like(grid, dtype=np.float32)
            norm_grid[grid == RED] = 1.0
            norm_grid[grid == BLUE] = -1.0
            # empty stays 0

            # --- Flatten ---
            X_list.append(norm_grid.flatten())

            # --- Target ---
            Y_list.append([float(score)])

        X = torch.tensor(X_list, dtype=torch.float32)
        Y = torch.tensor(Y_list, dtype=torch.float32)

        return X, Y

    @staticmethod
    def encode_single_board(board):
        """
        Convert board to normalized flat vector in [-1, 0, 1]

        Args:
            board: array([1., ...]) or list of length board_size^2

        Returns:
            list[float]: flattened normalized board
        """

        # Convert to numpy array
        grid = np.array(board, dtype=np.int8)

        # Normalize
        norm = np.zeros_like(grid, dtype=np.float32)
        norm[grid == RED] = 1.0
        norm[grid == BLUE] = -1.0
        # empty stays 0

        return norm.tolist()

    @staticmethod
    def predict_score(model, board, device):
        """
        Predict score for a given board.

        Returns:
            float
        """
        board_vector = DatabaseHandler.encode_single_board(board)

        x_tensor = torch.tensor(board_vector, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            prediction = model(x_tensor)

        return prediction.item()

    @staticmethod
    def load_network(model_path, device, net):
        """
        Loads a saved Net model from a .pth file.

        Args:
            model_path: Path to saved model
            device: Device to use
            net: Net model

        Returns:
            The model
        """
        print(f"Loading model from {model_path}...")

        # 1. Instantiate a fresh model
        model = net().to(device)

        # 2. Load the saved weights into the model
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

        # 3. Set the model to evaluation mode
        model.eval()

        return model