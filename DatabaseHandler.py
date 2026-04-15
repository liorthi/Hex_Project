import json
from pathlib import Path
import torch

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
                'ties': sum(1 for r in results if r['winner'] == 'TIE'),
            },
            'games': results
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Saved {len(results)} games to {filepath}")
        print(f"RED wins: {data['metadata']['red_wins']}")
        print(f"BLUE wins: {data['metadata']['blue_wins']}")
        print(f"Ties: {data['metadata']['ties']}")

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
        Load a board database from game_database/<filename> and one-hot encodes it

        Args:
            filename: Name of the JSON file (e.g. "Hex_database_games.json")

        Returns:
            (tensor: X_list, tensor: Y_list)
        """
        raw_data = DatabaseHandler.load_board_database(filename)

        X_list = []
        Y_list = []

        for board_str, values in raw_data.items():
            # Clean: "[001...]" -> "001..."
            clean_board = board_str[1:-1]
            clean_board = clean_board.replace(",", "").replace(" ", "")

            # Encode: 0 -> [1,0,0], 1 -> [0,1,0], 2 -> [0,0,1]
            board_vector = DatabaseHandler.encode_single_board(clean_board)

            X_list.append(board_vector)
            Y_list.append([values[0]])

        return torch.tensor(X_list), torch.tensor(Y_list)

    @staticmethod
    def encode_single_board(board_str):
        """
        One-hot encode a single board string.

        Args:
            board_str: String representation of board (e.g., "[0,0,1,2,...]")

        Returns:
            list: One-hot encoded board vector
        """
        # Encode: 0 -> [1,0,0], 1 -> [0,1,0], 2 -> [0,0,1]
        board_vector = []
        for char in board_str:
            val = int(char)
            one_hot = [0.0, 0.0, 0.0]
            one_hot[val] = 1.0
            board_vector.extend(one_hot)

        return board_vector

    @staticmethod
    def predict_score(model, board_str, device):
        """
        Takes a trained model and a board string, and outputs the predicted score.

        Args:
            model: Trained neural network model
            board_str: String representation of board
            device: Device to run prediction on

        Returns:
            float: Predicted score
        """
        # 1. Encode the board using our helper function
        board_vector = DatabaseHandler.encode_single_board(board_str)

        # 2. Convert to tensor and add a "batch" dimension (shape becomes [1, 147])
        x_tensor = torch.tensor([board_vector]).to(device)

        # 3. Make the prediction without calculating gradients
        with torch.no_grad():
            prediction = model(x_tensor)

        # 4. Extract the single float value from the resulting tensor
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