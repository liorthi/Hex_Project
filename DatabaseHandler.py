import json
from pathlib import Path

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


