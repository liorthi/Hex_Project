import sys
from PySide6.QtWidgets import QApplication
from controller import GameController
from ui import HexWidget
from player import RandomAI, HumanPlayer, GreedyAI, HeuristicAI
from board import RED, BLUE
from DatabaseHandler import DatabaseHandler
from Tournament import Tournament


def main():
    # Players available:
    # - HumanPlayer()
    # - RandomAI()
    # - GreedyAI(database_path, color)
    # - HeuristicAI(database_path, color)

    # ==========================
    # == CREATE DATABASE CODE ==
    # ==========================
    """
    number_of_games = 1000
    database_path = r"board_database_100_000_games_greedy.json"

    red = RandomAI()
    blue = HeuristicAI(database_path, BLUE)

    tournament = Tournament(
        num_games=number_of_games,
        board_size=7,
        red_player_class=red,
        blue_player_class=blue,
        gamma=0.9
    )

    print(f"Running {number_of_games} games...")
    results, board_database, winners = tournament.run_multiple_games(verbose=False)

    print(f"Winners: {winners}")

    # Save the board database (main output)
    # DatabaseHandler.save_board_database(board_database, filename=f"board_database_{number_of_games}_games_heuristic.json")

    # Optionally save game metadata
    # DatabaseHandler.save_games_to_json(results, filename)

    # Print some statistics
    avg_moves = sum(r['total_moves'] for r in results) / len(results)
    print(f"\nAverage game length: {avg_moves:.1f} moves")
    print(f"Unique board states: {len(board_database)}")

    # Show example of a board state entry
    if board_database:
        example_key = list(board_database.keys())[0]
        score, count = board_database[example_key]
        print(f"\nExample board state:")
        print(f"  Key: {example_key[:30]}...")
        print(f"  Average score: {score:.4f}")
        print(f"  Times seen: {count}")
    """

    # ==============
    # == GUI CODE ==
    # ==============

    # Setup players
    database_path = r"board_database_100_000_games_greedy.json"
    #red_player = RandomAI()
    red_player = HumanPlayer()
    blue_player = HeuristicAI(database_path, BLUE)
    #blue_player = HumanPlayer()

    # Create Qt application
    app = QApplication(sys.argv)

    # Create controller (Model + Controller)
    controller = GameController(
        board_size=7,
        red_player=red_player,
        blue_player=blue_player
    )

    # Create UI (View)
    ui = HexWidget(controller)

    # Connect controller to UI
    controller.set_ui(ui)

    # Show UI
    ui.show()

    # Run application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()