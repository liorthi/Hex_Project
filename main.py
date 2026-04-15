import sys
from PySide6.QtWidgets import QApplication
from controller import GameController
from ui import HexWidget
from player import RandomAI, HumanPlayer, GreedyAI, HeuristicAI, HexNet
from board import RED, BLUE
from DatabaseHandler import DatabaseHandler
from Tournament import Tournament
from AiManager import AiManager

import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt

# OPERATION MODES

def run_gui(database_path="board_database_100_000_games_greedy.json"):
    """
    Run the GUI application

    Args:
        database_path: Path to the board database file
    """
    # Setup players
    red_player = HumanPlayer()
    blue_player = HeuristicAI(database_path, BLUE)

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


def create_database(num_games=1000, database_path="board_database_100_000_games_greedy.json"):
    """
    Create a board database by running multiple games

    Args:
        num_games: Number of games to run
        database_path: Path to load existing database (for GreedyAI)
    """
    red = RandomAI()
    blue = GreedyAI(database_path, BLUE)

    tournament = Tournament(
        num_games=num_games,
        board_size=7,
        red_player_class=red,
        blue_player_class=blue,
        gamma=0.9
    )

    print(f"Running {num_games} games...")
    results, board_database, winners = tournament.run_multiple_games(verbose=False)

    print(f"Winners: {winners}")

    # Save the board database (main output)
    DatabaseHandler.save_board_database(
        board_database,
        filename=f"board_database_{num_games}_games_heuristic.json"
    )

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


def train_neural_network(database_filename="board_database_100_000_games_heuristic.json",
                         epochs=50,
                         batch_size=64,
                         learning_rate=0.001,
                         train_split=0.7,
                         model_save_path="hex_model.pth"):
    """
    Train a neural network on the board database

    Args:
        database_filename: JSON file containing board states and scores
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        train_split: Fraction of data to use for training (rest for testing)
        model_save_path: Path to save the trained model
    """
    # 0. Choose device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Device selected: {device}")

    # 1. Prepare Data
    X, Y = DatabaseHandler.load_board_database_encoded(database_filename)

    # 2. Configure dataloader and partition into train and test sets
    dataset = TensorDataset(X, Y)

    train_size = int(len(dataset) * train_split)
    test_size = len(dataset) - train_size

    generator = torch.Generator()
    generator.manual_seed(42)

    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size],
        generator=generator
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 3. Instantiate Model
    net = HexNet().to(device)

    # 4. Execute Training
    train_loss_history, test_loss_history = AiManager.train(
        net,
        train_loader,
        test_loader,
        device,
        epochs=epochs,
        learning_rate=learning_rate
    )

    # 5. Save the result
    torch.save(net.state_dict(), model_save_path)
    print(f"\nModel saved to {model_save_path}")

    # 6. Plot loss over epochs
    plt.figure()

    # Train loss (every epoch)
    epoch_axis = list(range(len(train_loss_history)))
    plt.plot(epoch_axis, train_loss_history, label="Train Loss")

    # Test loss (every 50 epochs)
    eval_axis = list(range(0, len(train_loss_history), 50))
    plt.plot(eval_axis, test_loss_history, label="Test Loss")

    plt.title('Loss over epochs: train and test')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.show()


def run_inference(model_path="hex_model.pth",
                  board_str="[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]"):
    """
    Run inference on a single board state

    Args:
        model_path: Path to the trained model
        board_str: String representation of the board state
    """
    device = torch.device("cpu")

    # Load the model
    model = DatabaseHandler.load_network(model_path, device, HexNet)

    # Predict score
    score = DatabaseHandler.predict_score(model, board_str, device)

    print(f'Board: {board_str}')
    print(f'Predicted Score: {score:.4f}')

    return score


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """
    Main entry point - configure operation mode here

    Available modes:
        - "GUI": Run the graphical user interface
        - "CREATE_DATABASE": Generate board database from games
        - "TRAIN": Train neural network on existing database
        - "INFERENCE": Run prediction on a single board state
    """

    operation_mode = "TRAIN"  # Options: "GUI", "CREATE_DATABASE", "TRAIN", "INFERENCE"

    if operation_mode == "GUI":
        run_gui(database_path="board_database_100_000_games_greedy.json")

    elif operation_mode == "CREATE_DATABASE":
        create_database(
            num_games=1000,
            database_path="board_database_100_000_games_greedy.json"
        )

    elif operation_mode == "TRAIN":
        train_neural_network(
            database_filename="board_database_100_000_games_heuristic.json",
            epochs=50,
            batch_size=64,
            learning_rate=0.001,
            train_split=0.7,
            model_save_path="hex_model.pth"
        )

    elif operation_mode == "INFERENCE":
        # Example board state (empty 7x7 board)
        board = "[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]"
        run_inference(
            model_path="hex_model_50_epochs.pth",
            board_str=board
        )

    else:
        print(f"Unknown operation mode: {operation_mode}")
        print("Available modes: GUI, CREATE_DATABASE, TRAIN, INFERENCE")


if __name__ == "__main__":
    main()