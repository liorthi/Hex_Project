import sys
from PySide6.QtWidgets import QApplication
from controller import GameController
from ui import HexWidget
from player import RandomAI, HumanPlayer, GreedyAI, HeuristicAI, NeuralAI
from board import RED, BLUE
from DatabaseHandler import DatabaseHandler
from Tournament import Tournament
from AiManager import AiManager, HexNet

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
import glob # for finding file names

# OPERATION MODES

def run_gui(database_path="board_database_100_000_games_greedy.json"):
    """
    Run the GUI application

    Args:
        database_path: Path to the board database file
    """

    # Setup players
    red_player = HumanPlayer()
    #blue_player = NeuralAI("hex_model_epoch_350.pth", BLUE)
    #blue_player = HeuristicAI(database_path, BLUE)
    #blue_player = GreedyAI(database_path, BLUE)

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


def create_database(num_games=1000, database_path="board_database_100_000_games_heuristic.json", save_games=True):
    """
    Create a board database by running multiple games

    Args:
        num_games: Number of games to run
        database_path: Path to load existing database (for GreedyAI)
        save_games: If true, save the game database
    """
    red = RandomAI()
    blue = NeuralAI("hex_model_epoch_350.pth", BLUE)
    #blue = GreedyAI(database_path, BLUE)

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
    if save_games:
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
                         model_save_name="hex_model",
                         save_interval=50):
    """
    Train a neural network on the board database

    Args:
        database_filename: JSON file containing board states and scores
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        train_split: Fraction of data to use for training (rest for testing)
        model_save_name: Name of model to save
        save_interval: Number of epochs to save model
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
    loss_fn = nn.MSELoss()
    optimizer = optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=0.01)

    train_loss_history = []
    test_loss_history = []

    epochs_left = epochs
    epochs_done = 0

    while True:
        epochs_now = min(save_interval, epochs_left)

        chunk_train, chunk_test = AiManager.train(
            net,
            train_loader,
            test_loader,
            device,
            epochs=epochs_now,
            learning_rate=learning_rate,
            loss_fn=loss_fn,
            optimizer=optimizer,
        )

        train_loss_history.extend(chunk_train)
        test_loss_history.extend(chunk_test)

        epochs_done += epochs_now
        torch.save(net.state_dict(), f"{model_save_name}_epoch_{epochs_done}.pth")
        print(f"Checkpoint saved at epoch {epochs_done}")

        epochs_left -= epochs_now
        if epochs_left <= 0:
            break

    # 6. Plot loss over epochs
    plt.figure()

    # Train loss (every epoch)
    epoch_axis = list(range(len(train_loss_history)))
    plt.plot(epoch_axis, train_loss_history, label="Train Loss")
    plt.plot(epoch_axis, test_loss_history, label="Test Loss")

    plt.title('Loss over epochs: train and test')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.show()

def evaluate_checkpoints(
        database_filename="board_database_100_000_games_heuristic.json",
        batch_size=64,
        train_split=0.7):
    """
    Evaluate a neural network on the board database
    Args:
        database_filename: JSON file containing board states and scores
        batch_size: Batch size for training
        train_split: Fraction of data to use for training (rest for testing)
    """

    # 0. Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load data with the same seed — identical split every time
    X, Y = DatabaseHandler.load_board_database_encoded(database_filename)
    dataset = TensorDataset(X, Y)

    train_size = int(len(dataset) * train_split)
    test_size = len(dataset) - train_size

    generator = torch.Generator()
    generator.manual_seed(42)  # same seed as training

    _, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 2. Find all checkpoint files
    checkpoints = sorted(glob.glob("hex_model_epoch_*.pth"))
    print(f"Found {len(checkpoints)} checkpoints\n")

    # 3. Evaluate each one
    results = []
    for path in checkpoints:
        net = HexNet().to(device)
        net.load_state_dict(torch.load(path, map_location=device))

        test_loss = AiManager.evaluate(net, test_loader, device)
        results.append((path, test_loss))
        print(f"{path:35s} | Test Loss: {test_loss:.5f}")

    # 4. Print the winner
    best_path, best_loss = min(results, key=lambda x: x[1])
    print(f"\nBest model: {best_path} with test loss {best_loss:.5f}")

    # 5. Plot
    epochs = [int(p.split("_epoch_")[1].replace(".pth", "")) for p, _ in results]
    losses = [loss for _, loss in results]

    plt.figure()
    plt.plot(epochs, losses, marker='o', label="Test Loss")
    plt.title("Test Loss per Checkpoint")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.grid(True)
    plt.show()

    return best_path

def main():
    """
    Main entry point - configure operation mode here

    Available modes:
        - "GUI": Run the graphical user interface
        - "CREATE_DATABASE": Generate board database from games
        - "TRAIN": Train neural network on existing database
        - "INFERENCE": Run prediction on a single board state
    """

    operation_mode = "GUI"  # Options: "GUI", "CREATE_DATABASE", "TRAIN", "EVALUATE_CHECKPOINTS"

    if operation_mode == "GUI":
        run_gui()

    elif operation_mode == "CREATE_DATABASE":
        create_database(
            num_games=1000,
            database_path="board_database_100_000_games_heuristic.json",
            save_games=False
        )

    elif operation_mode == "TRAIN":
        train_neural_network(
            database_filename="board_database_100_000_games_heuristic.json",
            epochs=1000,
            batch_size=64,
            learning_rate=0.01,
            train_split=0.7,
            model_save_name="hex_model",
            save_interval=50,
        )

    elif operation_mode == "EVALUATE_CHECKPOINTS":
        evaluate_checkpoints(
            database_filename="board_database_100_000_games_heuristic.json",
            batch_size = 64,
            train_split = 0.7
        )

    else:
        print(f"Unknown operation mode: {operation_mode}")
        print("Available modes: GUI, CREATE_DATABASE, TRAIN, INFERENCE")


if __name__ == "__main__":
    main()