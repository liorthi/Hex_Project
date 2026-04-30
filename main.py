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

def run_gui(database_path="board_database_100_000_games_heuristic.json"):
    """
    Run the GUI application

    Args:
        database_path: Path to the board database file
    """

    # Setup players
    red_player = HumanPlayer()
    #red_player = HeuristicAI(database_path, RED)

    blue_player = NeuralAI("hex_model_v2_epoch_40.pth", BLUE)
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
    blue = RandomAI()
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
                         early_stopping_patience=30,
                         save_interval=50):
    """
    Train a neural network on the board database with ReduceLROnPlateau scheduler

    Args:
        database_filename: JSON file containing board states and scores
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        train_split: Fraction of data to use for training (rest for testing)
        model_save_name: Name of model to save
        save_interval: Number of epochs between checkpoints
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

    # 3. Instantiate Model, Loss, Optimizer, and Scheduler
    net = HexNet().to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # ReduceLROnPlateau scheduler
    # Reduces learning rate when test loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',           # We want to minimize the loss
        factor=0.5,           # Reduce LR by half when plateau detected
        patience=10,          # Wait 10 epochs before reducing
        min_lr=1e-6          # Don't go below this learning rate
    )

    train_loss_history = []
    test_loss_history = []

    epochs_left = epochs
    epochs_done = 0


    train_loss_history, test_loss_history = AiManager.train(
        model=net,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=epochs,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        early_stopping_patience=early_stopping_patience,
        checkpoint_interval=save_interval,
        model_save_name=model_save_name
    )

    # 6. Plot loss over epochs
    plt.figure(figsize=(10, 6))

    # Train loss (every epoch)
    epoch_axis = list(range(1, len(train_loss_history) + 1))
    plt.plot(epoch_axis, train_loss_history, label="Train Loss", alpha=0.7)
    plt.plot(epoch_axis, test_loss_history, label="Test Loss", alpha=0.7)

    plt.title('Loss over epochs: train and test')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"\nTraining complete!")
    print(f"Final Train Loss: {train_loss_history[-1]:.6f}")
    print(f"Final Test Loss: {test_loss_history[-1]:.6f}")

def main():
    """
    Main entry point - configure operation mode here

    Available modes:
        - "GUI": Run the graphical user interface
        - "CREATE_DATABASE": Generate board database from games
        - "TRAIN": Train neural network on existing database
    """

    operation_mode = "TRAIN"  # Options: "GUI", "CREATE_DATABASE", "TRAIN"

    if operation_mode == "GUI":
        run_gui()

    elif operation_mode == "CREATE_DATABASE":
        create_database(
            num_games=100,
            database_path="board_database_100_000_games_heuristic.json",
            save_games=False
        )

    elif operation_mode == "TRAIN":
        train_neural_network(
            database_filename="board_database_100_000_games_heuristic.json",
            epochs=20,
            batch_size=64,
            learning_rate=0.01,
            train_split=0.7,
            model_save_name="hex_model_v2",
            early_stopping_patience=30,
            save_interval=40
        )

    else:
        print(f"Unknown operation mode: {operation_mode}")
        print("Available modes: GUI, CREATE_DATABASE, TRAIN, INFERENCE")


if __name__ == "__main__":
    main()