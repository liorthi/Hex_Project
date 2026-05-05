import torch
import torch.nn as nn
import torch.optim as optim

class AiManager:

    # Training
    @staticmethod
    def train(model,
              train_loader,
              test_loader,
              device,
              epochs=50,
              loss_fn=None,
              optimizer=None,
              scheduler=None,
              early_stopping_patience=30,
              checkpoint_interval=50,
              model_save_name="hex_model"):

        train_loss_history = []
        test_loss_history = []

        best_test_loss = float('inf')
        best_model_state = None
        epochs_without_improvement = 0

        print("\nStarting Training Loop...\n")

        for epoch in range(epochs):
            # ===================== TRAIN =====================
            model.train()
            total_train_loss = 0

            for batch_X, batch_Y in train_loader:
                batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)

                optimizer.zero_grad()
                y_pred = model(batch_X)
                loss = loss_fn(y_pred, batch_Y)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping to prevent exploding gradients

                optimizer.step()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            train_loss_history.append(avg_train_loss)

            # ===================== TEST =====================
            model.eval()
            total_test_loss = 0

            with torch.no_grad():
                for batch_X, batch_Y in test_loader:
                    batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
                    y_pred = model(batch_X)
                    loss = loss_fn(y_pred, batch_Y)
                    total_test_loss += loss.item()

            avg_test_loss = total_test_loss / len(test_loader)
            test_loss_history.append(avg_test_loss)

            # ===================== SCHEDULER =====================
            if scheduler is not None:
                scheduler.step(avg_test_loss)

            current_lr = optimizer.param_groups[0]['lr']

            print(f"Epoch {epoch + 1}/{epochs} | "
                  f"Train Loss: {avg_train_loss:.9e} | "
                  f"Test Loss: {avg_test_loss:.9e} | "
                  f"LR: {current_lr:.9e}", end='')

            # ===================== EARLY STOPPING =====================
            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                best_model_state = model.state_dict()  # save best weights
                epochs_without_improvement = 0
                print(" New best model found")
            else:
                epochs_without_improvement += 1
                print()

            if epochs_without_improvement >= early_stopping_patience:
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                break

            # ===================== CHECKPOINT =====================
            if (epoch + 1) % checkpoint_interval == 0:
                checkpoint_path = f"{model_save_name}_epoch_{epoch + 1}.pth"
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}")

        # ===================== RESTORE BEST MODEL =====================
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print("\n Best model weights restored")

        return train_loss_history, test_loss_history


    # Evaluation
    @staticmethod
    def evaluate(model, loader, device):
        model.eval()  # Set model to evaluation mode
        loss_fn = nn.MSELoss()
        total_loss = 0

        with torch.no_grad():  # Disable gradient calculation for efficiency
            for batch_X, batch_Y in loader:
                batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
                predictions = model(batch_X)
                loss = loss_fn(predictions, batch_Y)
                total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        model.train()  # Reset to training mode
        return avg_loss
    

class HexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(7*7, 128)
        self.layer2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        # Layer 1
        x = self.layer1(x)
        x = self.leaky_relu(x)

        # Layer 2
        x = self.layer2(x)
        x = self.leaky_relu(x)

        # Output layer
        x = self.output(x)
        x = torch.tanh(x)

        return x