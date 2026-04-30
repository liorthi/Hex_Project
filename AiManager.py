import torch
import torch.nn as nn

class AiManager:

    # Training
    @staticmethod
    def train(model, train_loader, test_loader, device, epochs=2000, learning_rate=0.001,
          loss_fn=None, optimizer=None):
        train_loss_history = []
        test_loss_history = []

        for epoch in range(epochs):
            # --- Training ---
            model.train()
            total_train_loss = 0

            for batch_X, batch_Y in train_loader:
                batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)

                optimizer.zero_grad()
                y_pred = model(batch_X)
                loss = loss_fn(y_pred, batch_Y)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            train_loss_history.append(avg_train_loss)

            # --- Evaluation ---
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

            print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Test Loss: {avg_test_loss:.6f}")

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
        self.layer1 = nn.Linear(7*7*3, 180)
        self.layer2 = nn.Linear(180, 64)
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
        x = torch.sigmoid(x)

        return x