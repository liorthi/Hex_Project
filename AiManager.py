import torch
import torch.nn as nn

class AiManager:

    # Training
    @staticmethod
    def train(model, train_loader, test_loader, device, epochs=2000, learning_rate=0.001,
              loss_fn = None, optimizer = None):

        train_loss_history = []
        test_loss_history = []

        for epoch in range(epochs):
            model.train()
            total_loss = 0

            for batch_X, batch_Y in train_loader:
                batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)

                optimizer.zero_grad()
                # Forward pass
                y_pred = model(batch_X)
                # Calculate loss
                loss = loss_fn(y_pred, batch_Y)
                # Backward pass
                loss.backward()
                # Weight update
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            train_loss_history.append(avg_loss)

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