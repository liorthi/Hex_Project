import torch
import torch.nn as nn
import torch.optim as optim

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