import torch
import torch.nn as nn
import torch.optim as optim
import time
from data import dataloaders
from model import MiniTransformer
from utils import set_seed, count_parameters


def train_model(model, train_loader, val_loader, epochs=20, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_accuracies = []

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            inputs, masks, labels = batch['input'].to(device), batch['mask'].to(device), batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(inputs, masks)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        val_acc = evaluate(model, val_loader, device)
        train_losses.append(total_loss / len(train_loader))
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch + 1}/{epochs} | Loss: {train_losses[-1]:.4f} | Val Acc: {val_acc:.4f}")

    total_time = (time.time() - start_time) / 60
    return train_losses, val_accuracies, total_time


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            inputs, masks, labels = batch['input'].to(device), batch['mask'].to(device), batch['label'].to(device)
            outputs = model(inputs, masks)
            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    return correct / total


if __name__ == "__main__":
    set_seed(42)
    # Model B
    train_l, val_l, test_l = dataloaders('train.csv', 'validation.csv', 'test.csv')
    model_b = MiniTransformer(num_heads=4, use_pe=True)
    train_model(model_b, train_l, val_l)
