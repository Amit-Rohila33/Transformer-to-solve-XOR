import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
import wandb

from transformer_model import TransformerModel
from dataset import XORParityDataset

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(dataloader.dataset)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
    return running_loss / len(dataloader.dataset)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    args = parser.parse_args()

    wandb.init(project="xor-parity", config=args)

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Load the dataset
    dataset = XORParityDataset("datasets/fixed_length_dataset.npy")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Define the model
    model = TransformerModel(input_dim=2, hidden_dim=32, output_dim=1, num_layers=2)
    model.to(args.device)

    # Define the loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, dataloader, criterion, optimizer, args.device)
        val_loss = evaluate(model, dataloader, criterion, args.device)

        wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        print(f"Epoch: {epoch}/{args.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "xor_model.pt")
    wandb.save("xor_model.pt")

if __name__ == "__main__":
    main()
