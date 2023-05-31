import argparse
from sklearn.model_selection import ParameterGrid

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()

    grid = {
        "batch_size": [32, 64],
        "epochs": [10, 20],
        "lr": [0.001, 0.01]
    }

    best_val_loss = float("inf")
    best_params = None

    for params in ParameterGrid(grid):
        wandb.init(project="xor-parity", config=params)

        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)

        # Load the dataset
        dataset = XORParityDataset("datasets/fixed_length_dataset.npy")
        dataloader = DataLoader(dataset, batch_size=params["batch_size"], shuffle=True)

        # Define the model
        model = TransformerModel(input_dim=2, hidden_dim=32, output_dim=1, num_layers=2)
        model.to(args.device)

        # Define the loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=params["lr"])

        # Training loop
        for epoch in range(1, params["epochs"] + 1):
            train_loss = train(model, dataloader, criterion, optimizer, args.device)
            val_loss = evaluate(model, dataloader, criterion, args.device)

            wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

            print(f"Epoch: {epoch}/{params['epochs']}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = params

    print(f"Best Hyperparameters: {best_params}")

if __name__ == "__main__":
    main()
