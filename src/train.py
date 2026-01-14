
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import AutoEncoder

def train_model(features, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoEncoder(
        config["model"]["input_dim"],
        config["model"]["hidden_dim"]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["model"]["learning_rate"])
    criterion = nn.MSELoss()

    dataset = TensorDataset(torch.tensor(features.values, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    for epoch in range(config["model"]["epochs"]):
        total_loss = 0
        for batch in loader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: loss={total_loss:.6f}")

    return model
