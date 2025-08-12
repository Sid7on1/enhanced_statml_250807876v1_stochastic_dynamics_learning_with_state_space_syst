import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Constants
MODEL_PATH = "models/eye_tracking_model.pth"
DATA_PATH = "data/eye_tracking_data.csv"

# Configuration class
class Config:
    def __init__(self):
        self.model_path = MODEL_PATH
        self.data_path = DATA_PATH
        self.batch_size = 32
        self.learning_rate = 0.001
        self.num_epochs = 100
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = 42
        self.log_interval = 10
        self.val_split = 0.2
        self.early_stopping_patience = 5
        self.velocity_threshold = 0.2  # From research paper
        self.flow_theory_constant = 0.5  # From research paper


# Dataset class
class EyeTrackingDataset(Dataset):
    def __init__(self, data_path: str):
        self.data = pd.read_csv(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.data.iloc[idx]
        return {"features": np.array(row[2:]), "target": row[1]}

# Model class
class EyeTrackingModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(EyeTrackingModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Data loader function
def load_data(config: Config) -> Tuple[DataLoader, DataLoader]:
    dataset = EyeTrackingDataset(config.data_path)
    train_data, val_data = torch.utils.data.random_split(dataset,
                                                       [int(len(dataset)*(1-config.val_split)),
                                                        int(len(dataset)*config.val_split)])
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False)
    return train_loader, val_loader

# Model training function
def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, config: Config) -> nn.Module:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=config.early_stopping_patience)

    best_loss = float('inf')
    for epoch in range(config.num_epochs):
        model.train()
        for batch in train_loader:
            features = batch["features"].to(config.device)
            target = batch["target"].to(config.device)

            optimizer.zero_grad()
            output = model(features)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        val_loss = self._validate(model, val_loader, criterion)
        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), config.model_path)
            logger.info(f"Epoch {epoch+1}: Validation loss improved to {val_loss:.4f}. Model saved.")
        else:
            logger.info(f"Epoch {epoch+1}: Validation loss did not improve. Best loss remains at {best_loss:.4f}.")

    return EyeTrackingModel(config.model_path)

# Model validation function
def _validate(model: nn.Module, val_loader: DataLoader, criterion: nn.Module) -> float:
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            features = batch["features"].to(config.device)
            target = batch["target"].to(config.device)

            output = model(features)
            loss = criterion(output, target)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    return val_loss

# Main function
def main():
    config = Config()
    train_loader, val_loader = load_data(config)
    model = EyeTrackingModel(input_dim=train_loader.dataset.data.shape[1]-1, hidden_dim=256, output_dim=1)
    model = train_model(model, train_loader, val_loader, config)

if __name__ == "__main__":
    main()