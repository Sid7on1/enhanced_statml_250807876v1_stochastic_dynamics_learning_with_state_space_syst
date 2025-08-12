import logging
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants and configuration
CONFIG = {
    'model_path': 'models',
    'data_path': 'data',
    'batch_size': 32,
    'epochs': 10,
    'learning_rate': 0.001,
    'log_interval': 100,
}

class StateSpaceSystemDataset(Dataset):
    def __init__(self, data: pd.DataFrame, seq_len: int):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx: int):
        seq = self.data.iloc[idx:idx + self.seq_len]
        x = seq[['x', 'y']].values
        y = seq['target'].values
        return {
            'x': torch.tensor(x, dtype=torch.float32),
            'y': torch.tensor(y, dtype=torch.float32),
        }

class StateSpaceSystemModel(nn.Module):
    def __init__(self):
        super(StateSpaceSystemModel, self).__init__()
        self.fc1 = nn.Linear(2, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class TrainingPipeline:
    def __init__(self, config: Dict):
        self.config = config
        self.model = StateSpaceSystemModel()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        data = pd.read_csv(os.path.join(self.config['data_path'], 'data.csv'))
        train_data, val_data = data.split(test_size=0.2, random_state=42)
        return train_data, val_data

    def create_dataset(self, data: pd.DataFrame, seq_len: int) -> StateSpaceSystemDataset:
        return StateSpaceSystemDataset(data, seq_len)

    def train(self, train_data: pd.DataFrame, val_data: pd.DataFrame):
        train_dataset = self.create_dataset(train_data, self.config['seq_len'])
        val_dataset = self.create_dataset(val_data, self.config['seq_len'])
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'], shuffle=False)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        criterion = nn.MSELoss()

        for epoch in range(self.config['epochs']):
            self.model.train()
            total_loss = 0
            for batch in train_loader:
                x, y = batch['x'].to(self.device), batch['y'].to(self.device)
                optimizer.zero_grad()
                outputs = self.model(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            self.model.eval()
            with torch.no_grad():
                val_loss = 0
                for batch in val_loader:
                    x, y = batch['x'].to(self.device), batch['y'].to(self.device)
                    outputs = self.model(x)
                    loss = criterion(outputs, y)
                    val_loss += loss.item()
            logging.info(f'Epoch {epoch+1}, Train Loss: {total_loss / len(train_loader)}, Val Loss: {val_loss / len(val_loader)}')
            if (epoch + 1) % self.config['log_interval'] == 0:
                torch.save(self.model.state_dict(), os.path.join(self.config['model_path'], f'model_{epoch+1}.pth'))

    def evaluate(self, val_data: pd.DataFrame):
        val_dataset = self.create_dataset(val_data, self.config['seq_len'])
        val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'], shuffle=False)
        self.model.eval()
        with torch.no_grad():
            total_loss = 0
            for batch in val_loader:
                x, y = batch['x'].to(self.device), batch['y'].to(self.device)
                outputs = self.model(x)
                loss = nn.MSELoss()(outputs, y)
                total_loss += loss.item()
        logging.info(f'Validation Loss: {total_loss / len(val_loader)}')

def main():
    config = CONFIG.copy()
    config['seq_len'] = 10
    pipeline = TrainingPipeline(config)
    train_data, val_data = pipeline.load_data()
    pipeline.train(train_data, val_data)
    pipeline.evaluate(val_data)

if __name__ == '__main__':
    main()