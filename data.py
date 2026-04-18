import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class TransformerDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv('test.csv')

        token_cols = []
        mask_cols = []

        for i in range(1, 21):
            token_cols.append(f'token_{i:02d}')
            mask_cols.append(f'mask_{i:02d}')

        self.tokens = torch.tensor(df[token_cols].values, dtype=torch.long)
        self.masks = torch.tensor(df[mask_cols].values, dtype=torch.float32)
        self.labels = torch.tensor(df['label'].values, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input': self.tokens[idx],
            'mask': self.masks[idx],
            'label': self.labels[idx]
        }


def dataloaders(train_path, val_path, test_path, batch_size = 32):

  train_ds = TransformerDataset(train_path)
  val_ds = TransformerDataset(val_path)
  test_ds = TransformerDataset(test_path)

  train_loader = DataLoader(train_ds, batch_size = batch_size, shuffle = True)
  val_loader = DataLoader(val_ds, batch_size = batch_size, shuffle = False)
  test_loader = DataLoader(test_ds, batch_size = batch_size, shuffle = False)

  return train_loader, val_loader, test_loader
