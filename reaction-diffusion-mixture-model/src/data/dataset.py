import torch
from torch.utils.data import Dataset


class HeatDataset(Dataset):
    def __init__(self, data, mask, time):
        self.data = {'heat': data, 'mask':mask, 'time': time}

    def __len__(self):
        return len(self.data['heat'])

    def __getitem__(self, idx):
        sample = {
            'heat': torch.tensor(self.data['heat'][idx], dtype=torch.float32),
            'mask': torch.tensor(self.data['mask'][idx], dtype=torch.float32),
            'time': torch.tensor(self.data['time'][idx], dtype=torch.float32)
        }
        return sample
