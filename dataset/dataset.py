import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, data):
      self.data = data

    def __len__(self):
      return len(self.data)

    def __getitem__(self, index):
      return {'ecg': torch.tensor(self.data[index]['ecg']),
              # 'spectrogram': torch.tensor(self.data[index]['spectrogram']),
              'label': torch.tensor([self.data[index]['label']])}