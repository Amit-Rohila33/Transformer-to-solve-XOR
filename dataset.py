import numpy as np
import torch
from torch.utils.data import Dataset

class XORParityDataset(Dataset):
    def __init__(self, filepath):
        self.data = np.load(filepath, allow_pickle=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sequence, xor_result = self.data[index]
        sequence_tensor = torch.tensor(sequence)
        xor_result_tensor = torch.tensor(xor_result, dtype=torch.float32)
        return sequence_tensor, xor_result_tensor
