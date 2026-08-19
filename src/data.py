import numpy as np
import torch
from torch.utils.data import Dataset


class MemmapDataset(Dataset):
    def __init__(self, data_path: str, context_length: int = 256, dtype=np.uint16):
        self.data = np.memmap(data_path, dtype=dtype, mode="r")
        self.context_length = context_length
        self.num_samples = len(self.data) - context_length

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx):
        if isinstance(idx, (list, np.ndarray, torch.Tensor)):
            idx_arr = np.asarray(idx)
            offsets = idx_arr[:, None] + np.arange(self.context_length + 1)
            chunk = self.data[offsets].astype(np.int64)

            x = torch.from_numpy(chunk[:, :-1]).contiguous()
            y = torch.from_numpy(chunk[:, 1:]).contiguous()
            return x, y
        else:
            chunk = self.data[idx : idx + self.context_length + 1].astype(np.int64)
            x = torch.from_numpy(chunk[:-1]).contiguous()
            y = torch.from_numpy(chunk[1:]).contiguous()
            return x, y