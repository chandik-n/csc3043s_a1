import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class MemmapDataset(Dataset):
    def __init__(self, data_path: str, context_length: int = 256, dtype=np.uint16):
        self.data = np.memmap(data_path, dtype=dtype, mode="r")
        self.context_length = context_length
        self.num_samples = len(self.data) - context_length

        if self.num_samples <= 0:
            raise ValueError(
                f"Data length ({len(self.data)}) must be greater than context_length ({context_length})"
            )

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
            # Single index fallback
            chunk = self.data[idx : idx + self.context_length + 1].astype(np.int64)
            x = torch.from_numpy(chunk[:-1]).contiguous()
            y = torch.from_numpy(chunk[1:]).contiguous()
            return x, y


def get_dataloader(
    data_path: str,
    batch_size: int,
    context_length: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    dataset = MemmapDataset(data_path, context_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )