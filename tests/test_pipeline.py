import sys
from pathlib import Path
import os
import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import MemmapDataset, get_dataloader


def test_data_pipeline():
    print("Testing dataset and dataloader...")

    #1. Create a dummy uint16 binary file (1000 sequential numbers)
    dummy_path = "test_dummy.bin"
    dummy_data = np.arange(1000, dtype=np.uint16)
    dummy_data.tofile(dummy_path)

    context_len = 32
    batch_sz = 4

    try:
        #2. Test MemmapDataset
        ds = MemmapDataset(dummy_path, context_length=context_len)
        print(f"Dataset length: {len(ds)}")
        assert len(ds) == 1000 - context_len, "Dataset length calculation is incorrect!"

        x, y = ds[0]
        print(f"Single x shape: {x.shape}, y shape: {y.shape}")
        assert x.dtype == torch.int64, "x tensor should be int64!"
        
        assert torch.equal(x[1:], y[:-1]), "y is not correctly shifted by 1 relative to x!"
        print("Dataset indexing and target shifting check passed!")

        #3. Test get_dataloader
        loader = get_dataloader(
            data_path=dummy_path,
            batch_size=batch_sz,
            context_length=context_len,
            shuffle=True,
            num_workers=0
        )

        batch_x, batch_y = next(iter(loader))
        print(f"Batch x shape: {batch_x.shape}, Batch y shape: {batch_y.shape}")
        assert batch_x.shape == (batch_sz, context_len), "Batch x shape mismatch!"
        assert batch_y.shape == (batch_sz, context_len), "Batch y shape mismatch!"

        assert torch.equal(batch_x[:, 1:], batch_y[:, :-1]), "Batch shift check failed!"
        print("Dataloader batching check passed!")

        print("\nAll dataset tests passed successfully!")

    finally:
        if "loader" in locals():
            del loader
        if "ds" in locals():
            del ds

        if os.path.exists(dummy_path):
            os.remove(dummy_path)


if __name__ == "__main__":
    test_data_pipeline()