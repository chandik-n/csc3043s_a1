import os
import numpy as np

os.makedirs("data", exist_ok=True)

train_data = np.random.randint(0, 1000, size=(100000,), dtype=np.uint16)
val_data = np.random.randint(0, 1000, size=(20000,), dtype=np.uint16)

train_data.tofile("data/train.bin")
val_data.tofile("data/val.bin")

print("Created dummy data/train.bin and data/val.bin successfully.")