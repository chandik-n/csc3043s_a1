import json
import os
import shutil
import subprocess
import sys
import numpy as np

TRAIN_DATA = "data/dummy_train.bin"
VAL_DATA = "data/dummy_val.bin"
FULL_DIR = "test_checkpoints/full"
RESUMED_DIR = "test_checkpoints/resumed"


def create_dummy_data():
    os.makedirs("data", exist_ok=True)
    data = np.random.randint(0, 1000, size=(300,), dtype=np.uint16)
    data.tofile(TRAIN_DATA)
    data.tofile(VAL_DATA)


def clean_dirs():
    for d in [FULL_DIR, RESUMED_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)


def run_command(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error executing command: {' '.join(cmd)}")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        sys.exit(1)


def read_logs(log_path):
    logs = {}
    epochs = {}
    with open(log_path, "r") as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                logs[entry["step"]] = entry["train_loss"]
                if "current_epoch" in entry:
                    epochs[entry["step"]] = entry["current_epoch"]
    return logs, epochs


def main():
    print("Creating dummy dataset (300 tokens = 10 batches/epoch to force 5 rollovers)...")
    create_dummy_data()
    clean_dirs()

    python_bin = sys.executable

    #1. Uninterrupted run: 50 steps
    print("\n=== 1. Running 50 steps uninterrupted ===")
    cmd_full = [
        python_bin,
        "-m",
        "src.train",
        "--train_data",
        TRAIN_DATA,
        "--val_data",
        VAL_DATA,
        "--batch_size",
        "16",
        "--context_length",
        "128",
        "--eval_interval",
        "10",
        "--eval_batches",
        "5",
        "--seed",
        "42",
        "--out_dir",
        FULL_DIR,
        "--max_steps",
        "50",
        "--save_interval",
        "50",
    ]
    run_command(cmd_full)

    #2. First 25 steps (saving checkpoint at step 25)
    print("=== 2. Running first 25 steps ===")
    cmd_part1 = [
        python_bin,
        "-m",
        "src.train",
        "--train_data",
        TRAIN_DATA,
        "--val_data",
        VAL_DATA,
        "--batch_size",
        "16",
        "--context_length",
        "128",
        "--eval_interval",
        "10",
        "--eval_batches",
        "5",
        "--seed",
        "42",
        "--out_dir",
        RESUMED_DIR,
        "--max_steps",
        "25",
        "--save_interval",
        "25",
    ]
    run_command(cmd_part1)

    #3. Resume from step 25 to 50
    print("=== 3. Resuming from step 25 to 50 ===")
    ckpt_25 = os.path.join(RESUMED_DIR, "checkpoint_step_25.pt")
    cmd_part2 = [
        python_bin,
        "-m",
        "src.train",
        "--train_data",
        TRAIN_DATA,
        "--val_data",
        VAL_DATA,
        "--batch_size",
        "16",
        "--context_length",
        "128",
        "--eval_interval",
        "10",
        "--eval_batches",
        "5",
        "--seed",
        "42",
        "--out_dir",
        RESUMED_DIR,
        "--max_steps",
        "50",
        "--save_interval",
        "25",
        "--resume",
        ckpt_25,
    ]
    run_command(cmd_part2)

    full_logs, full_epochs = read_logs(os.path.join(FULL_DIR, "train_log.jsonl"))
    resumed_logs, resumed_epochs = read_logs(os.path.join(RESUMED_DIR, "train_log.jsonl"))

    max_epoch = max(full_epochs.values()) if full_epochs else 0
    print(f"\nSANITY CHECK: Maximum epoch reached during run = {max_epoch}")
    if max_epoch == 0:
        print("FAILURE: Test did NOT cross epoch boundaries! Adjust dummy data size.")
        sys.exit(1)

    print("\nSTEP-BY-STEP COMPARISON (STEPS 26 - 50)")
    print(f"{'Step':<6} | {'Epoch':<6} | {'Full Loss':<12} | {'Resumed Loss':<14} | {'Match?'}")

    all_matched = True
    for step in range(26, 51):
        f_loss = full_logs.get(step)
        r_loss = resumed_logs.get(step)
        ep = full_epochs.get(step, "?")

        match = (f_loss is not None) and (r_loss is not None) and (abs(f_loss - r_loss) < 1e-5)
        if not match:
            all_matched = False

        status = "OK" if match else "MISMATCH"
        print(f"{step:<6} | {str(ep):<6} | {str(f_loss):<12} | {str(r_loss):<14} | {status}")

    if all_matched:
        print(
            f"\nSUCCESS: Resumed trajectory matches uninterrupted trajectory bitwise across {max_epoch + 1} epochs!"
        )
    else:
        print("\nFAILURE: Mismatch detected between uninterrupted and resumed runs.")
        sys.exit(1)


if __name__ == "__main__":
    main()