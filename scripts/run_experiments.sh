#!/bin/bash
# ==============================================================================
# SECTION 10.1: REPRODUCIBILITY ARTIFACT & EXPERIMENTAL LOG
# Exact commands used for every experiment in the report.
# ==============================================================================

TRAIN_DATA="data/train_4k.bin"
VAL_DATA="data/val_4k.bin"
LOG_DIR="logs"
CHECKPOINT_DIR="checkpoints"

MAX_STEPS=500
WARMUP_STEPS=50

mkdir -p $LOG_DIR $CHECKPOINT_DIR

echo "=========================================================="
echo "Starting Section 7.1 Learning Rate Sweep (500 Steps Each)"
echo "=========================================================="

# Section 7.1 LR Sweep: Run 1 (LR = 1e-4) - Successful Convergence
python -m src.train \
  --train_data data/train_4k.bin \
  --val_data data/val_4k.bin \
  --out_dir checkpoints/lr_sweep_1e-4 \
  --vocab_size 4000 \
  --n_layers 4 \
  --d_model 512 \
  --n_heads 8 \
  --d_ff 1344 \
  --batch_size 32 \
  --max_steps 500 \
  --warmup_steps 50 \
  --lr 1e-4 \
  --eval_interval 10 \
  --save_interval 500 \
  --seed 42

# Section 7.1 LR Sweep: Run 2 (LR = 3e-4) - Successful Convergence
python -m src.train \
  --train_data data/train_4k.bin \
  --val_data data/val_4k.bin \
  --out_dir checkpoints/lr_sweep_3e-4 \
  --vocab_size 4000 \
  --n_layers 4 \
  --d_model 512 \
  --n_heads 8 \
  --d_ff 1344 \
  --batch_size 32 \
  --max_steps 500 \
  --warmup_steps 50 \
  --lr 3e-4 \
  --eval_interval 10 \
  --save_interval 500 \
  --seed 42

# Section 7.1 LR Sweep: Run 3 (LR = 1e-3) - Successful Convergence
python -m src.train \
  --train_data data/train_4k.bin \
  --val_data data/val_4k.bin \
  --out_dir checkpoints/lr_sweep_1e-3 \
  --vocab_size 4000 \
  --n_layers 4 \
  --d_model 512 \
  --n_heads 8 \
  --d_ff 1344 \
  --batch_size 32 \
  --max_steps 500 \
  --warmup_steps 50 \
  --lr 1e-3 \
  --eval_interval 10 \
  --save_interval 500 \
  --seed 42

# Section 7.1 LR Sweep: Run 4 (LR = 3e-3) - Successful Convergence
python -m src.train \
  --train_data data/train_4k.bin \
  --val_data data/val_4k.bin \
  --out_dir checkpoints/lr_sweep_3e-3 \
  --vocab_size 4000 \
  --n_layers 4 \
  --d_model 512 \
  --n_heads 8 \
  --d_ff 1344 \
  --batch_size 32 \
  --max_steps 500 \
  --warmup_steps 50 \
  --lr 3e-3 \
  --eval_interval 10 \
  --save_interval 500 \
  --seed 42

# Section 7.1 LR Sweep: Run 5 (LR = 1e-2) - Over-shooting
python -m src.train \
  --train_data data/train_4k.bin \
  --val_data data/val_4k.bin \
  --out_dir checkpoints/lr_sweep_1e-2 \
  --vocab_size 4000 \
  --n_layers 4 \
  --d_model 512 \
  --n_heads 8 \
  --d_ff 1344 \
  --batch_size 32 \
  --max_steps 500 \
  --warmup_steps 50 \
  --lr 1e-2 \
  --eval_interval 10 \
  --save_interval 500 \
  --seed 42
