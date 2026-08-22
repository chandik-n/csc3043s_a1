# CSC3043S Assignment 1 — Language Model Training
# Student number: NDXCHA041

# === Overview ===
# A from-scratch byte-level BPE tokenizer, Transformer language model (RMSNorm, SwiGLU,
# RoPE, QK-norm, pre-norm blocks), training loop, and inference pipeline (KV cache,
# temperature/top-k/top-p sampling), trained on TinyStories V2.
#
# Note on compute environment: all GPU experiments were run on free-tier Google Colab
# (T4 GPU) rather than the assigned Dataspires L4 allocation, due to persistent access
# issues with Dataspires. This is disclosed throughout the report where relevant

# === Installation ===
# python -m venv venv
# source venv/bin/activate        # Windows: venv\Scripts\activate
# pip install -r requirements.txt
#
# Versions used:
# - Python 3.13.1
# - PyTorch 2.11.0+cu128 (Colab, GPU experiments) / CPU build (local, plotting & report work)
# - CUDA 13.0 (Colab T4)

# === Repository Layout ===
# src/                          # Source code
#   ├── tokenizer.py            # BPE tokenizer
#   ├── model.py                # TransformerLM
#   ├── train.py                # Training loop with resume
#   ├── evaluate.py             # Perplexity + BPC
#   ├── generate.py             # Sampling (temperature, top-k, top-p)
#   └── data.py                 # Memory-mapped data loader
# scripts/                      # Experiment scripts
#   ├── encode_corpus.py
#   ├── vocab_study.py
#   ├── run_experiments.sh      # Exact commands for every reported run
#   └── make_plots.py
# tests/                        # Tests (tokenizer, KV cache, model, resume)
# logs/                         # Results (.txt) and vocab study (.csv)
# configs/                      
# data/                         # Tokenizer vocab/merges (checkpoints and .bin not committed)

# === Reproducing the Results ===
# All exact commands used for every reported run are in scripts/run_experiments.sh.

# 1. Tokenizer training and vocab study:
python scripts/encode_corpus.py
python scripts/vocab_study.py
# Output: logs/vocab_study_results.csv (Q3), tokenizer files in data/.

# 2. LR sweep, ablations, vocabulary-size run (500 steps each, reduced from the
#    spec's standard 5,000 due to compute/time constraints):
python -m src.train --lr 3e-3 --max_steps 500 ...   # see run_experiments.sh for full grid

# 3. Final model (20,000 steps, resumed after Colab disconnect at step ~9945
#    from checkpoint_step_8000.pt):
python -m src.train --max_steps 20000 --resume checkpoints/final_model/checkpoint_step_8000.pt ...

# 4. Evaluation:
python -m src.evaluate --checkpoint checkpoints/final_model/checkpoint_step_20000.pt \
    --data data/val_holdout_4k.bin   # -> logs/val_holdout_results.txt
python -m src.evaluate --checkpoint checkpoints/final_model/checkpoint_step_20000.pt \
    --data data/test_4k.bin          # -> logs/test_set_results.txt (touched once)

# 5. Generation samples:
python -m src.generate ...   # -> samples.txt

# 6. Plots:
python scripts/make_plots.py

# === Where Each Report Figure Comes From ===
# | Figure | Source |
# |--------|--------|
# | Vocabulary compression curve (Q3) | logs/vocab_study_results.csv |
# | LR sweep plot (Q10) | logs/*.txt per-LR final validation losses |
#
# Note: per-step training curves (JSONL logs) for the LR sweep and ablation runs were
# lost to a Colab session disconnect before being committed to the repository. The figures
# above show final validation values only, not full training trajectories — see the
# report's Data Availability Note.

# === Key Results ===
# | Metric | Value |
# |--------|-------|
# | Best Learning Rate | 3e-3 |
# | Final Test Loss | 1.5618 |
# | Final Test BPC | 0.5661 |
# | Model Parameters | 16,552,960 |

# === Tests ===
# pytest tests/
#
# tests/test_resume.py verifies bitwise-identical loss trajectories between an
# uninterrupted run and a stop/resume run (steps 26–50 match exactly across 25 steps),
# confirming the checkpoint/resume mechanism used for the final model's disconnect recovery.

# === Known Limitations / Disclosures ===
# See the report's introduction and Data Availability Note for full details on:
# - Colab-vs-Dataspires substitution
# - 500-vs-5000 standard step count reduction
# - Late discovery and correction of validation/test split leakage (§8, Q18)
# - Lost per-step JSONL logs

# === AI Usage ===
# See AGENTS.md and the AI usage declaration in the report.